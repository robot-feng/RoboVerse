"""Prepare a reviewable AgiBot G1 URDF/MJCF asset from pinned Genie Sim assets.

The output retains source inertias, drive gains, and authored USD collision geometry.
It is intended for cross-backend validation, not production robot control.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import tempfile
import xml.etree.ElementTree as ET
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

REVISION = "a0813aad7c16165daffbe6c2737754e0344809ee"
_MIN_BACKEND_INERTIA_EIGENVALUE = 1.01e-6


def write_manifest(source: Path) -> Path:
    """Obtain checksums from the official pinned revision, not from a mirror."""
    from huggingface_hub import HfApi

    info = HfApi(endpoint="https://huggingface.co", token=False).dataset_info(
        "agibot-world/GenieSimAssets", revision=REVISION, files_metadata=True, token=False
    )
    if info.sha != REVISION or not info.siblings:
        raise ValueError("Official source did not return the pinned asset revision")
    files = []
    for item in info.siblings:
        if item.size is None or not item.blob_id:
            raise ValueError(f"Missing official checksum: {item.rfilename}")
        record = {"rfilename": item.rfilename, "blobId": item.blob_id, "size": item.size}
        if item.lfs is not None:
            record["lfs"] = {"sha256": item.lfs.sha256}
        files.append(record)
    source.mkdir(parents=True, exist_ok=True)
    output = source / "roboverse_source_manifest.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump({"repo": "agibot-world/GenieSimAssets", "revision": REVISION, "files": files}, stream, indent=2)
        stream.write("\n")
    return output


def _hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    """Return an installed distribution version for derived-asset provenance."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _source_drive_parameters(
    robot_usd: Path, physics_usd: Path | None = None
) -> dict[str, dict[str, float | None]]:
    """Read composed joint drives from the base physics layer and robot overrides."""
    from pxr import Sdf

    result: dict[str, dict[str, float | None]] = {}
    layers = []
    if physics_usd is not None and Path(physics_usd).is_file():
        layers.append((Path(physics_usd), "/genie/joints"))
    layers.append((Path(robot_usd), "/G01/joints"))
    for layer_path, joints_path in layers:
        layer = Sdf.Layer.FindOrOpen(str(layer_path.resolve()))
        if layer is None:
            raise ValueError(f"Cannot open pinned Genie Sim G1 drive USD: {layer_path}")
        joints = layer.GetPrimAtPath(joints_path)
        if joints is None:
            if layer_path == Path(robot_usd):
                raise ValueError(f"Pinned G1 robot USD has no {joints_path} overlay: {robot_usd}")
            raise ValueError(f"Pinned G1 physics USD has no {joints_path} prim: {layer_path}")
        for joint in joints.nameChildren:
            prefix = next(
                (
                    candidate
                    for candidate in ("angular", "linear")
                    if joint.properties.get(f"drive:{candidate}:physics:stiffness")
                    or joint.properties.get(f"drive:{candidate}:physics:damping")
                ),
                None,
            )
            if prefix is None:
                continue
            stiffness_attr = joint.properties.get(f"drive:{prefix}:physics:stiffness")
            damping_attr = joint.properties.get(f"drive:{prefix}:physics:damping")
            values = result.setdefault(
                joint.name, {"stiffness": 0.0, "damping": 0.0, "effort_limit_sim": None}
            )
            unit_scale = 180.0 / np.pi if prefix == "angular" else 1.0
            if stiffness_attr is not None and stiffness_attr.default is not None:
                values["stiffness"] = float(stiffness_attr.default) * unit_scale
            if damping_attr is not None and damping_attr.default is not None:
                values["damping"] = float(damping_attr.default) * unit_scale
            max_force_attr = joint.properties.get(f"drive:{prefix}:physics:maxForce")
            if max_force_attr is not None and max_force_attr.default is not None:
                max_force = float(max_force_attr.default)
                values["effort_limit_sim"] = max_force if np.isfinite(max_force) and max_force > 0 else None
            if (
                not np.isfinite(values["stiffness"])
                or not np.isfinite(values["damping"])
                or values["stiffness"] < 0
                or values["damping"] < 0
            ):
                raise ValueError(f"Pinned G1 USD has invalid drive parameters for {joint.name}")
    if not result:
        raise ValueError(f"Pinned G1 USD contains no authored joint drives: {robot_usd}")
    return result


def _replace_mjcf_actuators(
    mjcf_path: Path,
    drive_parameters: dict[str, dict[str, float | None]],
    *,
    active_joint_names: set[str] | None = None,
) -> None:
    """Write one MJCF position actuator per joint with the source USD gains."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    old_actuators = root.findall("actuator")
    for actuator in old_actuators:
        root.remove(actuator)
    actuators = ET.SubElement(root, "actuator")
    for index in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        if not name:
            raise ValueError("G1 MJCF contains an unnamed joint")
        if active_joint_names is not None and name not in active_joint_names:
            drive = {"stiffness": 0.0, "damping": 0.0, "effort_limit_sim": None}
        else:
            drive = drive_parameters.get(name, {"stiffness": 0.0, "damping": 0.0, "effort_limit_sim": None})
        actuator = ET.SubElement(
            actuators,
            "position",
            name=name,
            joint=name,
            kp=f"{drive['stiffness']:.12g}",
            kv=f"{drive['damping']:.12g}",
        )
        effort = drive["effort_limit_sim"]
        if effort is not None:
            actuator.set("forcerange", f"{-effort:.12g} {effort:.12g}")
    tree.write(mjcf_path, encoding="utf-8", xml_declaration=True)
    mujoco.MjModel.from_xml_path(str(mjcf_path))


def add_source_drives_to_prepared_asset(base_asset: Path, robot_usd: Path, output: Path) -> None:
    """Create a versioned G1 asset retaining source USD joint-drive parameters."""
    base_asset, robot_usd, output = Path(base_asset).resolve(), Path(robot_usd).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    provenance_path = base_asset / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected_sha = provenance.get("robot_usd_sha256")
    if (
        provenance.get("source_revision") != REVISION
        or provenance.get("robot_variant") != "G1_omnipicker"
        or expected_sha != _hash(robot_usd)
    ):
        raise ValueError("Source USD does not match the pinned prepared G1 asset provenance")
    physics_usd = robot_usd.parent / "configuration" / "robot_physics.usd"
    drive_parameters = _source_drive_parameters(robot_usd, physics_usd if physics_usd.is_file() else None)

    try:
        from roboverse_pack.tasks.icl.fruit_sorting import CONTROL_JOINTS
    except ImportError:
        CONTROL_JOINTS = ()
    missing = set(CONTROL_JOINTS) - set(drive_parameters)
    if missing:
        raise ValueError(f"Pinned G1 USD is missing controlled-joint drives: {sorted(missing)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    active_joints = set(CONTROL_JOINTS)
    with tempfile.TemporaryDirectory(prefix="g1-source-drives-", dir=output.parent) as temporary:
        dest = Path(temporary) / output.name
        shutil.copytree(base_asset, dest)
        mjcf_path = dest / "G1_omnipicker.xml"
        _replace_mjcf_actuators(mjcf_path, drive_parameters, active_joint_names=active_joints)
        updated = json.loads((dest / "provenance.json").read_text(encoding="utf-8"))
        updated["drive_source"] = {
            "repository": "https://huggingface.co/datasets/agibot-world/GenieSimAssets",
            "revision": REVISION,
            "robot_usd": "robot/G1_omnipicker/robot.usda",
            "robot_usd_sha256": expected_sha,
            "units": "angular stiffness and damping converted from per-degree to per-radian",
            "parameters": drive_parameters,
            "active_joint_names": sorted(active_joints) if active_joints is not None else None,
            "passive_joint_policy": (
                "zero gains on mimic/closed-loop followers to avoid redundant actuation"
                if active_joints is not None
                else None
            ),
        }
        updated["physics"] = (
            "Source URDF inertias retained except provenance-listed positive-definite inertia corrections. "
            "The base asset's collision geometry is retained. MJCF and RoboVerse actuator configuration preserve "
            "pinned G1 USD drive gains and authored force limits on policy-controlled joints; mimic/closed-loop "
            "followers use zero-gain observation actuators to avoid redundant drives. Cross-backend dynamics "
            "require runtime validation."
        )
        updated["prepared_mjcf_sha256"] = _hash(mjcf_path)
        updated["drive_mjcf_sha256"] = _hash(mjcf_path)
        (dest / "provenance.json").write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
        shutil.move(str(dest), str(output))


def align_base_collision_with_visual_floor(base_asset: Path, output: Path) -> float:
    """Create a versioned G1 asset whose base collision bottom matches its visible chassis."""
    import mujoco

    base_asset, output = Path(base_asset).resolve(), Path(output).resolve()
    provenance_path = base_asset / "provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(f"Prepared G1 asset has no provenance: {base_asset}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("format") != "roboverse.icl.geniesim_g1_robot.v4":
        raise ValueError("Ground alignment requires a prepared GenieSim G1 v4 asset")
    if output.exists():
        raise FileExistsError(output)

    urdf_root = ET.parse(base_asset / "G1_omnipicker.urdf").getroot()
    base_link = urdf_root.find("link[@name='base_link']")
    if base_link is None:
        raise ValueError("Prepared G1 asset has no base_link")

    def origin_transform(element: ET.Element) -> tuple[np.ndarray, np.ndarray]:
        origin = element.find("origin")
        xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        if xyz.shape != (3,) or rpy.shape != (3,) or not np.isfinite((*xyz, *rpy)).all():
            raise ValueError("G1 base collision/visual has an invalid URDF origin")
        return xyz, Rotation.from_euler("xyz", rpy).as_matrix()

    collision_min_z = float("inf")
    collisions = base_link.findall("collision")
    if not collisions:
        raise ValueError("Prepared G1 base_link has no collision geometry")
    for collision in collisions:
        geometry = collision.find("geometry")
        box = geometry.find("box") if geometry is not None else None
        if box is None:
            raise ValueError("Base ground alignment expects source box colliders on G1 base_link")
        size = np.fromstring(box.get("size", ""), sep=" ")
        if size.shape != (3,) or np.any(size <= 0) or not np.isfinite(size).all():
            raise ValueError("G1 base_link has an invalid collision box size")
        xyz, rotation = origin_transform(collision)
        corners = np.asarray(
            [[sx * size[0] / 2, sy * size[1] / 2, sz * size[2] / 2]
             for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
            dtype=np.float64,
        )
        collision_min_z = min(collision_min_z, float((corners @ rotation.T + xyz)[..., 2].min()))

    visual_min_z = float("inf")
    visuals = base_link.findall("visual")
    for visual in visuals:
        mesh_element = visual.find("geometry/mesh")
        if mesh_element is None:
            continue
        mesh_path = (base_asset / mesh_element.get("filename", "")).resolve()
        if not mesh_path.is_relative_to(base_asset) or not mesh_path.is_file():
            raise ValueError(f"G1 base visual mesh is missing or escapes its asset: {mesh_path}")
        mesh = trimesh.load(mesh_path, force="mesh")
        xyz, rotation = origin_transform(visual)
        visual_min_z = min(visual_min_z, float((np.asarray(mesh.vertices) @ rotation.T + xyz)[..., 2].min()))
    if not np.isfinite(collision_min_z) or not np.isfinite(visual_min_z):
        raise ValueError("Could not measure G1 base collision and visual floor bounds")
    shift_m = visual_min_z - collision_min_z
    if shift_m < -1e-8 or shift_m > 0.01:
        raise ValueError(f"G1 base collision/visual floor mismatch is outside the safe range: {shift_m:.6f} m")
    shift_m = max(0.0, shift_m)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="g1-ground-alignment-", dir=output.parent) as temporary:
        dest = Path(temporary) / output.name
        shutil.copytree(base_asset, dest)

        def shift_urdf(path: Path) -> int:
            root = ET.parse(path).getroot()
            link = root.find("link[@name='base_link']")
            if link is None:
                raise ValueError(f"Prepared G1 URDF has no base_link: {path}")
            count = 0
            for collision in link.findall("collision"):
                origin = collision.find("origin")
                if origin is None:
                    origin = ET.Element("origin", xyz="0 0 0", rpy="0 0 0")
                    collision.insert(0, origin)
                xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                if xyz.shape != (3,):
                    raise ValueError(f"Invalid collision origin in {path}")
                xyz[2] += shift_m
                origin.set("xyz", " ".join(f"{value:.12g}" for value in xyz))
                count += 1
            ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
            return count

        urdf_collision_count = shift_urdf(dest / "G1_omnipicker.urdf")
        newton_urdf = dest / "G1_omnipicker.newton.urdf"
        if newton_urdf.is_file():
            if shift_urdf(newton_urdf) != urdf_collision_count:
                raise ValueError("Newton G1 URDF has a different base collision layout")

        mjcf_path = dest / "G1_omnipicker.xml"
        mjcf_root = ET.parse(mjcf_path).getroot()
        mjcf_base = mjcf_root.find(".//body[@name='base_link']")
        if mjcf_base is None:
            raise ValueError("Prepared G1 MJCF has no base_link")
        collision_geoms = [geom for geom in mjcf_base.findall("geom") if geom.get("group") == "3"]
        if len(collision_geoms) != urdf_collision_count:
            raise ValueError("G1 MJCF base collision geoms do not match its URDF collision shapes")
        for geom in collision_geoms:
            pos = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
            if pos.shape != (3,):
                raise ValueError("G1 MJCF base collision geom has an invalid position")
            pos[2] += shift_m
            geom.set("pos", " ".join(f"{value:.12g}" for value in pos))
        ET.ElementTree(mjcf_root).write(mjcf_path, encoding="utf-8", xml_declaration=True)
        mujoco.MjModel.from_xml_path(str(mjcf_path))

        updated = json.loads((dest / "provenance.json").read_text(encoding="utf-8"))
        updated["collision_ground_alignment"] = {
            "link": "base_link",
            "method": "shift source collision proxies to match the lowest composed chassis visual",
            "source_asset": str(base_asset),
            "source_urdf_sha256": _hash(base_asset / "G1_omnipicker.urdf"),
            "visual_min_z_m": visual_min_z,
            "collision_min_z_before_m": collision_min_z,
            "collision_min_z_after_m": collision_min_z + shift_m,
            "vertical_shift_m": shift_m,
            "shifted_collision_count": urdf_collision_count,
        }
        updated["prepared_urdf_sha256"] = _hash(dest / "G1_omnipicker.urdf")
        updated["prepared_mjcf_sha256"] = _hash(mjcf_path)
        if newton_urdf.is_file():
            updated["prepared_newton_urdf_sha256"] = _hash(newton_urdf)
        (dest / "provenance.json").write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
        shutil.move(str(dest), str(output))
    return shift_m


def _verify(source: Path, relative: str, *, entry: dict | None = None) -> Path:
    if entry is None:
        manifest = json.loads((source / "roboverse_source_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("repo") != "agibot-world/GenieSimAssets" or manifest.get("revision") != REVISION:
            raise ValueError("Expected the pinned official Genie Sim manifest")
        entries = [item for item in manifest["files"] if item["rfilename"] == relative]
        if len(entries) != 1:
            raise ValueError(f"File absent from source manifest: {relative}")
        entry = entries[0]
    if entry.get("rfilename") != relative:
        raise ValueError(f"Source manifest entry does not match {relative}")
    path = (source / relative).resolve()
    if not path.is_relative_to(source.resolve()):
        raise ValueError(f"Unsafe source path: {relative}")
    if not path.is_file() or path.stat().st_size != entry["size"]:
        raise ValueError(f"Missing or incomplete source file: {relative}")
    if "lfs" in entry:
        actual, expected = _hash(path), entry["lfs"]["sha256"]
    else:
        digest = hashlib.sha1(usedforsecurity=False)
        digest.update(f"blob {path.stat().st_size}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        actual, expected = digest.hexdigest(), entry["blobId"]
    if actual != expected:
        raise ValueError(f"Source checksum mismatch: {relative}")
    return path


def _bound_color(prim):
    """Read the authored diffuse color, including bindings on parent Xforms."""
    from pxr import Usd

    current = prim
    while current and current.GetPath() != "/":
        relation = current.GetRelationship("material:binding")
        targets = relation.GetForwardedTargets() if relation else []
        if targets:
            material = current.GetStage().GetPrimAtPath(targets[0])
            for shader in Usd.PrimRange(material):
                texture = shader.GetAttribute("inputs:diffuse_texture")
                if texture and texture.HasAuthoredValueOpinion() and texture.Get():
                    raise ValueError(f"Textured USD material is not yet supported: {targets[0]}")
                color = shader.GetAttribute("inputs:diffuse_color_constant")
                if color and color.HasAuthoredValueOpinion():
                    value = color.Get()
                    return tuple(float(channel) for channel in value) + (1.0,)
            raise ValueError(f"Unsupported material without a diffuse color: {targets[0]}")
        current = current.GetParent()
    return None


def _visual_mesh(stage, link: str, *, mesh_parent=None, mesh_prim=None):
    from pxr import Usd, UsdGeom

    if (mesh_parent is None) != (mesh_prim is None):
        raise ValueError("mesh_parent and mesh_prim must be supplied together")
    parent = mesh_parent if mesh_parent is not None else stage.GetPrimAtPath(f"/visuals/{link}")
    if not parent:
        raise ValueError(f"No official USD visual for link {link}")
    meshes = (
        [mesh_prim] if mesh_prim is not None else [prim for prim in Usd.PrimRange(parent) if prim.IsA(UsdGeom.Mesh)]
    )
    if len(meshes) != 1:
        raise ValueError(f"Expected one USD mesh for {link}, got {len(meshes)}")
    prim = meshes[0]
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    if np.any(counts < 3) or counts.sum() != len(indices):
        raise ValueError(f"Malformed USD mesh topology for {link}")
    authored_normals = mesh.GetNormalsAttr().Get()
    if authored_normals is None or len(authored_normals) == 0:
        source_normals = np.zeros((len(indices), 3), dtype=np.float64)
        interpolation = UsdGeom.Tokens.faceVarying
    else:
        source_normals = np.asarray(authored_normals, dtype=np.float64)
        interpolation = mesh.GetNormalsInterpolation()
        if source_normals.ndim != 2 or source_normals.shape[1] != 3 or not np.isfinite(source_normals).all():
            raise ValueError(f"USD visual has invalid authored normals for {link}")
    faces, triangle_source, triangle_normal_source, offset = [], [], [], 0
    for face_index, count in enumerate(counts):
        face = indices[offset : offset + count]
        for index in range(1, len(face) - 1):
            triangle = (face[0], face[index], face[index + 1])
            faces.append(triangle)
            triangle_source.append(face_index)
            if interpolation == UsdGeom.Tokens.faceVarying:
                triangle_normal_source.append((offset, offset + index, offset + index + 1))
            elif interpolation in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying):
                triangle_normal_source.append(triangle)
            elif interpolation == UsdGeom.Tokens.uniform:
                triangle_normal_source.append((face_index,) * 3)
            elif interpolation == UsdGeom.Tokens.constant and len(source_normals) == 1:
                triangle_normal_source.append((0, 0, 0))
            else:
                raise ValueError(f"Unsupported USD normal interpolation {interpolation!r} for {link}")
        offset += count
    transform, reset = UsdGeom.XformCache().ComputeRelativeTransform(prim, parent)
    if reset:
        raise ValueError(f"Unexpected reset transform in {link}")
    matrix = np.asarray(transform, dtype=np.float64).T
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"USD visual transform is invalid for {link}")
    points = trimesh.transform_points(points, matrix)
    source_normals = source_normals @ np.linalg.inv(matrix[:3, :3])
    normal_lengths = np.linalg.norm(source_normals, axis=1)
    valid_normals = normal_lengths > 1e-12
    source_normals[valid_normals] /= normal_lengths[valid_normals, None]
    source_normals[~valid_normals] = 0
    triangles = np.asarray(faces, dtype=np.int64)
    triangle_source = np.asarray(triangle_source, dtype=np.int64)
    triangle_normal_source = np.asarray(triangle_normal_source, dtype=np.int64)
    colors = {}
    assigned = np.zeros(len(counts), dtype=bool)
    for subset in prim.GetChildren():
        if not subset.IsA(UsdGeom.Subset):
            continue
        indices = np.asarray(UsdGeom.Subset(subset).GetIndicesAttr().Get(), dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= len(counts)) or np.any(assigned[indices]):
            raise ValueError(f"Invalid or overlapping material subsets on {link}")
        color = _bound_color(subset)
        if color is None:
            raise ValueError(f"Material subset has no color: {subset.GetPath()}")
        colors.setdefault(color, []).extend(indices.tolist())
        assigned[indices] = True
    remaining = np.flatnonzero(~assigned)
    if len(remaining):
        color = _bound_color(prim)
        if color is None:
            raise ValueError(f"Unassigned faces with no material on {link}")
        colors.setdefault(color, []).extend(remaining.tolist())
    parts = []
    for color, face_ids in colors.items():
        selected = np.isin(triangle_source, face_ids)
        selected_triangles = triangles[selected]
        selected_normals = source_normals[triangle_normal_source[selected]]
        triangle_edges_1 = points[selected_triangles[:, 1]] - points[selected_triangles[:, 0]]
        triangle_edges_2 = points[selected_triangles[:, 2]] - points[selected_triangles[:, 0]]
        geometric_normals = np.cross(triangle_edges_1, triangle_edges_2)
        geometric_lengths = np.linalg.norm(geometric_normals, axis=1)
        valid_geometric_normals = geometric_lengths > 1e-12
        geometric_normals[valid_geometric_normals] /= geometric_lengths[valid_geometric_normals, None]
        missing_normals = np.linalg.norm(selected_normals, axis=2) <= 1e-12
        if np.any(missing_normals):
            fallback = np.broadcast_to(geometric_normals[:, None, :], selected_normals.shape).copy()
            if np.any(missing_normals.any(axis=1) & ~valid_geometric_normals):
                fallback_vertex_normals = trimesh.Trimesh(
                    vertices=points, faces=triangles, process=False
                ).vertex_normals
            for triangle_index in np.flatnonzero(missing_normals.any(axis=1) & ~valid_geometric_normals):
                vertex_normals = fallback_vertex_normals[selected_triangles[triangle_index]]
                average = vertex_normals.mean(axis=0)
                average_length = np.linalg.norm(average)
                if average_length <= 1e-12:
                    # The source triangle has zero area and therefore cannot
                    # contribute visible surface shading.
                    fallback[triangle_index] = (0.0, 0.0, 1.0)
                else:
                    fallback[triangle_index] = average / average_length
            selected_normals[missing_normals] = fallback[missing_normals]
        corner_points = selected_triangles.reshape(-1)
        corner_normals = selected_normals.reshape(-1, 3)
        # Split vertices only at authored normal seams, preserving smooth panels
        # and the sharp edges specified by the source USD.
        keys = np.column_stack((corner_points, np.round(corner_normals, decimals=7)))
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
        part = trimesh.Trimesh(
            vertices=points[unique_keys[:, 0].astype(np.int64)],
            faces=inverse.reshape(-1, 3),
            process=False,
        )
        part.vertex_normals = unique_keys[:, 1:]
        parts.append((color, part))
    return trimesh.Trimesh(vertices=points, faces=triangles, process=False), parts


def _select_robot_usd_variants(stage):
    """Load the source robot base and camera payloads used in its USD assembly."""
    from pxr import Usd

    root = stage.GetDefaultPrim()
    if not root:
        raise ValueError("Official robot USD has no default prim")
    variants = root.GetVariantSets()
    # Use the authored PhysX collision layer as the source of truth. Selecting
    # Physics=None leaves the robot with render meshes only and forces target
    # backends to infer collisions from unrelated URDF visuals.
    for name, selection in (("Physics", "PhysX"), ("Robot", "Robot"), ("Sensor", "Sensors")):
        variant_set = variants.GetVariantSet(name)
        if selection not in variant_set.GetVariantNames() or not variant_set.SetVariantSelection(selection):
            raise ValueError(f"Official robot USD lacks variant {name}={selection}")
    stage.Load(root.GetPath(), Usd.LoadWithDescendants)
    return root


def _open_robot_usd_for_preparation(robot_usd: Path, link_names: list[str]):
    """Open only physical colliders and task cameras, excluding the viewport-only camera reference."""
    from pxr import Sdf, Usd

    population = Usd.StagePopulationMask()
    for link_name in link_names:
        population.Add(Sdf.Path(f"/G01/{link_name}/collisions"))
        # Link visuals can carry an attachment transform in robot.usda that is
        # absent from the standalone geometry layer. Keep these Xform prims in
        # the composed stage so the converted URDF preserves the authored pose.
        population.Add(Sdf.Path(f"/G01/{link_name}/visuals"))
    for camera_name in ("Head_Camera", "Fisheye_Camera_01", "Fisheye_Camera_02"):
        population.Add(Sdf.Path(f"/G01/head_link2/{camera_name}"))
    for side in ("l", "r"):
        population.Add(Sdf.Path(f"/G01/gripper_{side}_base_link/Cam"))
    stage = Usd.Stage.OpenMasked(str(Path(robot_usd).resolve()), population, Usd.Stage.LoadNone)
    if not stage:
        raise ValueError(f"Cannot open source robot USD payloads: {robot_usd}")
    return stage


def _add_wrist_camera_visuals(stage, root: ET.Element, mesh_root: Path) -> list[dict]:
    """Copy Genie Sim's attached wrist camera housings into the G1 URDF visuals."""
    from pxr import Usd, UsdGeom

    source_root = stage.GetDefaultPrim()
    if not source_root:
        raise ValueError("Official robot USD has no default prim")
    records = []
    for side in ("l", "r"):
        link_name = f"gripper_{side}_base_link"
        link = root.find(f"link[@name='{link_name}']")
        parent = stage.GetPrimAtPath(f"{source_root.GetPath()}/{link_name}")
        camera_mount = stage.GetPrimAtPath(f"{source_root.GetPath()}/{link_name}/Cam")
        if link is None or not parent or not camera_mount:
            raise ValueError(f"Official USD wrist camera mount is missing for {link_name}")
        meshes = [prim for prim in Usd.PrimRange(camera_mount) if prim.IsA(UsdGeom.Mesh)]
        if len(meshes) != 2:
            raise ValueError(f"Expected two official wrist camera meshes for {link_name}, got {len(meshes)}")
        for mesh_prim in meshes:
            _, parts = _visual_mesh(stage, link_name, mesh_parent=parent, mesh_prim=mesh_prim)
            for part_index, (color, part) in enumerate(parts):
                mesh_name = f"{link_name}_camera_{mesh_prim.GetName()}_{part_index}.obj"
                part.export(mesh_root / mesh_name, include_normals=True)
                visual = ET.SubElement(link, "visual", name=f"geniesim_{mesh_name.removesuffix('.obj')}")
                geometry = ET.SubElement(visual, "geometry")
                ET.SubElement(geometry, "mesh", filename=f"meshes/{mesh_name}")
                material = ET.SubElement(visual, "material", name=f"geniesim_{mesh_name.removesuffix('.obj')}_mat")
                ET.SubElement(material, "color", rgba=" ".join(f"{value:.9g}" for value in color))
                records.append({"link": link_name, "source_prim": str(mesh_prim.GetPath()), "mesh": mesh_name})
    return records


def _usd_visual_origin(stage, link: str, *, root_path: str = "/genie") -> tuple[np.ndarray, np.ndarray]:
    """Return the composed USD visual-to-link pose, independently of URDF/FBX origins."""
    from pxr import UsdGeom

    body = stage.GetPrimAtPath(f"{root_path}/{link}")
    visual = stage.GetPrimAtPath(f"{root_path}/{link}/visuals")
    if not body or not visual:
        raise ValueError(f"No USD visual attachment for {link}")
    transform, reset = UsdGeom.XformCache().ComputeRelativeTransform(visual, body)
    matrix = np.asarray(transform, dtype=float).T
    rotation = matrix[:3, :3]
    if reset or not np.isfinite(matrix).all() or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"Non-rigid or reset USD visual attachment for {link}")
    if not np.isclose(np.linalg.det(rotation), 1, atol=1e-6):
        raise ValueError(f"Reflected USD visual attachment for {link}")
    return matrix[:3, 3], Rotation.from_matrix(rotation).as_euler("xyz")


def _set_origin(element: ET.Element, translation: np.ndarray, rpy: np.ndarray) -> None:
    origin = element.find("origin")
    if origin is None:
        origin = ET.SubElement(element, "origin")
    origin.set("xyz", " ".join(f"{value:.12g}" for value in translation))
    origin.set("rpy", " ".join(f"{value:.12g}" for value in rpy))


_COACD_COLLISION_OPTIONS = {
    "threshold": 0.1,
    "max_convex_hull": -1,
    "preprocess_mode": "auto",
    "preprocess_resolution": 50,
    "resolution": 1000,
    "mcts_nodes": 20,
    "mcts_iterations": 100,
    "mcts_max_depth": 3,
    "pca": False,
    "merge": True,
    "decimate": False,
    "max_ch_vertex": 256,
    "extrude": False,
    "extrude_margin": 0.1,
    "apx_mode": "ch",
    "seed": 0,
}


def _usd_prim_transform_in_link(stage, link_prim, prim) -> np.ndarray:
    """Return an authored collider transform as a column-vector link-local matrix."""
    from pxr import UsdGeom

    cache = UsdGeom.XformCache()
    link_world = np.asarray(cache.GetLocalToWorldTransform(link_prim), dtype=np.float64).T
    prim_world = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64).T
    if not np.isfinite(link_world).all() or not np.isfinite(prim_world).all():
        raise ValueError(f"Non-finite USD collision transform: {prim.GetPath()}")
    return np.linalg.inv(link_world) @ prim_world


def _decompose_usd_shape_transform(matrix: np.ndarray, source_prim: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a USD local transform into translation, proper rotation, and positive axis scales."""
    linear = np.asarray(matrix[:3, :3], dtype=np.float64)
    scale = np.linalg.norm(linear, axis=0)
    if not np.isfinite(scale).all() or np.any(scale < 1e-10):
        raise ValueError(f"Degenerate USD collision transform: {source_prim}")
    rotation = linear / scale[None, :]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"USD collision transform contains shear: {source_prim}")
    if np.linalg.det(rotation) < 0:
        # All supported primitive collision shapes are symmetric about their local axes.
        rotation[:, 0] *= -1
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"USD collision transform is not a proper rotation: {source_prim}")
    translation = np.asarray(matrix[:3, 3], dtype=np.float64)
    return translation, Rotation.from_matrix(rotation).as_euler("xyz"), scale


def _usd_mesh_collision_parts(mesh: trimesh.Trimesh, approximation: str) -> list[trimesh.Trimesh]:
    """Bake the convex collision representation authored by USD into watertight meshes."""
    source = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
    if approximation == "convexHull":
        parts = [source.convex_hull]
    elif approximation == "convexDecomposition":
        try:
            import coacd
        except ImportError as exc:
            raise ImportError(
                "G1 source USD requests convex decomposition; install the `icl-geniesim` extra to prepare collision assets"
            ) from exc
        decomposed = coacd.run_coacd(
            coacd.Mesh(source.vertices.astype(np.float32), source.faces.astype(np.int32)),
            **_COACD_COLLISION_OPTIONS,
        )
        parts = [trimesh.Trimesh(vertices=vertices, faces=faces, process=True) for vertices, faces in decomposed]
    else:
        raise ValueError(f"Unsupported USD mesh collision approximation: {approximation!r}")

    validated = []
    for part in parts:
        if len(part.faces) < 4 or not part.is_watertight or not part.is_convex or part.volume <= 0:
            raise ValueError(f"USD collision approximation {approximation!r} did not produce a valid convex solid")
        validated.append(part)
    if not validated:
        raise ValueError(f"USD collision approximation {approximation!r} produced no collision geometry")
    return validated


def _usd_link_collision_elements(
    stage,
    source_root,
    link_name: str,
    mesh_root: Path,
    mesh_cache: dict[str, list[trimesh.Trimesh]],
) -> tuple[list[ET.Element], list[dict]]:
    """Export every source USD collision primitive and mesh for one robot link."""
    from pxr import UsdGeom, UsdPhysics

    link_prim = stage.GetPrimAtPath(f"{source_root.GetPath()}/{link_name}")
    if not link_prim:
        return [], []
    collision_prefix = f"{link_prim.GetPath()}/collisions/"
    source_prims = sorted(
        (
            prim
            for prim in stage.Traverse()
            if str(prim.GetPath()).startswith(collision_prefix) and prim.HasAPI(UsdPhysics.CollisionAPI)
        ),
        key=lambda prim: str(prim.GetPath()),
    )
    elements: list[ET.Element] = []
    records: list[dict] = []
    for collision_index, prim in enumerate(source_prims):
        source_prim = str(prim.GetPath())
        transform = _usd_prim_transform_in_link(stage, link_prim, prim)
        record = {"link": link_name, "source_prim": source_prim}

        if prim.IsA(UsdGeom.Mesh):
            mesh_prim = UsdGeom.Mesh(prim)
            points = np.asarray(mesh_prim.GetPointsAttr().Get(), dtype=np.float64)
            face_counts = np.asarray(mesh_prim.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
            face_indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
            if not len(points) or not len(face_counts) or np.any(face_counts != 3):
                raise ValueError(f"USD collision mesh must contain indexed triangles: {source_prim}")
            faces = face_indices.reshape(-1, 3)
            if str(mesh_prim.GetOrientationAttr().Get()) == "leftHanded":
                faces = faces[:, [0, 2, 1]]
            source_mesh = trimesh.Trimesh(vertices=points, faces=faces, process=True)
            mesh_collision = UsdPhysics.MeshCollisionAPI(prim)
            approximation = str(mesh_collision.GetApproximationAttr().Get() or "convexHull")
            source_hash = hashlib.sha256(
                approximation.encode("utf-8")
                + source_mesh.vertices.astype("<f8").tobytes()
                + source_mesh.faces.astype("<i8").tobytes()
            ).hexdigest()
            parts = mesh_cache.get(source_hash)
            if parts is None:
                parts = _usd_mesh_collision_parts(source_mesh, approximation)
                mesh_cache[source_hash] = parts
            part_records = []
            for part_index, part in enumerate(parts):
                mesh_name = f"{link_name}_collision_{collision_index}_part_{part_index}.obj"
                link_vertices = (transform[:3, :3] @ part.vertices.T + transform[:3, 3:4]).T
                link_faces = part.faces[:, [0, 2, 1]] if np.linalg.det(transform[:3, :3]) < 0 else part.faces
                link_part = trimesh.Trimesh(vertices=link_vertices, faces=link_faces, process=True)
                if link_part.volume < 0:
                    link_part.invert()
                if not link_part.is_watertight or not link_part.is_convex or link_part.volume <= 0:
                    raise ValueError(f"Transformed USD collision part is invalid: {source_prim}")
                link_part.export(mesh_root / mesh_name)
                # URDF collision names are unnecessary here. MuJoCo expands
                # named collisions into multiple geoms and emits false
                # duplicate-name warnings during conversion.
                item = ET.Element("collision")
                ET.SubElement(item, "origin", xyz="0 0 0", rpy="0 0 0")
                item_geometry = ET.SubElement(item, "geometry")
                ET.SubElement(item_geometry, "mesh", filename=f"meshes/{mesh_name}")
                elements.append(item)
                part_records.append({
                    "mesh": mesh_name,
                    "sha256": _hash(mesh_root / mesh_name),
                    "vertices": len(link_part.vertices),
                    "faces": len(link_part.faces),
                    "watertight": link_part.is_watertight,
                })
            record.update({
                "geometry_type": "mesh",
                "source_approximation": approximation,
                "link_from_mesh": transform.tolist(),
                "source_vertices": len(source_mesh.vertices),
                "source_faces": len(source_mesh.faces),
                "source_watertight": source_mesh.is_watertight,
                "parts": part_records,
            })
            records.append(record)
            continue

        translation, rpy, scale = _decompose_usd_shape_transform(transform, source_prim)
        collision = ET.Element("collision")
        origin = ET.SubElement(collision, "origin")
        geometry = ET.SubElement(collision, "geometry")
        origin.set("xyz", " ".join(f"{value:.12g}" for value in translation))
        origin.set("rpy", " ".join(f"{value:.12g}" for value in rpy))
        if prim.IsA(UsdGeom.Cube):
            size = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) * scale
            ET.SubElement(geometry, "box", size=" ".join(f"{value:.12g}" for value in size))
            record.update({"geometry_type": "box", "size_m": size.tolist()})
        elif prim.IsA(UsdGeom.Cylinder):
            cylinder = UsdGeom.Cylinder(prim)
            axis = str(cylinder.GetAxisAttr().Get())
            axis_index = {"X": 0, "Y": 1, "Z": 2}.get(axis)
            if axis_index is None:
                raise ValueError(f"Unsupported USD cylinder axis {axis!r}: {source_prim}")
            radial_scales = [scale[index] for index in range(3) if index != axis_index]
            if not np.isclose(radial_scales[0], radial_scales[1], rtol=1e-5, atol=1e-8):
                raise ValueError(f"Non-circular USD cylinder cannot map to URDF cylinder: {source_prim}")
            radius = float(cylinder.GetRadiusAttr().Get()) * radial_scales[0]
            length = float(cylinder.GetHeightAttr().Get()) * scale[axis_index]
            if axis_index != 2:
                local_axis = np.eye(3)[axis_index]
                align = Rotation.align_vectors([local_axis], [[0.0, 0.0, 1.0]])[0]
                rotation = Rotation.from_euler("xyz", rpy) * align
                rpy = rotation.as_euler("xyz")
                origin.set("rpy", " ".join(f"{value:.12g}" for value in rpy))
            ET.SubElement(geometry, "cylinder", radius=f"{radius:.12g}", length=f"{length:.12g}")
            record.update({"geometry_type": "cylinder", "radius_m": radius, "length_m": length})
        elif prim.IsA(UsdGeom.Sphere):
            sphere = UsdGeom.Sphere(prim)
            if not np.allclose(scale, scale[0], rtol=1e-5, atol=1e-8):
                raise ValueError(f"Non-uniformly scaled USD sphere requires a mesh collision proxy: {source_prim}")
            radius = float(sphere.GetRadiusAttr().Get()) * float(scale[0])
            ET.SubElement(geometry, "sphere", radius=f"{radius:.12g}")
            record.update({"geometry_type": "sphere", "radius_m": radius})
        else:
            raise ValueError(f"Unsupported USD collision geometry {prim.GetTypeName()!r}: {source_prim}")
        elements.append(collision)
        records.append(record)
    return elements, records


def _sanitize_urdf_inertias(root: ET.Element) -> list[dict]:
    """Apply the same small isotropic inertia floor Newton otherwise adds at load time."""
    corrections = []
    for link in root.findall("link"):
        inertia = link.find("inertial/inertia")
        if inertia is None:
            continue
        names = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
        try:
            ixx, ixy, ixz, iyy, iyz, izz = (float(inertia.get(name)) for name in names)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"G1 link {link.get('name')!r} has an incomplete inertia tensor") from exc
        matrix = np.asarray([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=np.float64)
        if not np.isfinite(matrix).all():
            raise ValueError(f"G1 link {link.get('name')!r} has a non-finite inertia tensor")
        eigenvalues = np.linalg.eigvalsh(matrix)
        if eigenvalues[0] >= _MIN_BACKEND_INERTIA_EIGENVALUE:
            continue
        offset = _MIN_BACKEND_INERTIA_EIGENVALUE - float(eigenvalues[0])
        corrected = matrix + np.eye(3, dtype=np.float64) * offset
        for name, value in zip(
            names,
            (corrected[0, 0], corrected[0, 1], corrected[0, 2], corrected[1, 1], corrected[1, 2], corrected[2, 2]),
            strict=True,
        ):
            inertia.set(name, f"{value:.12g}")
        corrections.append({
            "link": link.get("name"),
            "diagonal_offset_kg_m2": offset,
            "principal_moments_before_kg_m2": eigenvalues.tolist(),
            "principal_moments_after_kg_m2": np.linalg.eigvalsh(corrected).tolist(),
        })
    return corrections


def prepare(source: Path, urdf: Path, output: Path, variant: str = "G1_120s") -> None:
    import mujoco
    from pxr import Usd, UsdGeom

    source, urdf, output = source.resolve(), urdf.resolve(), output.resolve()
    if variant not in {"G1_120s", "G1_omnipicker"}:
        raise ValueError(f"Unsupported AgiBot G1 variant: {variant}")
    if output.exists():
        raise FileExistsError(output)
    usd = _verify(source, f"robot/{variant}/configuration/{variant}_base.usd")
    robot_usd = _verify(source, f"robot/{variant}/robot.usda")
    camera_model_usd = _verify(source, f"robot/{variant}/configuration/Cam.usd")
    tree = ET.parse(urdf)
    root = tree.getroot()
    if root.tag != "robot" or root.find("link[@name='head_link2']") is None:
        raise ValueError("Expected a Genie Sim G1 robot URDF")
    has_pad_link = root.find("link[@name='gripper_l_inner_link5']") is not None
    if has_pad_link != (variant == "G1_120s"):
        raise ValueError(f"The supplied URDF does not match {variant}")
    inertia_corrections = _sanitize_urdf_inertias(root)
    stage = Usd.Stage.Open(str(usd))
    if not stage:
        raise ValueError("Cannot open the verified G1 geometry USD")
    camera_stage = _open_robot_usd_for_preparation(robot_usd, [link.get("name") for link in root.findall("link")])
    source_root = _select_robot_usd_variants(camera_stage)
    drive_parameters = _source_drive_parameters(
        robot_usd, source / "robot" / variant / "configuration" / "robot_physics.usd"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="g1-", dir=output.parent) as temp:
        dest = Path(temp) / "asset"
        meshes_dir = dest / "meshes"
        meshes_dir.mkdir(parents=True)
        links = []
        visual_attachment_links = {}
        collision_geometry = []
        collision_mesh_cache: dict[str, list[trimesh.Trimesh]] = {}
        for link in root.findall("link"):
            name = link.get("name")
            source_collisions, collision_records = _usd_link_collision_elements(
                camera_stage, source_root, name, meshes_dir, collision_mesh_cache
            )
            for previous in link.findall("collision"):
                link.remove(previous)
            link.extend(source_collisions)
            collision_geometry.extend(collision_records)

            if not link.findall("visual/geometry/mesh"):
                continue
            _, parts = _visual_mesh(stage, name)
            # The geometry comes from the base USD layer, but the composed
            # robot.usda can author an additional per-link visual attachment
            # (G1 Omnipicker's base visual is raised 0.211 m so its wheels sit
            # on the ground). Read that transform from the composed robot USD;
            # otherwise the chassis mesh clips through the floor.
            translation, rpy = _usd_visual_origin(camera_stage, name, root_path=str(source_root.GetPath()))
            visual_attachment_links[name] = {"xyz": translation.tolist(), "rpy": rpy.tolist()}
            for visual in list(link.findall("visual")):
                if visual.find("geometry/mesh") is None:
                    continue
                link.remove(visual)
                for index, (color, part) in enumerate(parts):
                    part_name = f"{name}_visual_{index}.obj"
                    part.export(meshes_dir / part_name, include_normals=True)
                    item = copy.deepcopy(visual)
                    item.find("geometry/mesh").set("filename", f"meshes/{part_name}")
                    _set_origin(item, translation, rpy)
                    for old in item.findall("material"):
                        item.remove(old)
                    material = ET.SubElement(item, "material", name=f"geniesim_{name}_{index}")
                    ET.SubElement(material, "color", rgba=" ".join(f"{value:.9g}" for value in color))
                    link.append(item)
            links.append(name)
        wrist_camera_visuals = _add_wrist_camera_visuals(camera_stage, root, meshes_dir)
        # Keep MuJoCo-only compiler directives out of the shared URDF. Genesis
        # falls back to its legacy parser when it sees this extension. Use a
        # temporary conversion copy so MuJoCo still retains visuals and links.
        for extension in root.findall("mujoco"):
            root.remove(extension)
        urdf_path = dest / f"{variant}.urdf"
        tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
        conversion_root = copy.deepcopy(root)
        extension = ET.SubElement(conversion_root, "mujoco")
        compiler = ET.SubElement(extension, "compiler")
        compiler.set("discardvisual", "false")
        compiler.set("fusestatic", "false")
        conversion_urdf = dest / f".{variant}.mujoco-conversion.urdf"
        ET.ElementTree(conversion_root).write(conversion_urdf, encoding="utf-8", xml_declaration=True)
        model = mujoco.MjModel.from_xml_path(str(conversion_urdf))
        mujoco.mj_saveLastXML(str(dest / f"{variant}.xml"), model)
        del model
        conversion_urdf.unlink()
        mjcf = ET.parse(dest / f"{variant}.xml")
        equality = mjcf.getroot().find("equality")
        if equality is None:
            equality = ET.SubElement(mjcf.getroot(), "equality")
        for loop in root.findall("loop_joint"):
            if loop.get("type") != "spherical":
                raise ValueError(f"Unsupported G1 loop joint: {loop.get('name')}")
            sites = []
            for tag in ("link1", "link2"):
                endpoint = loop.find(tag)
                body = mjcf.getroot().find(f".//body[@name='{endpoint.get('link')}']")
                if body is None:
                    raise ValueError(f"Missing body for loop joint {loop.get('name')}")
                site_name = f"{loop.get('name')}_{tag}"
                ET.SubElement(body, "site", name=site_name, pos=endpoint.get("xyz"), size="0.001")
                sites.append(site_name)
            ET.SubElement(equality, "connect", name=loop.get("name"), site1=sites[0], site2=sites[1])
        for joint in root.findall("joint"):
            mimic = joint.find("mimic")
            if mimic is not None:
                existing = [
                    item
                    for item in equality.findall("joint")
                    if item.get("joint1") == joint.get("name") and item.get("joint2") == mimic.get("joint")
                ]
                if existing:
                    continue
                offset = float(mimic.get("offset", "0"))
                multiplier = float(mimic.get("multiplier", "1"))
                ET.SubElement(
                    equality,
                    "joint",
                    name=f"mimic_{joint.get('name')}",
                    joint1=joint.get("name"),
                    joint2=mimic.get("joint"),
                    polycoef=f"{offset} {multiplier} 0 0 0",
                )
        for geom in mjcf.findall(".//worldbody//geom"):
            if geom.get("contype") != "0" or geom.get("conaffinity") != "0":
                # Keep collision geometry active but hide it in the default view.
                geom.set("group", "3")
        mjcf.write(dest / f"{variant}.xml", encoding="utf-8", xml_declaration=True)
        try:
            from roboverse_pack.tasks.icl.fruit_sorting import CONTROL_JOINTS
        except ImportError:
            CONTROL_JOINTS = ()
        active_joints = set(CONTROL_JOINTS) if variant == "G1_omnipicker" else None
        _replace_mjcf_actuators(
            dest / f"{variant}.xml", drive_parameters, active_joint_names=active_joints
        )
        mujoco.MjModel.from_xml_path(str(dest / f"{variant}.xml"))
        cameras = {}
        root_camera_path = camera_stage.GetDefaultPrim().GetPath()
        for name in ("Head_Camera", "Fisheye_Camera_01", "Fisheye_Camera_02"):
            prim = camera_stage.GetPrimAtPath(f"{root_camera_path}/head_link2/{name}")
            if not prim or not prim.IsA(UsdGeom.Camera):
                raise ValueError(f"Missing source camera {name}")
            attributes = {}
            attributes["source_prim"] = str(prim.GetPath())
            for key in (
                "cameraProjectionType",
                "focalLength",
                "horizontalAperture",
                "verticalAperture",
                "fthetaPolyB",
                "fthetaPolyC",
            ):
                attribute = prim.GetAttribute(key)
                if attribute and attribute.HasAuthoredValueOpinion():
                    value = attribute.Get()
                    attributes[key] = float(value) if isinstance(value, (int, float)) else str(value)
            # Preserve the authored image size and OpenCV calibration for the
            # head RGB camera. The source fisheye prims do not declare a native
            # image size; do not invent one from the aperture ratio.
            prefix = "omni:lensdistortion:opencvPinhole:"
            size_attr = prim.GetAttribute(prefix + "imageSize")
            if size_attr and size_attr.HasAuthoredValueOpinion():
                size = size_attr.Get()
                if len(size) != 2 or any(int(value) <= 0 for value in size):
                    raise ValueError(f"Invalid source camera image size for {name}")
                attributes["native_image_size"] = [int(value) for value in size]
                calibration = {}
                for key in ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2"):
                    attribute = prim.GetAttribute(prefix + key)
                    if attribute and attribute.HasAuthoredValueOpinion():
                        calibration[key] = float(attribute.Get())
                attributes["opencv_pinhole"] = calibration
            attributes["head_link_from_camera_usd_row_matrix"] = np.asarray(
                UsdGeom.Xformable(prim).GetLocalTransformation()
            ).tolist()
            cameras[name] = attributes
        info = {
            "format": "roboverse.icl.geniesim_g1_robot.v4",
            "source": "https://huggingface.co/datasets/agibot-world/GenieSimAssets",
            "source_revision": REVISION,
            "asset_license": "CC-BY-NC-SA-4.0",
            "robot_variant": variant,
            "usd_sha256": _hash(usd),
            "robot_usd_sha256": _hash(robot_usd),
            "camera_model_usd_sha256": _hash(camera_model_usd),
            "camera_calibration_source": {
                "usd": f"robot/{variant}/robot.usda",
                "sha256": _hash(robot_usd),
                "projection_note": "OpenCV coefficients are source metadata; backend distortion support is separate",
            },
            "urdf_source": f"Genie Sim source/geniesim_benchmark/robot/isaac_sim/robot_urdf/{variant}.urdf",
            "urdf_sha256": _hash(urdf),
            "urdf_license": "MPL-2.0",
            "mesh_links": links,
            "geometry": "USD link-relative visual mesh subsets, diffuse colors, authored normals and official "
            "wrist camera housings preserved in OBJ visuals; collision geometry is exported from the source USD "
            "PhysicsCollisionAPI with its authored primitive, convexHull or convexDecomposition approximation",
            "collision_geometry": {
                "source": "robot.usda PhysicsCollisionAPI",
                "coacd_options": _COACD_COLLISION_OPTIONS,
                "coacd_version": _package_version("coacd"),
                "urdf_collision_names": "omitted to avoid MuJoCo importer duplicate-name warnings",
                "records": collision_geometry,
            },
            "wrist_camera_visuals": wrist_camera_visuals,
            "visual_attachment_transforms": {
                "source": "robot.usda composed link visuals relative to link",
                "source_root": str(source_root.GetPath()),
                "source_sha256": _hash(robot_usd),
                "links": visual_attachment_links,
            },
            "physics": "Source URDF inertias retained except the provenance-listed isotropic diagonal offsets "
            "needed to match Newton's positive-definite inertia floor; source USD collision shapes and "
            "approximations are translated into shared URDF/MJCF geometry. MJCF position drives preserve the "
            "pinned USD stiffness, damping and authored force limits. Cross-backend dynamics require runtime validation",
            "drive_source": {
                "repository": "https://huggingface.co/datasets/agibot-world/GenieSimAssets",
                "revision": REVISION,
                "robot_usd": f"robot/{variant}/robot.usda",
                "robot_usd_sha256": _hash(robot_usd),
                "units": "angular stiffness and damping converted from per-degree to per-radian",
                "parameters": drive_parameters,
                "active_joint_names": sorted(active_joints) if active_joints is not None else None,
                "passive_joint_policy": (
                    "zero gains on mimic/closed-loop followers to avoid redundant actuation"
                    if active_joints is not None
                    else None
                ),
            },
            "inertia_floor_kg_m2": _MIN_BACKEND_INERTIA_EIGENVALUE,
            "inertia_corrections": inertia_corrections,
            "prepared_urdf_sha256": _hash(dest / f"{variant}.urdf"),
            "prepared_mjcf_sha256": _hash(dest / f"{variant}.xml"),
            "loop_constraints": len(root.findall("loop_joint")),
            "mimic_constraints": sum(joint.find("mimic") is not None for joint in root.findall("joint")),
            "cameras": cameras,
        }
        (dest / "provenance.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        shutil.move(str(dest), str(output))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base-asset", type=Path, help="prepared G1 asset to upgrade with source USD drives")
    parser.add_argument("--apply-source-drives-from", type=Path, help="pinned robot.usda drive source")
    parser.add_argument(
        "--align-base-ground-collision-from-asset",
        type=Path,
        help="create a versioned asset whose base collision floor matches its composed visual floor",
    )
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--variant", choices=("G1_120s", "G1_omnipicker"), default="G1_120s")
    args = parser.parse_args()
    if args.apply_source_drives_from is not None:
        if args.base_asset is None or args.output is None:
            parser.error("--base-asset and --output are required with --apply-source-drives-from")
        add_source_drives_to_prepared_asset(args.base_asset, args.apply_source_drives_from, args.output)
        print(f"Prepared source-driven G1 asset: {args.output}")
        return
    if args.align_base_ground_collision_from_asset is not None:
        if args.output is None:
            parser.error("--output is required with --align-base-ground-collision-from-asset")
        shift_m = align_base_collision_with_visual_floor(args.align_base_ground_collision_from_asset, args.output)
        print(f"Prepared ground-aligned G1 asset: {args.output} (base collision shifted by {shift_m:.9f} m)")
        return
    if args.write_manifest:
        if args.source is None:
            parser.error("--source is required with --write-manifest")
        print(f"Saved official checksum manifest: {write_manifest(args.source)}")
        return
    if args.source is None or args.urdf is None or args.output is None:
        parser.error("--urdf and --output are required when preparing an asset")
    prepare(args.source, args.urdf, args.output, args.variant)
    print(f"Prepared G1 simulation asset: {args.output}")


if __name__ == "__main__":
    main()
