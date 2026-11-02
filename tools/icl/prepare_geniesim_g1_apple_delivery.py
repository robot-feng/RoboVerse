"""Import the supplied G1 apple scene bundle as a portable RoboVerse task scene."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.icl.prepare_geniesim_objects import _triangles, _write_obj, _write_wrappers

_BUNDLE_ID = "g1_apple_scene_delivery_20261031_v4"
_SCENE_FORMAT = "roboverse.icl.fruit_scene.v1"
_OBJECT_FORMAT = "roboverse.icl.g1_apple_delivery_object.v4"
_PLATE_USD_SHA256 = "981b9bd1f7d8abc7d21a3435853e41e057694decffaafbfae68e2031a4ca0b94"
_MAX_TEXTURE_SIZE = 1024
_JOINT_POSE_PATTERN = re.compile(
    r'over "(?P<name>[^\"]+)"\s*\{\s*'
    r"float state:(?P<kind>angular|linear):physics:position\s*=\s*(?P<value>[-+0-9.eE]+)",
    re.MULTILINE,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_source_bundle(source_root: Path, archive_sha256: str) -> dict:
    manifest_path = source_root / "delivery_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing G1 Apple delivery manifest: {manifest_path}")
    delivery = json.loads(manifest_path.read_text(encoding="utf-8"))
    if delivery.get("schema") != "embodiforge.g1_apple_scene_delivery.v1":
        raise ValueError("Unsupported G1 Apple scene delivery manifest")
    for name, record in delivery.get("files", {}).items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe delivery manifest path: {name}")
        path = source_root.joinpath(*relative.parts)
        if not path.is_file() or path.stat().st_size != record.get("bytes") or _sha256(path) != record.get("sha256"):
            raise ValueError(f"G1 Apple delivery file failed hash/size verification: {name}")
    manifest_path_2 = source_root / "manifest.json"
    if not manifest_path_2.is_file():
        raise FileNotFoundError(f"Missing source scene manifest: {manifest_path_2}")
    manifest = json.loads(manifest_path_2.read_text(encoding="utf-8"))
    return {
        "delivery": delivery,
        "manifest": manifest,
        "delivery_manifest_sha256": _sha256(manifest_path),
        "manifest_sha256": _sha256(manifest_path_2),
        "archive_sha256": archive_sha256,
    }


def _extract_bundle(archive_path: Path, source_root: Path) -> str:
    archive_path, source_root = Path(archive_path).resolve(), Path(source_root).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"G1 Apple scene archive does not exist: {archive_path}")
    archive_sha256 = _sha256(archive_path)
    recorded_hash = source_root / "source_archive.sha256"
    if source_root.exists():
        if not recorded_hash.is_file() or recorded_hash.read_text(encoding="ascii").strip() != archive_sha256:
            raise FileExistsError(f"Refusing to reuse an unrelated source bundle: {source_root}")
        _verify_source_bundle(source_root, archive_sha256)
        return archive_sha256

    source_root.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"{_BUNDLE_ID}/"
    with ZipFile(archive_path) as bundle:
        members = bundle.infolist()
        by_name = {item.filename.removeprefix(prefix): item for item in members if item.filename.startswith(prefix)}
        if len(by_name) != len(members) or "delivery_manifest.json" not in by_name:
            raise ValueError("Archive entries do not match the expected G1 Apple delivery root")
        delivery = json.loads(bundle.read(by_name["delivery_manifest.json"]).decode("utf-8"))
        expected = delivery.get("files", {})
        if set(expected) - set(by_name):
            raise ValueError(f"Archive is missing delivery files: {sorted(set(expected) - set(by_name))}")
        total_uncompressed = 0
        for name, record in expected.items():
            item = by_name[name]
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Unsafe G1 Apple delivery entry: {name}")
            data = bundle.read(item)
            if len(data) != record.get("bytes") or hashlib.sha256(data).hexdigest() != record.get("sha256"):
                raise ValueError(f"Archive hash/size mismatch: {name}")
            total_uncompressed += len(data)
        if total_uncompressed > 512 * 1024 * 1024:
            raise ValueError(f"Unexpected G1 Apple archive expansion size: {total_uncompressed}")

        with tempfile.TemporaryDirectory(prefix="rvsrc_", dir=source_root.parent) as temporary:
            staged = Path(temporary) / "bundle"
            staged.mkdir()
            for name, item in by_name.items():
                relative = PurePosixPath(name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Unsafe archive path: {name}")
                target = staged.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(bundle.read(item))
            (staged / "source_archive.sha256").write_text(archive_sha256 + "\n", encoding="ascii")
            _verify_source_bundle(staged, archive_sha256)
            shutil.move(str(staged), source_root)
    return archive_sha256


def _delivery_material_groups(stage, mesh_prim, source_root: Path, asset_dir: Path, asset_name: str) -> list[dict]:
    """Read standard USD Preview Surface materials in the supplied package."""
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    face_count = len(UsdGeom.Mesh(mesh_prim).GetFaceVertexCountsAttr().Get())
    groups: list[dict] = []
    assigned = np.zeros(face_count, dtype=bool)
    lookup = {}

    def resolve_material(binding_prim) -> tuple[Path | None, tuple[float, float, float]]:
        material, _ = UsdShade.MaterialBindingAPI(binding_prim).ComputeBoundMaterial()
        if not material:
            return None, (0.72, 0.72, 0.72)
        shaders = [prim for prim in Usd.PrimRange(material.GetPrim()) if prim.IsA(UsdShade.Shader)]
        shader = next(
            (
                UsdShade.Shader(prim)
                for prim in shaders
                if UsdShade.Shader(prim).GetIdAttr().Get() == "UsdPreviewSurface"
            ),
            None,
        )
        if shader is None:
            shader = next(
                (UsdShade.Shader(prim) for prim in shaders if UsdShade.Shader(prim).GetInput("diffuse_color_constant")),
                None,
            )
        if shader is None:
            return None, (0.72, 0.72, 0.72)
        diffuse_input = shader.GetInput("diffuseColor") or shader.GetInput("diffuse_color_constant")
        color_value = diffuse_input.Get() if diffuse_input else None
        color = tuple(float(value) for value in color_value)[:3] if color_value is not None else (0.72, 0.72, 0.72)
        file_value = None
        if diffuse_input:
            connections, _ = diffuse_input.GetConnectedSources()
            for connection in connections:
                texture_shader = UsdShade.Shader(stage.GetPrimAtPath(connection.source.GetPath()))
                if texture_shader and texture_shader.GetIdAttr().Get() == "UsdUVTexture":
                    file_input = texture_shader.GetInput("file")
                    file_value = file_input.Get() if file_input else None
                    break
        if file_value is None:
            mdl_shader = next(
                (UsdShade.Shader(prim) for prim in shaders if UsdShade.Shader(prim).GetInput("diffuse_texture")),
                None,
            )
            file_input = mdl_shader.GetInput("diffuse_texture") if mdl_shader else None
            file_value = file_input.Get() if file_input else None
        if file_value is None:
            return None, color
        if isinstance(file_value, Sdf.AssetPath):
            texture_path = Path(file_value.resolvedPath) if file_value.resolvedPath else asset_dir / file_value.path
        else:
            texture_path = asset_dir / str(file_value)
        texture_path = texture_path.resolve()
        if not texture_path.is_relative_to(source_root.resolve()) or not texture_path.is_file():
            raise FileNotFoundError(f"Missing or unsafe delivery USD texture: {texture_path}")
        return texture_path, color

    def append_group(selected_faces: np.ndarray, binding_prim) -> None:
        texture_path, color = resolve_material(binding_prim)
        # The delivered USD Preview Surface omits the plate's source pink MDL
        # tint. Preserve the visibly pink dish shown by its bundled reference.
        if asset_name == "plate.usd" and texture_path is None:
            color = (0.78, 0.16, 0.39)
        key = (str(texture_path) if texture_path is not None else None, tuple(color))
        group_index = lookup.get(key)
        if group_index is None:
            group_index = len(groups)
            lookup[key] = group_index
            groups.append({"face_indices": [], "texture_path": texture_path, "color": color})
        groups[group_index]["face_indices"].extend(int(index) for index in selected_faces)

    for child in mesh_prim.GetChildren():
        if not child.IsA(UsdGeom.Subset):
            continue
        subset = UsdGeom.Subset(child)
        if subset.GetElementTypeAttr().Get() != UsdGeom.Tokens.face:
            raise ValueError(f"Unsupported delivery material subset: {child.GetPath()}")
        indices = np.asarray(subset.GetIndicesAttr().Get(), dtype=np.int64)
        if (
            not len(indices)
            or np.any(indices < 0)
            or np.any(indices >= face_count)
            or len(np.unique(indices)) != len(indices)
            or np.any(assigned[indices])
        ):
            raise ValueError(f"Invalid or overlapping delivery material subset: {child.GetPath()}")
        assigned[indices] = True
        append_group(indices, child)
    remaining = np.flatnonzero(~assigned)
    if len(remaining):
        append_group(remaining, mesh_prim)
    for group in groups:
        group["face_indices"] = np.asarray(group["face_indices"], dtype=np.int64)
    return groups


def _write_newton_plate_visual(destination: Path, source_usd_sha256: str) -> dict:
    """Merge the delivery plate's overlapping bottom faces for Newton rendering.

    The supplied plate is assembled from intersecting solids, with many top
    triangles at the same height. Newton's ray renderer chooses between their
    differently smoothed normals, causing speckles. Keep the imported source
    mesh and the collision model untouched; only replace that coplanar visual
    surface in a separate Newton URDF. The source hash and topology checks
    prevent applying this asset-specific correction to a different plate.
    """
    if source_usd_sha256 != _PLATE_USD_SHA256:
        raise ValueError("The delivered plate USD differs from the version supported by its Newton visual repair")
    original = destination / "visual_0.obj"
    mesh = trimesh.load(original, force="mesh", process=False)
    triangles = np.asarray(mesh.triangles)
    top_z = 0.002
    selected = np.all(np.isclose(triangles[:, :, 2], top_z, rtol=0.0, atol=1e-7), axis=1)
    selected &= np.asarray(mesh.face_normals)[:, 2] > 0.9
    xy = np.unique(np.round(triangles[selected, :, :2].reshape(-1, 2), 9), axis=0)
    if len(mesh.faces) != 968 or int(selected.sum()) != 132 or len(xy) < 3:
        raise ValueError("The delivered plate mesh topology no longer matches the Newton visual repair")
    hull = ConvexHull(xy)
    if len(hull.vertices) != 36 or float(mesh.area_faces[selected].sum()) < float(hull.volume) * 1.5:
        raise ValueError("The delivered plate bottom no longer has the expected overlapping convex surface")
    perimeter = xy[hull.vertices]
    top_vertices = np.column_stack((np.vstack((perimeter.mean(axis=0), perimeter)), np.full(len(perimeter) + 1, top_z)))
    top_faces = np.asarray([[0, i + 1, (i + 1) % len(perimeter) + 1] for i in range(len(perimeter))], dtype=np.int64)
    remainder_path = destination / "visual_0_newton_remainder.obj"
    top_path = destination / "visual_0_newton_top.obj"
    trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[~selected], process=False).export(remainder_path)
    trimesh.Trimesh(vertices=top_vertices, faces=top_faces, process=False).export(top_path)

    tree = ET.parse(destination / "object.urdf")
    link = tree.getroot().find("link")
    if link is None or len(link.findall("visual")) != 1:
        raise ValueError("The delivered plate URDF does not have the expected single visual")
    visual = link.find("visual")
    visual_mesh = visual.find("geometry/mesh")
    if visual_mesh is None or visual_mesh.get("filename") != original.name:
        raise ValueError("The delivered plate URDF does not reference its source visual mesh")
    visual_mesh.set("filename", remainder_path.name)
    top_visual = copy.deepcopy(visual)
    top_visual.find("geometry/mesh").set("filename", top_path.name)
    link.insert(list(link).index(visual) + 1, top_visual)
    newton_urdf = destination / "object_newton.urdf"
    tree.write(newton_urdf, encoding="utf-8", xml_declaration=True)
    return {
        "method": "replace overlapping coplanar bottom faces with one convex surface for Newton rendering",
        "source_visual": original.name,
        "source_visual_sha256": _sha256(original),
        "urdf": newton_urdf.name,
        "visual_meshes": [remainder_path.name, top_path.name],
        "visual_meshes_sha256": {path.name: _sha256(path) for path in (remainder_path, top_path)},
        "removed_overlapping_faces": int(selected.sum()),
        "replacement_faces": len(top_faces),
        "surface_height_m": top_z,
        "collision_geometry": "unchanged",
    }


def _convert_rigid_usd(
    source_usd: Path,
    output: Path,
    *,
    asset_id: str,
    body_name: str,
    mass_kg: float,
    source_archive_sha256: str,
) -> dict:
    """Convert one delivery USD asset, preserving diffuse UVs and authored colliders."""
    from pxr import Usd, UsdGeom

    source_usd, output = Path(source_usd).resolve(), Path(output).resolve()
    if output.exists():
        existing = output / "provenance.json"
        if existing.is_file():
            metadata = json.loads(existing.read_text(encoding="utf-8"))
            if (
                metadata.get("format") == _OBJECT_FORMAT
                and metadata.get("asset_id") == asset_id
                and metadata.get("source_archive_sha256") == source_archive_sha256
                and all((output / name).is_file() for name in ("object.urdf", "object.xml"))
            ):
                return metadata
        raise FileExistsError(f"Refusing to overwrite a different prepared G1 Apple object: {output}")
    stage = Usd.Stage.Open(str(source_usd))
    if (
        not stage
        or UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z
        or not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0)
    ):
        raise ValueError(f"Expected a Z-up, meter-scaled delivery USD asset: {source_usd}")

    from tools.icl.prepare_geniesim_objects import _inertial_values

    xform_cache = UsdGeom.XformCache()
    source_root = source_usd.parents[1]
    visual_files: list[dict] = []
    visual_bounds: list[np.ndarray] = []
    collision_parts: list[tuple[np.ndarray, np.ndarray]] = []
    texture_hashes: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="rvobj_", dir=output.parent) as temporary:
        staged = Path(temporary) / "asset"
        staged.mkdir()
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            is_collision = "/collisions/" in f"{prim.GetPath()}/"
            points, faces, uv_values, uv_corners, source_faces = _triangles(prim)
            matrix = np.asarray(xform_cache.GetLocalToWorldTransform(prim), dtype=np.float64)
            points = (points @ matrix[:3, :3] + matrix[3, :3]) * float(UsdGeom.GetStageMetersPerUnit(stage))
            if is_collision:
                collision_parts.append((points, faces))
                continue
            if not len(faces):
                continue
            visual_bounds.extend([points.min(axis=0), points.max(axis=0)])
            for group in _delivery_material_groups(stage, prim, source_root, source_usd.parent, source_usd.name):
                selected = np.isin(source_faces, group["face_indices"])
                if not np.any(selected):
                    continue
                index = len(visual_files)
                mesh_name, material_name = f"visual_{index}.obj", f"material_{index}"
                texture_name = None
                source_texture = group["texture_path"]
                if source_texture is not None:
                    source_texture = Path(source_texture).resolve()
                    if not source_texture.is_relative_to(source_root) or not source_texture.is_file():
                        raise FileNotFoundError(f"Unsafe or missing delivery texture: {source_texture}")
                    texture_name = f"material_{index}.png"
                    with Image.open(source_texture) as texture:
                        texture = texture.convert("RGBA" if "A" in texture.getbands() else "RGB")
                        texture.thumbnail((_MAX_TEXTURE_SIZE, _MAX_TEXTURE_SIZE), Image.Resampling.LANCZOS)
                        texture.save(staged / texture_name, format="PNG", optimize=True)
                    texture_hashes[source_texture.relative_to(source_root).as_posix()] = _sha256(source_texture)
                    material_file = f"newmtl {material_name}\nKd 1 1 1\nmap_Kd {texture_name}\n"
                else:
                    material_file = (
                        f"newmtl {material_name}\nKd "
                        + " ".join(f"{float(value):.8g}" for value in group["color"])
                        + "\n"
                    )
                _write_obj(
                    staged / mesh_name,
                    points,
                    faces[selected],
                    uv_values,
                    uv_corners[selected] if uv_corners is not None else None,
                    material_name,
                )
                (staged / f"{material_name}.mtl").write_text(material_file, encoding="utf-8")
                visual_files.append({"obj": mesh_name, "texture": texture_name, "color": tuple(group["color"])})

        if not visual_files or not visual_bounds:
            raise ValueError(f"No visible USD mesh geometry found in {source_usd}")
        bounds = np.asarray([np.min(visual_bounds, axis=0), np.max(visual_bounds, axis=0)], dtype=np.float64)
        if not collision_parts:
            extents = bounds[1] - bounds[0]
            hull = trimesh.creation.box(extents=extents)
            hull.apply_translation(bounds.mean(axis=0))
            collision_parts.append((np.asarray(hull.vertices), np.asarray(hull.faces)))
        center, raw_inertia, inertia = _inertial_values(mass_kg, bounds)
        _write_wrappers(staged, body_name, mass_kg, bounds, visual_files, collision_parts)
        source_usd_sha256 = _sha256(source_usd)
        newton_visual = _write_newton_plate_visual(staged, source_usd_sha256) if body_name == "pink_plate" else None
        provenance = {
            "format": _OBJECT_FORMAT,
            "asset_id": asset_id,
            "source_repository": "user-provided G1 apple scene delivery bundle",
            "source_archive_sha256": source_archive_sha256,
            "source_usd": source_usd.relative_to(source_root).as_posix(),
            "source_usd_sha256": source_usd_sha256,
            "source_texture_sha256": texture_hashes,
            "mass_kg": float(mass_kg),
            "bounds_m": bounds.tolist(),
            "inertia_diagonal_kg_m2": inertia.tolist(),
            "inertia_floor_adjustments": [
                {"axis": axis, "before": float(raw), "after": float(final)}
                for axis, raw, final in zip(("x", "y", "z"), raw_inertia, inertia, strict=True)
                if raw < final
            ],
            "visual_mesh_count": len(visual_files),
            "collision": "authored USD collision meshes when present; otherwise a conservative bounding box",
            "appearance": f"Diffuse USD textures use UVs and are capped at {_MAX_TEXTURE_SIZE}px",
            "limitations": "Static plate surface collision and object mass/inertia are approximations",
        }
        if newton_visual is not None:
            provenance["newton_visual"] = newton_visual
        (staged / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        shutil.move(str(staged), output)
    return provenance


def _scene_primitives(source_root: Path) -> list[dict]:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(source_root / "usd" / "scene.usda"))
    if not stage:
        raise ValueError("Could not open the delivered USD scene")
    xform_cache = UsdGeom.XformCache()
    result = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Cube) or not str(prim.GetPath()).startswith("/World/Environment/"):
            continue
        cube = UsdGeom.Cube(prim)
        size = float(cube.GetSizeAttr().Get())
        matrix = np.asarray(xform_cache.GetLocalToWorldTransform(prim), dtype=np.float64)
        axes_scale = np.linalg.norm(matrix[:3, :3], axis=1)
        dimensions = size * axes_scale
        rotation_matrix = matrix[:3, :3] / axes_scale[:, None]
        quaternion_xyzw = Rotation.from_matrix(rotation_matrix.T).as_quat()
        display_color = UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayColor").Get()
        color = tuple(float(value) for value in display_color[0]) if display_color else (0.7, 0.7, 0.7)
        name = re.sub(r"(?<!^)(?=[A-Z])", "_", prim.GetName()).lower()
        result.append({
            "name": name,
            "size": dimensions.tolist(),
            "position": matrix[3, :3].tolist(),
            "orientation_wxyz": [float(quaternion_xyzw[3]), *map(float, quaternion_xyzw[:3])],
            "color": list(color),
            "fixed": True,
        })
    expected_names = {"ground", "table_top", "table_pedestal", "back_wall"}
    if {item["name"] for item in result} != expected_names:
        raise ValueError("The delivered G1 apple scene does not have the expected table/environment cubes")
    # MetaSim supplies its own backend ground. Omit both the source floor cube
    # (which overlaps that ground) and the rear wall, so the ICL scene stays
    # focused on the robot, table, apple, and plate.
    return [item for item in result if item["name"] not in {"ground", "back_wall"}]


def _robot_initial_state(source_root: Path, robot_root: Path) -> dict:
    from roboverse_pack.tasks.icl.fruit_sorting import CONTROL_JOINTS, build_geniesim_g1_robot

    scene_text = (source_root / "usd" / "scene.usda").read_text(encoding="utf-8")
    robot_block = re.search(r'def Xform "Robot".*?\{(?P<body>.*?)\n\s*\}', scene_text, re.DOTALL)
    if not robot_block:
        raise ValueError("The delivered USD scene is missing its G1 root transform")
    translation_match = re.search(r"xformOp:translate\s*=\s*\(([^)]+)\)", robot_block["body"])
    rotation_match = re.search(r"xformOp:rotateXYZ\s*=\s*\(([^)]+)\)", robot_block["body"])
    if not translation_match or not rotation_match:
        raise ValueError("The delivered G1 root transform needs translate and rotateXYZ values")
    position = tuple(float(value.strip()) for value in translation_match.group(1).split(","))
    euler_xyz = tuple(float(value.strip()) for value in rotation_match.group(1).split(","))
    quaternion_xyzw = Rotation.from_euler("xyz", euler_xyz, degrees=True).as_quat()

    pose_path = source_root / "usd" / "robot" / "g1_task_pose.usda"
    pose_values = {}
    for match in _JOINT_POSE_PATTERN.finditer(pose_path.read_text(encoding="utf-8")):
        value = float(match["value"])
        pose_values[match["name"]] = math.radians(value) if match["kind"] == "angular" else value
    robot = build_geniesim_g1_robot(robot_root)
    joint_positions = {name: pose_values[name] for name in CONTROL_JOINTS if name in pose_values}
    missing = (
        set(CONTROL_JOINTS) - set(joint_positions) - {"idx41_gripper_l_outer_joint1", "idx81_gripper_r_outer_joint1"}
    )
    if missing:
        raise ValueError(f"The delivered task pose is missing G1 controls: {sorted(missing)}")
    for gripper in ("idx41_gripper_l_outer_joint1", "idx81_gripper_r_outer_joint1"):
        joint_positions.setdefault(gripper, robot.default_joint_positions[gripper])
    return {
        "position": list(position),
        "orientation_wxyz": [float(quaternion_xyzw[3]), *map(float, quaternion_xyzw[:3])],
        "joint_positions": joint_positions,
        "source_pose": "usd/robot/g1_task_pose.usda and usd/scene.usda",
    }


def _build_scene(source_root: Path, output: Path, robot_root: Path, archive_hash: str, source: dict) -> Path:
    """Build converted objects and scene metadata into a fresh staging directory."""
    output.mkdir(parents=True, exist_ok=False)
    asset_specs = (("apple", 0.15), ("pink_plate", 0.12))
    object_data = {}
    for object_name, mass in asset_specs:
        asset_root = output / object_name
        usd_name = "plate" if object_name == "pink_plate" else object_name
        usd_path = source_root / "usd" / "assets" / f"{usd_name}.usd"
        provenance = _convert_rigid_usd(
            usd_path,
            asset_root,
            asset_id=f"{_BUNDLE_ID}_{object_name}",
            body_name=object_name,
            mass_kg=mass,
            source_archive_sha256=archive_hash,
        )
        object_data[object_name] = provenance

    robot_state = _robot_initial_state(source_root, robot_root)
    primitives = _scene_primitives(source_root)
    placements = {"apple": (-0.12, 0.475, 0.78), "pink_plate": (0.13, 0.525, 0.78)}
    bounds = {
        "apple": object_data["apple"]["bounds_m"],
        "pink_plate": object_data["pink_plate"]["bounds_m"],
    }
    objects = {
        name: {
            "fixed": name == "pink_plate",
            "position": list(position),
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "bounds_m": bounds[name],
        }
        for name, position in placements.items()
    }
    objects["pink_plate"]["goal_bounds_m"] = [[-0.065, -0.065, 0.02], [0.065, 0.065, 0.075]]
    objects["pink_plate"]["newton_urdf"] = object_data["pink_plate"]["newton_visual"]["urdf"]
    scene_metadata = {
        "format": _SCENE_FORMAT,
        "scene_id": _BUNDLE_ID,
        "task": "pick the apple and place it on the pink plate",
        "target": "apple",
        "destination": "pink_plate",
        "objects": objects,
        "primitives": primitives,
        "omitted_source_primitives": ["back_wall", "ground"],
        "ground_rendering": "MetaSim backend ground; source USD floor cube omitted to prevent coplanar z-fighting",
        "robot_initial_state": robot_state,
        "source": {
            "archive": _BUNDLE_ID + ".zip",
            "archive_sha256": archive_hash,
            "delivery_manifest_sha256": source["delivery_manifest_sha256"],
            "source_manifest_sha256": source["manifest_sha256"],
        },
        "success_condition": "apple center remains inside the plate goal region for two consecutive observations",
        "limitations": [
            "The original scene estimates table dimensions; they are not measured.",
            "Apple mass, plate collision, and inertias use portable approximations.",
            "The state-based checker does not verify finger contact or grasp force.",
            "Source real-camera calibration and robot joint zero/sign alignment remain unverified.",
        ],
    }
    scene_path = output / "scene.json"
    encoded = json.dumps(scene_metadata, indent=2, ensure_ascii=False) + "\n"
    if scene_path.exists():
        existing = json.loads(scene_path.read_text(encoding="utf-8"))
        if existing != scene_metadata:
            raise FileExistsError(f"Refusing to replace a different G1 Apple scene specification: {scene_path}")
    else:
        scene_path.write_text(encoded, encoding="utf-8")
    return scene_path


def _validate_prepared_scene(output: Path, archive_hash: str) -> Path:
    """Check an existing preparation before treating it as an idempotent result."""
    scene_path = output / "scene.json"
    if not scene_path.is_file():
        raise FileExistsError(f"Refusing to overwrite an incomplete G1 Apple task scene: {output}")
    metadata = json.loads(scene_path.read_text(encoding="utf-8"))
    if metadata.get("format") != _SCENE_FORMAT or metadata.get("source", {}).get("archive_sha256") != archive_hash:
        raise FileExistsError(f"Prepared G1 Apple task scene belongs to a different source bundle: {output}")
    primitive_names = {item.get("name") for item in metadata.get("primitives", [])}
    if primitive_names != {"table_top", "table_pedestal"} or metadata.get("omitted_source_primitives") != [
        "back_wall", "ground"
    ]:
        raise FileExistsError(f"Prepared G1 Apple task scene must omit its source wall and floor: {scene_path}")
    for name in ("apple", "pink_plate"):
        asset_root = output / name
        provenance_path = asset_root / "provenance.json"
        if not all((asset_root / filename).is_file() for filename in ("object.urdf", "object.xml")):
            raise FileExistsError(f"Prepared G1 Apple object is incomplete: {asset_root}")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("source_archive_sha256") != archive_hash or provenance.get("format") != _OBJECT_FORMAT:
            raise FileExistsError(f"Prepared G1 Apple object has different source provenance: {asset_root}")
        if name == "pink_plate":
            newton_visual = provenance.get("newton_visual", {})
            if not isinstance(newton_visual, dict):
                raise FileExistsError(f"Prepared G1 Apple plate has invalid Newton visual provenance: {asset_root}")
            meshes = newton_visual.get("visual_meshes", [])
            mesh_hashes = newton_visual.get("visual_meshes_sha256", {})
            if not isinstance(meshes, list) or not isinstance(mesh_hashes, dict):
                raise FileExistsError(f"Prepared G1 Apple plate has invalid Newton visual provenance: {asset_root}")
            required = [newton_visual.get("urdf"), newton_visual.get("source_visual"), *meshes]
            if len(meshes) != 2 or not all(
                isinstance(item, str)
                and bool(item)
                and Path(item).name == item
                and "\\" not in item
                and (asset_root / item).is_file()
                for item in required
            ):
                raise FileExistsError(f"Prepared G1 Apple plate is missing its Newton visual meshes: {asset_root}")
            hashes = {newton_visual["source_visual"]: newton_visual.get("source_visual_sha256")}
            hashes.update(mesh_hashes)
            if set(hashes) != {newton_visual["source_visual"], *meshes} or any(
                not isinstance(expected, str) or _sha256(asset_root / item) != expected
                for item, expected in hashes.items()
            ):
                raise FileExistsError(f"Prepared G1 Apple plate visual meshes failed hash verification: {asset_root}")
    return scene_path


def prepare_scene(archive: Path, source_root: Path, output: Path, robot_root: Path) -> Path:
    """Prepare atomically so an interrupted conversion never leaves partial assets."""
    archive = Path(archive).resolve()
    source_root, output, robot_root = Path(source_root).resolve(), Path(output).resolve(), Path(robot_root).resolve()
    archive_hash = _extract_bundle(archive, source_root)
    source = _verify_source_bundle(source_root, archive_hash)
    if output.exists():
        return _validate_prepared_scene(output, archive_hash)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rv_", dir=output.parent) as temporary:
        staged_output = Path(temporary) / "scene"
        staged_scene = _build_scene(source_root, staged_output, robot_root, archive_hash, source)
        _validate_prepared_scene(staged_output, archive_hash)
        if output.exists():
            raise FileExistsError(f"G1 Apple scene output appeared during preparation: {output}")
        shutil.move(str(staged_output), output)
    return output / "scene.json"


def main() -> None:
    from roboverse_pack.tasks.icl.fruit_sorting import ROBOT_ROOT

    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Verified user-provided delivery ZIP")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=root / "datasets" / "icl" / "sources" / _BUNDLE_ID,
        help="Local extracted source bundle path (default: datasets/icl/sources)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "roboverse_data" / "icl" / "scenes" / f"{_BUNDLE_ID}_r4",
        help="Prepared task scene path under roboverse_data/icl/scenes",
    )
    parser.add_argument("--robot-root", type=Path, default=ROBOT_ROOT, help="Prepared AgiBot G1 robot model")
    args = parser.parse_args()
    scene_path = prepare_scene(args.archive, args.source_root, args.output, args.robot_root)
    print(f"Prepared G1 Apple delivery task scene: {scene_path}")


if __name__ == "__main__":
    main()
