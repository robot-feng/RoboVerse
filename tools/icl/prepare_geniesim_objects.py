"""Convert rigid Genie Sim USD assets into portable MetaSim object wrappers."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from roboverse_pack.tasks.icl.asset_sources import HISTORICAL_ASSETS, source_revision
from tools.icl.prepare_geniesim_g1 import REVISION, _verify

_OBJECT_FORMAT_V1 = "roboverse.icl.geniesim_rigid_object.v1"
_OBJECT_FORMAT_V2 = "roboverse.icl.geniesim_rigid_object.v2"
_OBJECT_FORMAT = "roboverse.icl.geniesim_rigid_object.v3"
_USD_PHYSICS_OBJECT_FORMAT = "roboverse.icl.geniesim_rigid_object.v4"
_MIN_OBJECT_INERTIA = 1.1e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_rigid_physics(stage) -> dict:
    """Extract authored rigid masses and collision-bound physics materials from composed USD.

    This is asset-layer evidence, before any scene-level overrides. It does not
    infer missing mass from visual bounds or conflate static/dynamic friction.
    """
    from pxr import Usd, UsdPhysics, UsdShade

    errors = stage.GetCompositionErrors()
    if errors:
        raise ValueError(f"USD asset has unresolved composition errors: {errors[0]}")
    kilograms_per_unit = float(UsdPhysics.GetStageKilogramsPerUnit(stage))
    if not np.isfinite(kilograms_per_unit) or kilograms_per_unit <= 0:
        raise ValueError("USD asset has invalid mass units")
    bodies = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        rigid = UsdPhysics.RigidBodyAPI(prim)
        if rigid.GetRigidBodyEnabledAttr().Get() is False:
            continue
        mass_api = UsdPhysics.MassAPI(prim)
        # USD explicitly gives an authored ancestor mass precedence over an
        # authored descendant mass. Keep its source path for provenance.
        chain = []
        ancestor = prim
        while ancestor and ancestor.IsValid():
            chain.append(ancestor)
            ancestor = ancestor.GetParent()
        mass_value, mass_path = None, None
        for ancestor in reversed(chain):
            attr = UsdPhysics.MassAPI(ancestor).GetMassAttr()
            value = attr.Get() if attr and attr.HasAuthoredValueOpinion() else None
            if value is not None and float(value) > 0:
                mass_value, mass_path = value, str(ancestor.GetPath())
                break
        mass = float(mass_value) * kilograms_per_unit if mass_value is not None and float(mass_value) > 0 else None
        if mass is not None and not np.isfinite(mass):
            raise ValueError(f"Non-finite USD mass on {prim.GetPath()}")
        colliders = []
        for child in Usd.PrimRange(prim):
            owner = child
            while owner and owner != prim and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                owner = owner.GetParent()
            if owner != prim:
                continue
            if not child.HasAPI(UsdPhysics.CollisionAPI):
                continue
            collision = UsdPhysics.CollisionAPI(child)
            if collision.GetCollisionEnabledAttr().Get() is False:
                continue
            material, _ = UsdShade.MaterialBindingAPI(child).ComputeBoundMaterial("physics")
            entry = {"prim_path": str(child.GetPath()), "material_path": None, "static_friction": None,
                     "dynamic_friction": None}
            if material and material.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
                api = UsdPhysics.MaterialAPI(material.GetPrim())
                entry["material_path"] = str(material.GetPath())
                for key, value in (("static_friction", api.GetStaticFrictionAttr()),
                                   ("dynamic_friction", api.GetDynamicFrictionAttr())):
                    if value and value.HasAuthoredValueOpinion():
                        coefficient = float(value.Get())
                        if not np.isfinite(coefficient) or coefficient < 0:
                            raise ValueError(f"Invalid USD friction on {material.GetPath()}")
                        entry[key] = coefficient
            colliders.append(entry)
        authored_inertial = {}
        for name, attr in (
            ("center_of_mass", mass_api.GetCenterOfMassAttr()),
            ("diagonal_inertia", mass_api.GetDiagonalInertiaAttr()),
            ("principal_axes", mass_api.GetPrincipalAxesAttr()),
            ("density", mass_api.GetDensityAttr()),
        ):
            if attr and attr.HasAuthoredValueOpinion():
                value = attr.Get()
                if hasattr(value, "GetReal"):
                    authored_inertial[name] = [float(value.GetReal()), *[float(x) for x in value.GetImaginary()]]
                elif isinstance(value, (int, float)):
                    authored_inertial[name] = float(value)
                else:
                    authored_inertial[name] = [float(x) for x in value]
        bodies.append({"prim_path": str(prim.GetPath()), "authored_mass_kg": mass,
                       "mass_authored_prim_path": mass_path, "authored_inertial": authored_inertial,
                       "colliders": colliders})
    return {"format": "roboverse.icl.geniesim_source_rigid_physics.v1",
            "kilograms_per_unit": kilograms_per_unit, "rigid_bodies": bodies}


def _single_usd_rigid_physics(stage, source: Path, asset_id: str) -> dict:
    """Resolve one rigid asset's USD mass and a documented single-friction proxy."""
    physics = _source_rigid_physics(stage)
    bodies = physics["rigid_bodies"]
    if len(bodies) != 1 or bodies[0]["authored_mass_kg"] is None:
        raise ValueError(f"{asset_id} needs one rigid body with an explicit source USD mass")
    if any(key in bodies[0]["authored_inertial"] for key in ("center_of_mass", "diagonal_inertia", "principal_axes")):
        raise ValueError(f"{asset_id} has authored inertial properties requiring an exact frame conversion")
    colliders = bodies[0]["colliders"]
    friction_pairs = {
        (item["static_friction"], item["dynamic_friction"])
        for item in colliders
        if item["material_path"] is not None
    }
    if len(friction_pairs) > 1:
        raise ValueError(f"{asset_id} has multiple physical materials; per-collider mapping is required")
    if any(item["material_path"] is None for item in colliders) and friction_pairs:
        raise ValueError(f"{asset_id} mixes bound and unbound collider materials")
    static, dynamic = next(iter(friction_pairs), (None, None))
    friction_proxy = dynamic if dynamic is not None else static
    layer_hashes = {}
    for layer in stage.GetUsedLayers():
        if not layer.realPath:
            continue  # anonymous USD session layers are not source assets
        path = Path(layer.realPath).resolve()
        if not path.is_relative_to(source.resolve()):
            raise ValueError(f"USD layer is outside the verified source tree: {path}")
        relative = path.relative_to(source.resolve()).as_posix()
        verified = _verify_asset_file(source, relative, asset_id)
        layer_hashes[relative] = _sha256(verified)
    if not layer_hashes:
        raise ValueError(f"{asset_id} has no file-backed USD source layers")
    physics["used_layer_sha256"] = layer_hashes
    physics["scene_overrides_applied"] = False
    physics["source_static_friction"] = static
    physics["source_dynamic_friction"] = dynamic
    physics["single_coefficient_proxy"] = friction_proxy
    physics["single_coefficient_proxy_reason"] = (
        "source dynamic friction, or static friction if dynamic is unauthored; "
        "solver friction and contact-pair combination differ from PhysX"
        if friction_proxy is not None else "source collider has no authored bound friction"
    )
    return physics


def _source_subframes(stage) -> dict[str, list[list[float]]]:
    """Read verified source entity frame relative to the converted object root."""
    from pxr import UsdGeom

    root = stage.GetPrimAtPath("/World")
    entity = stage.GetPrimAtPath("/World/entity")
    if not root or not entity:
        return {}
    transform, reset = UsdGeom.XformCache().ComputeRelativeTransform(entity, root)
    matrix = np.asarray(transform, dtype=np.float64).T
    rotation = matrix[:3, :3]
    if (reset or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1])
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4)):
        raise ValueError("Source entity frame is not a rigid transform relative to /World")
    return {"entity": matrix.tolist()}


def refresh_source_subframes(source: Path, output: Path, asset_id: str) -> None:
    """Update only frame provenance after verifying all original asset inputs."""
    from pxr import Usd

    provenance_path = output / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    directory = _find_asset(source, asset_id)
    if (provenance.get("format") != _OBJECT_FORMAT or provenance.get("asset_id") != asset_id
            or provenance.get("source_revision") != source_revision(asset_id)
            or provenance.get("source_asset_directory") != directory):
        raise ValueError(f"Prepared object provenance does not match requested source: {asset_id}")
    source_hashes = provenance.get("source_files_sha256", {})
    layer = provenance.get("source_layer", "Aligned.usda")
    if layer not in source_hashes:
        raise ValueError(f"Prepared object is missing source-layer hash: {asset_id}")
    for filename, expected in source_hashes.items():
        if _sha256(_verify_asset_file(source, f"{directory}/{filename}", asset_id)) != expected:
            raise ValueError(f"Source checksum mismatch for {asset_id}/{filename}")
    stage = Usd.Stage.Open(str(source / directory / layer))
    if not stage:
        raise ValueError(f"Cannot open source layer for {asset_id}")
    provenance["source_subframes"] = _source_subframes(stage)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output, delete=False) as stream:
        staged = Path(stream.name)
        json.dump(provenance, stream, indent=2)
        stream.write("\n")
    staged.replace(provenance_path)


def _inertial_values(mass: float, bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = np.asarray(bounds[1] - bounds[0], dtype=np.float64)
    raw = np.asarray(
        [
            mass / 12 * (size[1] ** 2 + size[2] ** 2),
            mass / 12 * (size[0] ** 2 + size[2] ** 2),
            mass / 12 * (size[0] ** 2 + size[1] ** 2),
        ],
        dtype=np.float64,
    )
    return bounds.mean(axis=0), raw, np.maximum(raw, _MIN_OBJECT_INERTIA)


def _upgrade_prepared_object(source: Path, output: Path, asset_id: str) -> None:
    """Upgrade verified wrappers, normalize texture tint and floor tiny inertias for Newton."""
    from pxr import Usd, UsdGeom

    provenance_path = output / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (
        provenance.get("format") not in {_OBJECT_FORMAT_V1, _OBJECT_FORMAT_V2, _OBJECT_FORMAT}
        or provenance.get("asset_id") != asset_id
        or provenance.get("source_revision") != REVISION
        or provenance.get("source_asset_directory") != _find_asset(source, asset_id)
    ):
        raise ValueError(f"Existing asset is not a verified older Genie Sim conversion: {output}")
    for filename, expected_hash in provenance.get("source_files_sha256", {}).items():
        verified = _verify(source, f"{provenance['source_asset_directory']}/{filename}")
        if _sha256(verified) != expected_hash:
            raise ValueError(f"Pinned source file changed for {asset_id}: {filename}")

    source_asset = source / provenance["source_asset_directory"]
    stage = Usd.Stage.Open(str(source_asset / "Aligned.usda"))
    if not stage:
        raise ValueError(f"Could not reopen pinned source USD for {asset_id}")
    material_records: list[tuple[tuple[float, float, float], Path | None]] = []
    for mesh_prim in (prim for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/entity")) if prim.IsA(UsdGeom.Mesh)):
        _, _, _, _, source_faces = _triangles(mesh_prim)
        for group in _material_groups(stage, mesh_prim, source_asset):
            if np.any(np.isin(source_faces, group["face_indices"])):
                material_records.append((tuple(group["color"]), group["texture_path"]))

    urdf_path = output / "object.urdf"
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    upgraded = 0
    visuals = root.findall(".//visual")
    if len(visuals) != len(material_records):
        raise ValueError(f"Source material groups no longer match converted meshes for {asset_id}")
    for index, (visual, (color, source_texture)) in enumerate(zip(visuals, material_records, strict=True)):
        mesh = visual.find("geometry/mesh")
        if mesh is None:
            raise ValueError(f"Unexpected non-mesh visual in prepared asset {asset_id}")
        match = re.fullmatch(r"visual_(\d+)\.obj", Path(mesh.get("filename", "")).name)
        if match is None or int(match.group(1)) != index:
            raise ValueError(f"Unexpected visual mesh in prepared asset {asset_id}")
        material = visual.find("material")
        if material is None:
            raise ValueError(f"Missing visual material in prepared asset {asset_id}")
        color_element = material.find("color")
        if color_element is None:
            color_element = ET.SubElement(material, "color")
        urdf_color = (1.0, 1.0, 1.0) if source_texture is not None else color
        color_element.set("rgba", " ".join(f"{value:.9g}" for value in (*urdf_color, 1.0)))
        texture_path = output / f"material_{index}.png"
        if source_texture is not None:
            if not texture_path.is_file():
                raise FileNotFoundError(f"Converted texture is missing for {asset_id}: {texture_path}")
            texture = material.find("texture")
            if texture is None:
                texture = ET.SubElement(material, "texture")
            texture.set("filename", texture_path.name)
            upgraded += 1
        elif material.find("texture") is not None:
            raise ValueError(f"Converted asset {asset_id} has an unexpected texture on visual_{index}")
    bounds = np.asarray(provenance.get("bounds_m"), dtype=np.float64)
    mass = float(provenance.get("mass_kg", 0.0))
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or mass <= 0:
        raise ValueError(f"Prepared object has invalid inertial provenance: {asset_id}")
    center, raw_inertia, final_inertia = _inertial_values(mass, bounds)
    inertial = root.find("link/inertial")
    if inertial is None:
        raise ValueError(f"Prepared URDF has no root inertia for {asset_id}")
    inertial_origin = inertial.find("origin")
    if inertial_origin is None:
        inertial_origin = ET.SubElement(inertial, "origin")
    inertial_origin.set("xyz", " ".join(f"{value:.9g}" for value in center))
    inertia = inertial.find("inertia")
    if inertia is None:
        raise ValueError(f"Prepared URDF has no inertia tensor for {asset_id}")
    for key, value in zip(("ixx", "iyy", "izz"), final_inertia, strict=True):
        inertia.set(key, f"{value:.9g}")
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)

    mjcf_path = output / "object.xml"
    mjcf_tree = ET.parse(mjcf_path)
    mjcf_root = mjcf_tree.getroot()
    for index, (color, _) in enumerate(material_records):
        material = mjcf_root.find(f"asset/material[@name='material_{index}']")
        if material is None:
            raise ValueError(f"Converted MJCF material_{index} is missing for {asset_id}")
        material.set("rgba", " ".join(f"{value:.9g}" for value in (*color, 1.0)))
    mjcf_inertial = mjcf_root.find(".//inertial")
    if mjcf_inertial is None:
        raise ValueError(f"Prepared MJCF has no inertial element for {asset_id}")
    mjcf_inertial.set("pos", " ".join(f"{value:.9g}" for value in center))
    mjcf_inertial.set("mass", f"{mass:.9g}")
    mjcf_inertial.set("diaginertia", " ".join(f"{value:.9g}" for value in final_inertia))
    mjcf_tree.write(mjcf_path, encoding="utf-8", xml_declaration=True)

    provenance["format"] = _OBJECT_FORMAT
    provenance["appearance"] = (
        "MJCF retains source diffuse colors; textured URDF visuals use neutral white so Newton does not multiply the source texture by a second tint; non-diffuse PBR maps are omitted"
    )
    provenance["urdf_textured_visual_count"] = upgraded
    provenance["urdf_textured_material_tint"] = "neutral_white"
    provenance["inertia_floor_kg_m2"] = _MIN_OBJECT_INERTIA
    provenance["inertia_diagonal_kg_m2"] = final_inertia.tolist()
    provenance["inertia_floor_adjustments"] = [
        {"axis": axis, "before": float(before), "after": _MIN_OBJECT_INERTIA}
        for axis, before in zip(("x", "y", "z"), raw_inertia, strict=True)
        if before < _MIN_OBJECT_INERTIA
    ]
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def _find_asset(source: Path, asset_id: str) -> str:
    manifest = json.loads((source / "roboverse_source_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("repo") != "agibot-world/GenieSimAssets" or manifest.get("revision") != source_revision(asset_id):
        raise ValueError("Genie Sim asset source is not at the pinned revision")
    matches = sorted({
        str(Path(item["rfilename"]).parent).replace("\\", "/")
        for item in manifest["files"]
        if item["rfilename"].endswith(f"/{asset_id}/Aligned.usda")
    })
    # Some IDs are shared by the object library and the baked scene/background
    # cache (or by Genie-object and GDK-scene exports). Always choose the
    # authored object-library entry when it exists; scene exports are often
    # pre-positioned or collision-baked for one particular room.
    object_library = [path for path in matches if path.startswith("objects/")]
    if object_library:
        matches = object_library
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one asset directory for {asset_id}; manifest matches: {matches}")
    if asset_id in HISTORICAL_ASSETS and matches[0] != HISTORICAL_ASSETS[asset_id]["directory"]:
        raise ValueError(f"Unexpected historical asset directory: {matches[0]}")
    return matches[0]


def _verify_asset_file(source: Path, relative: str, asset_id: str) -> Path:
    """Verify official manifest entries under the asset-specific pinned revision."""
    manifest = json.loads((source / "roboverse_source_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("repo") != "agibot-world/GenieSimAssets" or manifest.get("revision") != source_revision(asset_id):
        raise ValueError("Expected the asset-specific pinned official Genie Sim manifest")
    entries = [item for item in manifest["files"] if item["rfilename"] == relative]
    if len(entries) != 1:
        raise ValueError(f"File absent from source manifest: {relative}")
    return _verify(source, relative, entry=entries[0])


def _asset_texture(stage, mesh_prim, asset_dir: Path) -> tuple[Path | None, tuple[float, float, float]]:
    from pxr import Usd, UsdShade

    material, _ = UsdShade.MaterialBindingAPI(mesh_prim).ComputeBoundMaterial()
    if not material:
        return None, (0.7, 0.7, 0.7)
    shaders = [prim for prim in Usd.PrimRange(material.GetPrim()) if prim.IsA(UsdShade.Shader)]
    if len(shaders) != 1:
        raise ValueError(f"Expected one direct material shader on {mesh_prim.GetPath()}")
    shader = UsdShade.Shader(shaders[0])
    color = shader.GetInput("diffuse_color_constant")
    diffuse = tuple(float(value) for value in color.Get())[:3] if color and color.Get() else (0.7, 0.7, 0.7)
    texture = shader.GetInput("diffuse_texture")
    if not texture or not texture.Get():
        return None, diffuse
    value = texture.Get()
    resolved = Path(value.resolvedPath) if value.resolvedPath else asset_dir / value.path
    resolved = resolved.resolve()
    if not resolved.is_relative_to(asset_dir.resolve()) or not resolved.is_file():
        raise FileNotFoundError(f"Missing or unsafe USD color texture: {value.path}")
    return resolved, diffuse


def _triangles(mesh_prim):
    from pxr import UsdGeom

    mesh = UsdGeom.Mesh(mesh_prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    if not len(points) or np.any(counts < 3) or int(counts.sum()) != len(indices):
        raise ValueError(f"Invalid USD topology: {mesh_prim.GetPath()}")
    faces, face_corners, source_faces, offset = [], [], [], 0
    for face_index, count in enumerate(counts):
        face = indices[offset : offset + count]
        for index in range(1, count - 1):
            faces.append((face[0], face[index], face[index + 1]))
            face_corners.append((offset, offset + index, offset + index + 1))
            source_faces.append(face_index)
        offset += count
    primvar = UsdGeom.PrimvarsAPI(mesh_prim).GetPrimvar("st")
    uv_values = np.asarray(primvar.ComputeFlattened(), dtype=np.float64) if primvar else np.empty((0, 2))
    uv_corners = None
    if len(uv_values):
        if primvar.GetInterpolation() == UsdGeom.Tokens.faceVarying:
            uv_corners = np.asarray(face_corners, dtype=np.int64)
        elif primvar.GetInterpolation() == UsdGeom.Tokens.vertex:
            uv_corners = np.asarray(faces, dtype=np.int64)
        elif primvar.GetInterpolation() == UsdGeom.Tokens.constant and len(uv_values) == 1:
            uv_corners = np.zeros((len(faces), 3), dtype=np.int64)
        else:
            raise ValueError(f"Unsupported USD UV interpolation on {mesh_prim.GetPath()}: {primvar.GetInterpolation()}")
        if int(uv_corners.max(initial=-1)) >= len(uv_values):
            raise ValueError(f"Invalid USD UV indexing on {mesh_prim.GetPath()}")
    return points, np.asarray(faces, dtype=np.int64), uv_values, uv_corners, np.asarray(source_faces, dtype=np.int64)


def _material_groups(stage, mesh_prim, asset_dir: Path) -> list[dict]:
    """Group source polygon faces by their bound USD diffuse material."""
    from pxr import UsdGeom

    face_count = len(UsdGeom.Mesh(mesh_prim).GetFaceVertexCountsAttr().Get())
    assigned = np.zeros(face_count, dtype=bool)
    groups: list[dict] = []
    lookup: dict[tuple[str | None, tuple[float, float, float]], int] = {}

    def add_group(selected_faces: np.ndarray, binding_prim) -> None:
        texture_path, color = _asset_texture(stage, binding_prim, asset_dir)
        key = (str(texture_path) if texture_path is not None else None, tuple(color))
        group_index = lookup.get(key)
        if group_index is None:
            group_index = len(groups)
            lookup[key] = group_index
            groups.append({"face_indices": [], "texture_path": texture_path, "color": color})
        groups[group_index]["face_indices"].extend(int(index) for index in selected_faces)

    for subset_prim in mesh_prim.GetChildren():
        if not subset_prim.IsA(UsdGeom.Subset):
            continue
        subset = UsdGeom.Subset(subset_prim)
        element_type = subset.GetElementTypeAttr().Get()
        if element_type != UsdGeom.Tokens.face:
            raise ValueError(f"Unsupported material subset element type {element_type!r}: {subset_prim.GetPath()}")
        selected = np.asarray(subset.GetIndicesAttr().Get(), dtype=np.int64)
        if (
            not len(selected)
            or np.any(selected < 0)
            or np.any(selected >= face_count)
            or len(np.unique(selected)) != len(selected)
            or np.any(assigned[selected])
        ):
            raise ValueError(f"Invalid or overlapping material subset: {subset_prim.GetPath()}")
        assigned[selected] = True
        add_group(selected, subset_prim)

    unassigned = np.flatnonzero(~assigned)
    if len(unassigned):
        add_group(unassigned, mesh_prim)

    for group in groups:
        group["face_indices"] = np.asarray(group["face_indices"], dtype=np.int64)
    return groups


def _write_obj(path: Path, points, faces, uv_values, uv_corners, material_name: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"mtllib {material_name}.mtl\nusemtl {material_name}\n")
        for point in points:
            stream.write("v {:.9g} {:.9g} {:.9g}\n".format(*point))
        if uv_corners is not None:
            for uv in uv_values:
                stream.write("vt {:.9g} {:.9g}\n".format(*uv))
        for face_index, face in enumerate(faces):
            if uv_corners is None:
                stream.write("f " + " ".join(str(int(vertex) + 1) for vertex in face) + "\n")
            else:
                uv_ids = uv_corners[face_index]
                stream.write(
                    "f "
                    + " ".join(f"{int(vertex) + 1}/{int(uv) + 1}" for vertex, uv in zip(face, uv_ids, strict=True))
                    + "\n"
                )


def _write_wrappers(
    destination: Path,
    name: str,
    mass: float,
    bounds: np.ndarray,
    visual_files: list[dict],
    collision_parts: list[tuple[np.ndarray, np.ndarray]] | None = None,
    friction_proxy: float | None = None,
) -> None:
    size = bounds[1] - bounds[0]
    center = bounds.mean(axis=0)
    if np.any(size <= 0) or not np.isfinite(size).all() or not np.isfinite(mass) or mass <= 0:
        raise ValueError(f"Invalid dimensions or mass for {name}")
    center, raw_inertia, final_inertia = _inertial_values(mass, bounds)
    if friction_proxy is not None and (not np.isfinite(friction_proxy) or friction_proxy < 0):
        raise ValueError("Friction proxy must be finite and non-negative")
    friction_attr = f' friction="{friction_proxy:.9g} 0.005 0.0001"' if friction_proxy is not None else ""
    ix, iy, iz = final_inertia
    asset_xml = [f'<mesh name="visual_{i}" file="{item["obj"]}"/>' for i, item in enumerate(visual_files)]
    if collision_parts:
        for index, (vertices, faces) in enumerate(collision_parts):
            mesh_path = destination / f"collision_{index}.obj"
            trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(mesh_path)
            asset_xml.append(f'<mesh name="collision_{index}" file="{mesh_path.name}" inertia="shell"/>')
        # MuJoCo shows groups 0-2 by default. Keep collision-only meshes
        # collidable but hide them from normal RGB observations.
        collision_geoms = [
            f'<geom name="collision_{index}" type="mesh" mesh="collision_{index}" group="3"{friction_attr}/>'
            for index in range(len(collision_parts))
        ]
        urdf_collisions = [
            f'<collision><geometry><mesh filename="collision_{index}.obj"/></geometry></collision>'
            for index in range(len(collision_parts))
        ]
    else:
        collision = trimesh.creation.box(extents=size)
        collision.apply_translation(center)
        collision.export(destination / "collision.obj")
        asset_xml.append('<mesh name="collision" file="collision.obj"/>')
        collision_geoms = [
            f'<geom name="collision" type="mesh" mesh="collision" rgba="0 0 0 0" group="3"{friction_attr}/>'
        ]
        urdf_collisions = ['<collision><geometry><mesh filename="collision.obj"/></geometry></collision>']
    for index, item in enumerate(visual_files):
        rgba = " ".join(f"{value:.8g}" for value in item["color"])
        if item["texture"]:
            asset_xml.append(f'<texture name="texture_{index}" type="2d" file="{item["texture"]}"/>')
            asset_xml.append(f'<material name="material_{index}" texture="texture_{index}" rgba="{rgba} 1"/>')
        else:
            asset_xml.append(f'<material name="material_{index}" rgba="{rgba} 1"/>')
    geoms = [
        f'<geom name="visual_{i}" type="mesh" mesh="visual_{i}" material="material_{i}" contype="0" conaffinity="0" group="1"/>'
        for i in range(len(visual_files))
    ]
    geoms.extend(collision_geoms)
    (destination / "object.xml").write_text(
        f'<mujoco model="{name}"><asset>{"".join(asset_xml)}</asset><worldbody><body name="{name}">'
        f'<inertial pos="{center[0]} {center[1]} {center[2]}" mass="{mass}" '
        f'diaginertia="{ix} {iy} {iz}"/>{"".join(geoms)}</body></worldbody></mujoco>\n',
        encoding="utf-8",
    )
    visual_urdf = []
    for item in visual_files:
        texture = f'<texture filename="{item["texture"]}"/>' if item["texture"] else ""
        urdf_color = (1.0, 1.0, 1.0) if item["texture"] else item["color"]
        visual_urdf.append(
            f'<visual><geometry><mesh filename="{item["obj"]}"/></geometry>'
            f'<material name="material"><color rgba="{urdf_color[0]} {urdf_color[1]} {urdf_color[2]} 1"/>'
            f"{texture}</material></visual>"
        )
    (destination / "object.urdf").write_text(
        f'<?xml version="1.0"?><robot name="{name}"><link name="{name}"><inertial>'
        f'<origin xyz="{center[0]} {center[1]} {center[2]}"/><mass value="{mass}"/>'
        f'<inertia ixx="{ix}" ixy="0" ixz="0" iyy="{iy}" iyz="0" izz="{iz}"/>'
        f"</inertial>{''.join(visual_urdf)}{''.join(urdf_collisions)}"
        f"</link></robot>\n",
        encoding="utf-8",
    )


def prepare_object(
    source: Path,
    output: Path,
    asset_id: str,
    *,
    instance_name: str | None = None,
    texture_size: int = 1024,
    flatten_articulated_static: bool = False,
    source_layer: str = "Aligned.usda",
    physics_source: str = "metadata",
) -> dict:
    """Convert one rigid source asset with per-face diffuse materials and its object frame intact."""
    from pxr import Usd, UsdGeom

    source, output = source.resolve(), output.resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", asset_id):
        raise ValueError(f"Invalid Genie Sim asset ID: {asset_id!r}")
    if not 64 <= texture_size <= 4096:
        raise ValueError("texture_size must be between 64 and 4096")
    if source_layer not in {"Aligned.usda", "Aligned.usd"}:
        raise ValueError(f"Unsupported Genie Sim source layer: {source_layer!r}")
    if physics_source not in {"metadata", "usd"}:
        raise ValueError("physics_source must be 'metadata' or 'usd'")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prepared Genie Sim object: {output}")
    name = instance_name or asset_id
    if not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ValueError(f"Invalid RoboVerse object instance name: {name!r}")
    relative = _find_asset(source, asset_id)
    source_files = {
        name: _verify_asset_file(source, f"{relative}/{name}", asset_id)
        for name in ("Aligned.usda", "Aligned.usd", "object_parameters.json", "item.py")
    }
    if asset_id in HISTORICAL_ASSETS:
        for filename, expected in HISTORICAL_ASSETS[asset_id]["sha256"].items():
            path = _verify_asset_file(source, f"{relative}/{filename}", asset_id)
            if _sha256(path) != expected:
                raise ValueError(f"Historical asset checksum mismatch: {asset_id}/{filename}")
            source_files[filename] = path
    item_info = ast.literal_eval(source_files["item.py"].read_text(encoding="utf-8"))
    source_articulated = bool(item_info.get("has_joint") or item_info.get("has_articulation"))
    if source_articulated and not flatten_articulated_static:
        raise ValueError(f"Articulated source asset {asset_id} needs an articulation-specific converter")
    stage = Usd.Stage.Open(str(source_files[source_layer]))
    if (
        not stage
        or UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z
        or not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0)
    ):
        raise ValueError(f"Expected a Z-up Genie Sim object in meters: {asset_id}")
    mesh_root = stage.GetPrimAtPath("/World/entity" if source_layer == "Aligned.usda" else "/World")
    mesh_prims = [
        prim for prim in Usd.PrimRange(mesh_root)
        if prim.IsA(UsdGeom.Mesh)
        and (source_layer == "Aligned.usda" or "visual" in str(prim.GetPath()).lower().split("/"))
    ]
    if not mesh_prims:
        raise ValueError(f"No visual meshes found for {asset_id}")
    output.parent.mkdir(parents=True, exist_ok=True)
    parameters = json.loads(source_files["object_parameters.json"].read_text(encoding="utf-8"))
    mass = float(parameters.get("mass", 0.0))
    source_physics = None
    if physics_source == "usd":
        if source_articulated:
            raise ValueError("Source USD rigid physics cannot flatten an articulated asset")
        source_physics = _single_usd_rigid_physics(stage, source, asset_id)
        mass = source_physics["rigid_bodies"][0]["authored_mass_kg"]
    with tempfile.TemporaryDirectory(prefix=f"geniesim-{asset_id}-", dir=output.parent) as temporary:
        destination = Path(temporary) / asset_id
        destination.mkdir()
        root = stage.GetPrimAtPath("/World")
        visual_files, bounds_list, collision_parts = [], [], []
        for prim in mesh_prims:
            points, faces, uv_values, uv_corners, source_faces = _triangles(prim)
            transform, reset = UsdGeom.XformCache().ComputeRelativeTransform(prim, root)
            if reset:
                raise ValueError(f"Unexpected reset transform in {prim.GetPath()}")
            mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)
            mesh.apply_transform(np.asarray(transform).T)
            bounds_list.extend(mesh.bounds.tolist())
            if source_articulated and flatten_articulated_static:
                collision_parts.append((np.asarray(mesh.vertices), np.asarray(mesh.faces)))
            material_groups = _material_groups(stage, prim, Path(source_files["Aligned.usda"]).parent)
            for group in material_groups:
                selected = np.isin(source_faces, group["face_indices"])
                if not np.any(selected):
                    continue
                visual_index = len(visual_files)
                color = group["color"]
                texture_path = group["texture_path"]
                texture_name = None
                if texture_path is not None:
                    relative_texture = texture_path.relative_to(source.resolve()).as_posix()
                    source_texture = _verify_asset_file(source, relative_texture, asset_id)
                    texture_name = f"material_{visual_index}.png"
                    with Image.open(source_texture) as image:
                        image = image.convert("RGB")
                        image.thumbnail((texture_size, texture_size), Image.Resampling.LANCZOS)
                        image.save(destination / texture_name)
                obj_name, material_name = f"visual_{visual_index}.obj", f"material_{visual_index}"
                part_uv_corners = uv_corners[selected] if uv_corners is not None else None
                _write_obj(
                    destination / obj_name,
                    mesh.vertices,
                    mesh.faces[selected],
                    uv_values,
                    part_uv_corners,
                    material_name,
                )
                if texture_name:
                    mtl = f"newmtl {material_name}\nKd 1 1 1\nmap_Kd {texture_name}\n"
                else:
                    mtl = f"newmtl {material_name}\nKd {' '.join(map(str, color))}\n"
                (destination / f"{material_name}.mtl").write_text(mtl, encoding="utf-8")
                visual_files.append({"obj": obj_name, "texture": texture_name, "color": color})
        bounds = np.asarray([np.min(bounds_list, axis=0), np.max(bounds_list, axis=0)], dtype=np.float64)
        _, raw_inertia, final_inertia = _inertial_values(mass, bounds)
        _write_wrappers(
            destination,
            name,
            mass,
            bounds,
            visual_files,
            collision_parts if source_articulated and flatten_articulated_static else None,
            friction_proxy=source_physics["single_coefficient_proxy"] if source_physics else None,
        )
        provenance = {
            "format": _USD_PHYSICS_OBJECT_FORMAT if source_physics else _OBJECT_FORMAT,
            "source_repository": "https://huggingface.co/datasets/agibot-world/GenieSimAssets",
            "source_revision": source_revision(asset_id),
            "asset_license": "CC-BY-NC-SA-4.0; retain source terms",
            "asset_id": asset_id,
            "instance_name": name,
            "source_layer": source_layer,
            "source_asset_directory": relative,
            "source_files_sha256": {name: _sha256(path) for name, path in source_files.items()},
            "source_subframes": _source_subframes(stage),
            "mass_kg": mass,
            "bounds_m": bounds.tolist(),
            "inertia_floor_kg_m2": _MIN_OBJECT_INERTIA,
            "inertia_diagonal_kg_m2": final_inertia.tolist(),
            "inertia_floor_adjustments": [
                {"axis": axis, "before": float(before), "after": _MIN_OBJECT_INERTIA}
                for axis, before in zip(("x", "y", "z"), raw_inertia, strict=True)
                if before < _MIN_OBJECT_INERTIA
            ],
            "visual_mesh_count": len(visual_files),
            "collision": (
                "one convex mesh collider per source part at its authored default pose; static approximation"
                if source_articulated and flatten_articulated_static
                else "axis-aligned bounding box of visual bounds; unsuitable for hollow containers or articulated assets"
            ),
            "appearance": f"Per-face diffuse textures are downsampled to at most {texture_size}px; MJCF keeps the USD base color and textured URDF visuals use neutral white; other PBR maps are omitted",
            "articulation_supported": False,
            "urdf_textured_material_tint": "neutral_white",
            "source_articulated": source_articulated,
            "articulation_flattened_as_static_geometry": source_articulated and flatten_articulated_static,
        }
        if source_physics is not None:
            provenance["source_physics"] = source_physics
            provenance["physics_mapping"] = {
                "mass": "authored USD rigid-body mass, before scene-layer overrides",
                "inertia": "box-bounds estimate with documented inertia floor; not source PhysX inertia",
                "mujoco_and_genesis": "MJCF single-coefficient friction proxy; pair combination may differ from PhysX",
                "newton": "URDF mass only; source friction is recorded but not applied by this converter",
            }
        (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        shutil.move(str(destination), output)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Pinned local Genie Sim asset snapshot")
    parser.add_argument("--output", type=Path, required=True, help="Output root under roboverse_data/icl/objects")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--asset-id", action="append", help="Source asset ID; repeat for multiple objects")
    selection.add_argument("--spec", type=Path, help="Imported benchmark.json; prepare all referenced asset IDs")
    parser.add_argument("--instance-name", help="Set the generated URDF/MJCF root name for a single instance")
    parser.add_argument(
        "--source-layer",
        choices=("Aligned.usda", "Aligned.usd"),
        default="Aligned.usda",
        help="USD layer to convert; use Aligned.usd when a scene directly references that raw layer",
    )
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument(
        "--physics-source", choices=("metadata", "usd"), default="metadata",
        help="use metadata mass (legacy) or authored USD rigid mass and a documented friction proxy",
    )
    parser.add_argument(
        "--refresh-subframes", action="store_true",
        help="Verify source hashes and add source subframe transforms to existing object provenance",
    )
    parser.add_argument(
        "--flatten-articulated-static",
        action="store_true",
        help="Flatten articulated assets in the selected spec subset at authored poses as static mesh geometry",
    )
    parser.add_argument("--robot", choices=("g1_op", "g2_op"), help="Filter --spec asset IDs to one robot family")
    args = parser.parse_args()
    if not 64 <= args.texture_size <= 4096:
        parser.error("--texture-size must be between 64 and 4096")
    if args.spec and args.instance_name:
        parser.error("--instance-name can only be used with one --asset-id")
    if args.source_layer != "Aligned.usda" and (args.spec or len(args.asset_id or []) != 1):
        parser.error("--source-layer Aligned.usd requires exactly one --asset-id and an explicit output directory")
    if args.physics_source == "usd" and args.refresh_subframes:
        parser.error("--refresh-subframes only supports the legacy metadata-physics asset version")
    if args.robot and not args.spec:
        parser.error("--robot can only be used with --spec")
    if args.flatten_articulated_static and args.spec and not args.robot:
        parser.error("--flatten-articulated-static with --spec requires an explicit --robot filter")
    if args.spec:
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        if args.robot:
            asset_ids = sorted({
                asset_id
                for run in spec.get("runs", [])
                if run.get("robot") == args.robot
                for instance_key in run.get("instances", [])
                for asset_id in spec.get("instances", {}).get(instance_key, {}).get("asset_ids", [])
            })
        else:
            asset_ids = list(spec.get("asset_ids", []))
        if not asset_ids:
            parser.error("The imported benchmark specification has no matching asset_ids")
    else:
        asset_ids = args.asset_id or []
    if len(set(asset_ids)) != len(asset_ids):
        parser.error("Asset IDs must be unique")
    if args.flatten_articulated_static and len(asset_ids) != 1 and not args.robot:
        parser.error("--flatten-articulated-static requires one --asset-id or a filtered --spec")
    output_root = args.output.resolve()
    is_single_direct_output = not args.spec and len(asset_ids) == 1
    if not is_single_direct_output:
        output_root.mkdir(parents=True, exist_ok=True)
    else:
        output_root.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    for asset_id in asset_ids:
        output = output_root / asset_id if not is_single_direct_output else output_root
        if output.exists():
            existing = output / "provenance.json"
            try:
                if args.refresh_subframes:
                    refresh_source_subframes(args.source.resolve(), output, asset_id)
                provenance = json.loads(existing.read_text(encoding="utf-8"))
                needs_urdf_material_upgrade = provenance.get("format") in {_OBJECT_FORMAT_V1, _OBJECT_FORMAT_V2} or (
                    provenance.get("format") == _OBJECT_FORMAT
                    and provenance.get("urdf_textured_material_tint") != "neutral_white"
                )
                needs_inertia_upgrade = provenance.get("inertia_floor_kg_m2") != _MIN_OBJECT_INERTIA
                if (
                    args.physics_source == "metadata"
                    and args.source_layer == "Aligned.usda"
                    and (needs_urdf_material_upgrade or needs_inertia_upgrade)
                    and provenance.get("source_revision") == REVISION
                    and provenance.get("asset_id") == asset_id
                    and provenance.get("source_layer", "Aligned.usda") == args.source_layer
                ):
                    _upgrade_prepared_object(args.source, output, asset_id)
                    provenance = json.loads(existing.read_text(encoding="utf-8"))
                required = ("object.urdf", "object.xml", "provenance.json", "collision.obj")
                if (
                    provenance.get("format") == (_USD_PHYSICS_OBJECT_FORMAT if args.physics_source == "usd" else _OBJECT_FORMAT)
                    and provenance.get("source_revision") == source_revision(asset_id)
                    and provenance.get("asset_id") == asset_id
                    and provenance.get("source_layer", "Aligned.usda") == args.source_layer
                    and (
                        all((output / name).is_file() for name in required)
                        or bool(list(output.glob("collision_*.obj")))
                    )
                    and (
                        not args.flatten_articulated_static
                        or not provenance.get("source_articulated", False)
                        or provenance.get("articulation_flattened_as_static_geometry") is True
                    )
                    and provenance.get("urdf_textured_material_tint") == "neutral_white"
                ):
                    print(f"Already prepared {asset_id} -> {output}")
                    continue
            except (OSError, json.JSONDecodeError):
                pass
            failures.append((asset_id, f"output already exists and is not a verified prepared object: {output}"))
            continue
        try:
            result = prepare_object(
                args.source,
                output,
                asset_id,
                instance_name=args.instance_name if len(asset_ids) == 1 else None,
                texture_size=args.texture_size,
                flatten_articulated_static=args.flatten_articulated_static,
                source_layer=args.source_layer,
                physics_source=args.physics_source,
            )
            print(f"Prepared {result['asset_id']} ({result['visual_mesh_count']} meshes) -> {output}")
        except Exception as exc:
            failures.append((asset_id, f"{type(exc).__name__}: {exc}"))
            print(f"FAILED {asset_id}: {failures[-1][1]}")
    summary = output_root / "prepare_failures.json"
    if failures:
        summary.write_text(
            json.dumps([{"asset_id": asset, "error": error} for asset, error in failures], indent=2) + "\n",
            encoding="utf-8",
        )
        raise SystemExit(f"Prepared with {len(failures)} failures; see {summary}")
    if summary.is_file():
        summary.unlink()


if __name__ == "__main__":
    main()
