"""Convert pinned RoboTwin base3 GLBs into textured URDF/MJCF task assets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

import numpy as np
import trimesh

from roboverse_pack.tasks.imitator_game.robotwin_assets import (
    ROBOTWIN_COMMIT_DATE,
    ROBOTWIN_DATASET_LICENSE_SOURCE,
    ROBOTWIN_LICENSE,
    ROBOTWIN_LICENSE_SOURCE,
    ROBOTWIN_PROFILES,
    ROBOTWIN_REPOSITORY,
    ROBOTWIN_REVISION,
    RobotwinAssetProfile,
    sha256_file,
    validate_robotwin_source,
)


def _load_scene(path: Path, *, scale: tuple[float, float, float]) -> list[trimesh.Trimesh]:
    scene = trimesh.load(path, force="scene", process=False)
    if not isinstance(scene, trimesh.Scene):
        raise ValueError(f"Expected GLB scene at {path}")
    pieces = scene.dump(concatenate=False)
    if not pieces:
        raise ValueError(f"GLB contains no meshes: {path}")
    scaled = []
    scale_vector = np.asarray(scale, dtype=np.float64)
    for index, mesh in enumerate(pieces):
        if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
            raise ValueError(f"GLB mesh {index} is empty or not triangular: {path}")
        if not np.isfinite(mesh.vertices).all():
            raise ValueError(f"GLB mesh {index} contains non-finite vertices: {path}")
        mesh = mesh.copy()
        mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * scale_vector
        scaled.append(mesh)
    return scaled


def _write_obj(mesh: trimesh.Trimesh, path: Path, *, textured: bool) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    exported = trimesh.exchange.obj.export_obj(
        mesh,
        include_normals=True,
        include_color=not textured,
        include_texture=textured,
        return_texture=textured,
        digits=12,
        mtl_name=path.with_suffix(".mtl").name,
    )
    resources: list[Path] = [path]
    if textured:
        obj_text, sidecars = exported
        path.write_text(obj_text, encoding="utf-8")
        for name, content in sidecars.items():
            safe_name = Path(name).name
            output_path = path.parent / safe_name
            output_path.write_bytes(content if isinstance(content, bytes) else str(content).encode("utf-8"))
            resources.append(output_path)
    else:
        path.write_text(exported, encoding="utf-8")
    return resources


def _texture_from_mtl(mesh_dir: Path, mtl_files: list[Path]) -> Path | None:
    for mtl_path in mtl_files:
        if mtl_path.suffix.lower() != ".mtl":
            continue
        for line in mtl_path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.strip().split(maxsplit=1)
            if len(fields) == 2 and fields[0] == "map_Kd":
                image = (mesh_dir / fields[1].strip()).resolve()
                if image.is_file() and mesh_dir.resolve() in image.parents:
                    return image
    return None


def _fmt(values) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _bounds(meshes: list[trimesh.Trimesh]) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.concatenate([mesh.vertices for mesh in meshes], axis=0)
    if not np.isfinite(vertices).all():
        raise ValueError("Converted asset has non-finite vertices")
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    if np.any(lower >= upper):
        raise ValueError("Converted asset has degenerate bounds")
    return lower, upper


def _box_inertia(mass_kg: float, lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    extents = upper - lower
    center = (upper + lower) / 2
    ixx, iyy, izz = (mass_kg / 12 * (extents[(axis + 1) % 3] ** 2 + extents[(axis + 2) % 3] ** 2) for axis in range(3))
    return center, np.asarray((ixx, iyy, izz), dtype=np.float64)


def _write_wrappers(
    asset_dir: Path,
    visual_meshes: list[trimesh.Trimesh],
    collision_meshes: list[trimesh.Trimesh],
    profile: RobotwinAssetProfile,
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = _bounds(collision_meshes)
    visual_files: list[tuple[str, Path, Path | None]] = []

    for index, mesh in enumerate(visual_meshes):
        mesh_dir = asset_dir / "visual" / f"mesh_{index:03d}"
        obj_path = mesh_dir / "visual.obj"
        sidecars = _write_obj(mesh, obj_path, textured=True)
        texture = _texture_from_mtl(mesh_dir, sidecars)
        visual_files.append((f"visual_{index:03d}", obj_path.relative_to(asset_dir), texture))

    collision_files: list[tuple[str, Path]] = []
    for index, mesh in enumerate(collision_meshes):
        if not mesh.is_convex:
            raise ValueError(f"Source collision component {index} is not convex; refusing to substitute a hull")
        obj_path = asset_dir / "collision" / f"component_{index:03d}.obj"
        _write_obj(mesh, obj_path, textured=False)
        collision_files.append((f"collision_{index:03d}", obj_path.relative_to(asset_dir)))

    mass_kg = profile.mass_kg if profile.mass_kg is not None else 1.0
    inertial_center, inertia = _box_inertia(mass_kg, lower, upper)
    urdf = ET.Element("robot", {"name": profile.source_object})
    link = ET.SubElement(urdf, "link", {"name": "object"})
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": _fmt(inertial_center), "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": f"{mass_kg:.12g}"})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": f"{inertia[0]:.12g}",
            "iyy": f"{inertia[1]:.12g}",
            "izz": f"{inertia[2]:.12g}",
            "ixy": "0",
            "ixz": "0",
            "iyz": "0",
        },
    )
    for _, relative_path, _ in visual_files:
        visual = ET.SubElement(link, "visual")
        geometry = ET.SubElement(visual, "geometry")
        ET.SubElement(geometry, "mesh", {"filename": relative_path.as_posix()})
    for _, relative_path in collision_files:
        collision = ET.SubElement(link, "collision")
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "mesh", {"filename": relative_path.as_posix()})
    urdf_path = asset_dir / "object.urdf"
    ET.indent(urdf, space="  ")
    ET.ElementTree(urdf).write(urdf_path, encoding="utf-8", xml_declaration=True)

    mjcf = ET.Element("mujoco", {"model": profile.source_object})
    ET.SubElement(mjcf, "compiler", {"meshdir": ".", "autolimits": "true"})
    assets = ET.SubElement(mjcf, "asset")
    for name, relative_path, texture in visual_files:
        ET.SubElement(assets, "mesh", {"name": name, "file": relative_path.as_posix()})
        if texture is not None:
            texture_name = f"{name}_texture"
            material_name = f"{name}_material"
            ET.SubElement(
                assets,
                "texture",
                {"name": texture_name, "type": "2d", "file": texture.relative_to(asset_dir).as_posix()},
            )
            ET.SubElement(assets, "material", {"name": material_name, "texture": texture_name})
    for name, relative_path in collision_files:
        ET.SubElement(assets, "mesh", {"name": name, "file": relative_path.as_posix()})
    worldbody = ET.SubElement(mjcf, "worldbody")
    body = ET.SubElement(worldbody, "body", {"name": "object"})
    ET.SubElement(
        body,
        "inertial",
        {
            "pos": _fmt(inertial_center),
            "mass": f"{mass_kg:.12g}",
            "diaginertia": _fmt(inertia),
        },
    )
    for name, _, texture in visual_files:
        attributes = {"type": "mesh", "mesh": name, "group": "1", "contype": "0", "conaffinity": "0"}
        if texture is not None:
            attributes["material"] = f"{name}_material"
        else:
            attributes["rgba"] = "0.75 0.75 0.75 1"
        ET.SubElement(body, "geom", attributes)
    for name, _ in collision_files:
        ET.SubElement(body, "geom", {"type": "mesh", "mesh": name, "group": "0"})
    mjcf_path = asset_dir / "object.xml"
    ET.indent(mjcf, space="  ")
    ET.ElementTree(mjcf).write(mjcf_path, encoding="utf-8", xml_declaration=True)
    return lower, upper


def prepare_robotwin_asset(
    *,
    profile: RobotwinAssetProfile,
    source_directory: Path,
    output_directory: Path,
) -> Path:
    """Create verified, scale-baked simulator wrappers without modifying downloaded files."""
    source_directory, output_directory = Path(source_directory), Path(output_directory)
    if output_directory.exists():
        raise FileExistsError(f"Refusing to replace prepared Robotwin asset: {output_directory}")
    source_provenance, _ = validate_robotwin_source(source_directory, profile)
    output_directory.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix=f"robotwin-{profile.key}-", dir=output_directory.parent) as temporary:
        stage = Path(temporary) / "asset"
        stage.mkdir()
        source_map = {
            "visual/base3.glb": "source/visual/base3.glb",
            "collision/base3.glb": "source/collision/base3.glb",
            "model_data3.json": "model_data3.json",
        }
        for relative_path, output_relative in source_map.items():
            destination = stage / output_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_directory / relative_path, destination)
        (stage / "source_provenance.json").write_text(json.dumps(source_provenance, indent=2) + "\n", encoding="utf-8")

        visual_meshes = _load_scene(source_directory / "visual" / "base3.glb", scale=profile.baked_scale)
        collision_meshes = _load_scene(source_directory / "collision" / "base3.glb", scale=profile.baked_scale)
        if len(collision_meshes) < 2:
            raise ValueError("Pinned RoboTwin collision GLB must retain its convex decomposition components")
        lower, upper = _write_wrappers(stage, visual_meshes, collision_meshes, profile)
        source_files = {path: digest for path, (digest, _) in profile.source_files.items()}
        generated_files = {}
        for file_path in sorted(stage.rglob("*")):
            if file_path.is_file() and file_path.name not in {"provenance.json", "source_provenance.json"}:
                generated_files[file_path.relative_to(stage).as_posix()] = sha256_file(file_path)
        provenance = {
            "asset_schema": "roboverse.imitator_game.robotwin_prepared.v1",
            "source_repository": ROBOTWIN_REPOSITORY,
            "source_revision": ROBOTWIN_REVISION,
            "source_commit_date": ROBOTWIN_COMMIT_DATE,
            "source_object": profile.source_object,
            "model_id": profile.model_id,
            "upstream_variant": "base3",
            "asset_license": ROBOTWIN_LICENSE,
            "license_source": ROBOTWIN_LICENSE_SOURCE,
            "dataset_license_source": ROBOTWIN_DATASET_LICENSE_SOURCE,
            "official_object_documentation": source_provenance.get("official_object_documentation"),
            "metadata_scale": list(profile.metadata_scale),
            "baked_scale": list(profile.baked_scale),
            "task_scale_semantics": profile.task_scale_semantics,
            "source_files": source_map,
            "source_files_sha256": source_files,
            "generated_files_sha256": generated_files,
            "mass_kg": profile.mass_kg if profile.mass_kg is not None else 1.0,
            "mass_semantics": (
                "Task-defined 0.5 kg dynamic tissue box."
                if profile.mass_kg is not None
                else "Static basket; wrapper mass is non-operative because the task fixes the body."
            ),
            "inertia": "Diagonal box approximation from scaled collision bounds; not source inertial parity.",
            "collision": (
                "Each source convex-decomposition scene component remains a separate collision mesh; no hull replacement."
            ),
            "visual_materials": (
                "Base color UV textures are exported to OBJ/MTL and referenced from MJCF; glTF metallic/roughness "
                "channels are not represented by the URDF/MJCF conversion."
            ),
            "local_axes_and_origin": "Source GLB local axes, origin and scene-node transforms are preserved.",
            "bounds_m": [lower.tolist(), upper.tolist()],
            "visual_component_count": len(visual_meshes),
            "collision_component_count": len(collision_meshes),
        }
        (stage / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        output_directory.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(output_directory)
    return output_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(ROBOTWIN_PROFILES), required=True)
    parser.add_argument(
        "--source-directory",
        type=Path,
        required=True,
        help="Downloaded object folder containing model_data3.json and visual/collision base3.glb files",
    )
    parser.add_argument("--output-directory", type=Path, required=True, help="New isolated prepared wrapper folder")
    args = parser.parse_args()
    result = prepare_robotwin_asset(
        profile=ROBOTWIN_PROFILES[args.profile],
        source_directory=args.source_directory,
        output_directory=args.output_directory,
    )
    print(result)


if __name__ == "__main__":
    main()
