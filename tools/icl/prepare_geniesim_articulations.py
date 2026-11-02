"""Convert supported articulated Genie Sim USD objects into portable MetaSim assets."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.icl.prepare_geniesim_g1 import REVISION, _verify
from tools.icl.prepare_geniesim_objects import _find_asset, _material_groups, _triangles, _write_obj

_FORMAT = "roboverse.icl.geniesim_articulated_object.v1"
_SUPPORTED_MOTION = {"PhysicsFixedJoint", "PhysicsRevoluteJoint"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vec(value, size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"Invalid {label}: expected {size} finite values")
    return result


def _quat_xyzw(value) -> np.ndarray:
    return _vec([*value.GetImaginary(), value.GetReal()], 4, "USD quaternion") / np.linalg.norm(
        _vec([*value.GetImaginary(), value.GetReal()], 4, "USD quaternion")
    )


def _rotation_text(quat_xyzw: np.ndarray, *, order: str) -> str:
    if order == "urdf":
        values = Rotation.from_quat(quat_xyzw).as_euler("xyz")
    else:
        values = Rotation.from_quat(quat_xyzw).as_quat(scalar_first=True)
    return " ".join(f"{value:.10g}" for value in values)


def _fmt(values) -> str:
    return " ".join(f"{float(value):.10g}" for value in values)


def _link_mass(asset_id: str, name: str) -> float:
    """Use explicit, conservative engineering masses because source USD omits link masses."""
    defaults = {
        "benchmark_door_000": {"body": 1.0, "part_00": 5.0, "part_03": 0.1, "part_07": 0.05},
        "benchmark_laptop_000": {"body": 0.5, "part_00": 0.3, "part_01": 0.1},
    }
    return defaults.get(asset_id, {}).get(name, 0.1)


def _inertial(mass: float, bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = bounds.mean(axis=0)
    size = np.maximum(bounds[1] - bounds[0], 0.01)
    inertia = np.asarray(
        [
            mass * (size[1] ** 2 + size[2] ** 2) / 12,
            mass * (size[0] ** 2 + size[2] ** 2) / 12,
            mass * (size[0] ** 2 + size[1] ** 2) / 12,
        ],
        dtype=np.float64,
    )
    inertia = np.maximum(inertia, 1.1e-6)
    return center, inertia


def _source_model(source_root: Path, asset_id: str):
    from pxr import Usd, UsdGeom, UsdPhysics

    relative = _find_asset(source_root, asset_id)
    source_dir = source_root / relative
    model_file = next(
        (source_dir / name for name in ("Aligned.usda", "Aligned.usd") if (source_dir / name).is_file()), None
    )
    if model_file is None:
        raise FileNotFoundError(f"No aligned USD model for {asset_id}: {source_dir}")
    item_file = _verify(source_root, f"{relative}/item.py")
    parameters_file = _verify(source_root, f"{relative}/object_parameters.json")
    item = ast.literal_eval(item_file.read_text(encoding="utf-8"))
    if not item.get("has_joint") and not item.get("has_articulation"):
        raise ValueError(f"Source object {asset_id} is not articulated")
    stage = Usd.Stage.Open(str(model_file))
    root = stage.GetPrimAtPath("/World/entity") if stage else None
    if not root:
        raise ValueError(f"Missing /World/entity in {model_file}")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z or not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0):
        raise ValueError(f"Expected Z-up meters in {model_file}")
    bodies = {str(prim.GetPath()): prim for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
    joints = [prim for prim in Usd.PrimRange(root) if prim.IsA(UsdPhysics.Joint)]
    if not bodies or not joints:
        raise ValueError(f"No rigid links or joints found in {asset_id}")
    incoming: set[str] = set()
    edges = []
    for prim in joints:
        if prim.GetTypeName() not in _SUPPORTED_MOTION:
            raise ValueError(f"Unsupported articulated joint type {prim.GetTypeName()} in {prim.GetPath()}")
        joint = UsdPhysics.Joint(prim)
        parent_targets = [str(path) for path in joint.GetBody0Rel().GetTargets()]
        child_targets = [str(path) for path in joint.GetBody1Rel().GetTargets()]
        if len(parent_targets) > 1 or len(child_targets) != 1 or child_targets[0] not in bodies:
            raise ValueError(f"Unsupported joint body relationship in {prim.GetPath()}")
        parent = parent_targets[0] if parent_targets else None
        child = child_targets[0]
        if parent is None:
            continue  # Source's fixed world anchor for the base link.
        if parent not in bodies or child in incoming:
            raise ValueError(f"Articulation is not a tree at {prim.GetPath()}")
        incoming.add(child)
        pos0 = _vec(joint.GetLocalPos0Attr().Get(), 3, f"{prim.GetPath()} localPos0")
        quat0 = _quat_xyzw(joint.GetLocalRot0Attr().Get())
        pos1 = _vec(joint.GetLocalPos1Attr().Get(), 3, f"{prim.GetPath()} localPos1")
        quat1 = _quat_xyzw(joint.GetLocalRot1Attr().Get())
        if np.linalg.norm(pos1) > 1e-6 or not np.allclose(quat1, [0, 0, 0, 1], atol=1e-6):
            raise ValueError(f"Non-identity child joint frame is not yet supported: {prim.GetPath()}")
        edge = {"prim": prim, "parent": parent, "child": child, "pos": pos0, "quat": quat0}
        if prim.IsA(UsdPhysics.RevoluteJoint):
            revolute = UsdPhysics.RevoluteJoint(prim)
            axis = str(revolute.GetAxisAttr().Get()).upper()
            edge["type"] = "revolute"
            edge["axis"] = {"X": [1.0, 0.0, 0.0], "Y": [0.0, 1.0, 0.0], "Z": [0.0, 0.0, 1.0]}[axis]
            edge["limits"] = [
                math.radians(float(revolute.GetLowerLimitAttr().Get())),
                math.radians(float(revolute.GetUpperLimitAttr().Get())),
            ]
        else:
            edge["type"] = "fixed"
        edges.append(edge)
    roots = set(bodies) - incoming
    if len(roots) != 1:
        raise ValueError(f"Expected one articulated root link in {asset_id}, found {sorted(roots)}")
    root_path = next(iter(roots))
    children: dict[str, list[dict]] = {path: [] for path in bodies}
    for edge in edges:
        children[edge["parent"]].append(edge)
    for group in children.values():
        group.sort(key=lambda edge: str(edge["prim"].GetPath()))
    return stage, root, bodies, children, edges, root_path, source_dir, model_file, item_file, parameters_file


def prepare_articulated_object(
    source_root: Path,
    output: Path,
    asset_id: str,
    *,
    texture_size: int = 1024,
) -> dict:
    """Export source body meshes and fixed/revolute joints to URDF and MuJoCo MJCF."""
    from pxr import Usd, UsdGeom, UsdPhysics

    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prepared articulation: {output}")
    if not 64 <= texture_size <= 4096:
        raise ValueError("texture_size must be between 64 and 4096")
    stage, entity, bodies, children, edges, root_path, source_dir, model_file, item_file, parameters_file = (
        _source_model(source_root, asset_id)
    )
    root_body = bodies[root_path]
    cache = UsdGeom.XformCache()
    root_world = np.asarray(cache.GetLocalToWorldTransform(root_body), dtype=np.float64)
    root_inverse = np.linalg.inv(root_world)
    link_paths = {path: prim for path, prim in bodies.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    material_records = []
    exported_meshes: dict[str, dict[str, list]] = {}
    bounds_by_link: dict[str, list[np.ndarray]] = {path: [] for path in bodies}

    with tempfile.TemporaryDirectory(prefix=f"geniesim-articulation-{asset_id}-", dir=output.parent) as temporary:
        destination = Path(temporary) / asset_id
        destination.mkdir()
        mesh_counter = 0
        for link_path, body in bodies.items():
            link_name = body.GetName()
            exported_meshes[link_path] = {"visuals": [], "collisions": []}
            for prim in Usd.PrimRange(body):
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                points, faces, uv_values, uv_corners, source_faces = _triangles(prim)
                transform, reset = UsdGeom.XformCache().ComputeRelativeTransform(prim, body)
                if reset:
                    raise ValueError(f"Unexpected reset transform in {prim.GetPath()}")
                mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)
                mesh.apply_transform(np.asarray(transform).T)
                if not np.isfinite(mesh.vertices).all() or mesh.is_empty:
                    raise ValueError(f"Invalid mesh geometry in {prim.GetPath()}")
                bounds_by_link[link_path].extend(mesh.bounds)
                has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
                if has_collision:
                    collision = mesh.convex_hull
                    collision_name = f"collision_{mesh_counter}.obj"
                    collision.export(destination / collision_name)
                    exported_meshes[link_path]["collisions"].append(collision_name)
                if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    for group in _material_groups(stage, prim, source_dir):
                        selected = np.isin(source_faces, group["face_indices"])
                        if not np.any(selected):
                            continue
                        index = len(exported_meshes[link_path]["visuals"])
                        object_name, material_name = f"visual_{link_name}_{index}.obj", f"material_{link_name}_{index}"
                        _write_obj(
                            destination / object_name,
                            mesh.vertices,
                            mesh.faces[selected],
                            uv_values,
                            uv_corners[selected] if uv_corners is not None else None,
                            material_name,
                        )
                        texture_name = None
                        texture_path = group["texture_path"]
                        if texture_path is not None:
                            texture_name = f"texture_{link_name}_{index}.png"
                            with Image.open(
                                _verify(source_root, texture_path.relative_to(source_root).as_posix())
                            ) as image:
                                image = image.convert("RGB")
                                image.thumbnail((texture_size, texture_size), Image.Resampling.LANCZOS)
                                image.save(destination / texture_name)
                            mtl = f"newmtl {material_name}\nKd 1 1 1\nmap_Kd {texture_name}\n"
                        else:
                            color = group["color"]
                            mtl = f"newmtl {material_name}\nKd {' '.join(map(str, color))}\n"
                        (destination / f"{material_name}.mtl").write_text(mtl, encoding="utf-8")
                        record = {
                            "obj": object_name,
                            "texture": texture_name,
                            "color": list(group["color"]),
                            "material": material_name,
                        }
                        exported_meshes[link_path]["visuals"].append(record)
                        material_records.append(record)
                mesh_counter += 1
            if not bounds_by_link[link_path]:
                raise ValueError(f"Source rigid body has no mesh geometry: {link_path}")

        # Assemble source-space link frames and all per-link visual/collision geoms.
        xml_assets: list[str] = []
        urdf_links: list[str] = []
        body_xml: dict[str, str] = {}
        for link_path, body in bodies.items():
            link_name = body.GetName()
            bounds = np.asarray([np.min(bounds_by_link[link_path], axis=0), np.max(bounds_by_link[link_path], axis=0)])
            mass = _link_mass(asset_id, link_name)
            center, inertia = _inertial(mass, bounds)
            xml_geoms = []
            urdf_visuals = []
            urdf_collisions = []
            for index, filename in enumerate(exported_meshes[link_path]["collisions"]):
                mesh_name = f"collision_{link_name}_{index}"
                xml_assets.append(f'<mesh name="{mesh_name}" file="{filename}"/>')
                xml_geoms.append(f'<geom name="{mesh_name}" type="mesh" mesh="{mesh_name}"/>')
                urdf_collisions.append(f'<collision><geometry><mesh filename="{filename}"/></geometry></collision>')
            for index, visual in enumerate(exported_meshes[link_path]["visuals"]):
                mesh_name = f"visual_{link_name}_{index}"
                material_name = f"mat_{link_name}_{index}"
                xml_assets.append(f'<mesh name="{mesh_name}" file="{visual["obj"]}"/>')
                color = " ".join(f"{value:.8g}" for value in visual["color"])
                if visual["texture"]:
                    texture_name = f"tex_{link_name}_{index}"
                    xml_assets.append(f'<texture name="{texture_name}" type="2d" file="{visual["texture"]}"/>')
                    xml_assets.append(f'<material name="{material_name}" texture="{texture_name}" rgba="{color} 1"/>')
                else:
                    xml_assets.append(f'<material name="{material_name}" rgba="{color} 1"/>')
                xml_geoms.append(
                    f'<geom name="{mesh_name}" type="mesh" mesh="{mesh_name}" material="{material_name}" '
                    'contype="0" conaffinity="0" group="1"/>'
                )
                urdf_color = "1 1 1 1" if visual["texture"] else f"{color} 1"
                texture = f'<texture filename="{visual["texture"]}"/>' if visual["texture"] else ""
                urdf_visuals.append(
                    f'<visual><geometry><mesh filename="{visual["obj"]}"/></geometry>'
                    f'<material name="{material_name}"><color rgba="{urdf_color}"/>{texture}</material></visual>'
                )
            body_xml[link_path] = (
                f'<inertial pos="{_fmt(center)}" mass="{mass:.9g}" diaginertia="{_fmt(inertia)}"/>{"".join(xml_geoms)}'
            )
            inertia_urdf = (
                f'<inertial><origin xyz="{_fmt(center)}"/><mass value="{mass:.9g}"/>'
                f'<inertia ixx="{inertia[0]:.9g}" ixy="0" ixz="0" iyy="{inertia[1]:.9g}" iyz="0" izz="{inertia[2]:.9g}"/></inertial>'
            )
            urdf_links.append(
                f'<link name="{link_name}">{inertia_urdf}{"".join(urdf_visuals)}{"".join(urdf_collisions)}</link>'
            )

        joint_records = []

        def mjcf_body(path: str) -> str:
            body = link_paths[path]
            name = body.GetName()
            child_xml = []
            for edge in children[path]:
                prim, child_path = edge["prim"], edge["child"]
                child_name = link_paths[child_path].GetName()
                pos, quat = _fmt(edge["pos"]), _fmt(edge["quat"][[3, 0, 1, 2]])
                if edge["type"] == "revolute":
                    lo, hi = edge["limits"]
                    joint_name = prim.GetName()
                    damping = 0.05
                    friction = 0.01
                    joint_records.append({
                        "name": joint_name,
                        "source_path": str(prim.GetPath()),
                        "limits_rad": [lo, hi],
                    })
                    child_xml.append(
                        f'<body name="{child_name}" pos="{pos}" quat="{quat}">'
                        f'<joint name="{joint_name}" type="hinge" axis="{_fmt(edge["axis"])}" '
                        f'range="{lo:.10g} {hi:.10g}" limited="true" damping="{damping}" frictionloss="{friction}"/>'
                        f"{body_xml[child_path]}{''.join(mjcf_body(child_path))}</body>"
                    )
                else:
                    child_xml.append(
                        f'<body name="{child_name}" pos="{pos}" quat="{quat}">'
                        f"{body_xml[child_path]}{''.join(mjcf_body(child_path))}</body>"
                    )
            return child_xml

        root_name = link_paths[root_path].GetName()
        root_xml = f'<body name="{root_name}">{body_xml[root_path]}{"".join(mjcf_body(root_path))}</body>'
        (destination / "object.xml").write_text(
            f'<mujoco model="{asset_id}"><compiler angle="radian" meshdir="."/><option gravity="0 0 -9.81"/>'
            f"<asset>{''.join(xml_assets)}</asset><worldbody>{root_xml}</worldbody></mujoco>\n",
            encoding="utf-8",
        )

        # Write URDF edges in source traversal order so state ordering is reproducible.
        urdf_joints = []
        source_joint_names = []
        for edge in edges:
            prim = edge["prim"]
            parent_name, child_name = link_paths[edge["parent"]].GetName(), link_paths[edge["child"]].GetName()
            joint_name = prim.GetName()
            xyz = _fmt(edge["pos"])
            rpy = _rotation_text(edge["quat"], order="urdf")
            if edge["type"] == "revolute":
                lo, hi = edge["limits"]
                source_joint_names.append(joint_name)
                urdf_joints.append(
                    f'<joint name="{joint_name}" type="revolute"><parent link="{parent_name}"/><child link="{child_name}"/>'
                    f'<origin xyz="{xyz}" rpy="{rpy}"/><axis xyz="{_fmt(edge["axis"])}"/>'
                    f'<limit lower="{lo:.10g}" upper="{hi:.10g}" effort="100" velocity="3"/>'
                    '<dynamics damping="0.05" friction="0.01"/></joint>'
                )
            else:
                urdf_joints.append(
                    f'<joint name="{joint_name}" type="fixed"><parent link="{parent_name}"/><child link="{child_name}"/>'
                    f'<origin xyz="{xyz}" rpy="{rpy}"/></joint>'
                )
        (destination / "object.urdf").write_text(
            f'<?xml version="1.0"?><robot name="{asset_id}">{"".join(urdf_links)}{"".join(urdf_joints)}</robot>\n',
            encoding="utf-8",
        )
        asset_parameters = json.loads(parameters_file.read_text(encoding="utf-8"))
        provenance = {
            "format": _FORMAT,
            "source_repository": "https://huggingface.co/datasets/agibot-world/GenieSimAssets",
            "source_revision": REVISION,
            "asset_license": "CC-BY-NC-SA-4.0; retain source terms",
            "asset_id": asset_id,
            "source_asset_directory": _find_asset(source_root, asset_id),
            "source_files_sha256": {
                "Aligned.usd": _sha256(model_file),
                "item.py": _sha256(item_file),
                "object_parameters.json": _sha256(parameters_file),
            },
            "root_link": link_paths[root_path].GetName(),
            "root_link_transform_in_entity": root_world.T.tolist(),
            "joint_order": source_joint_names,
            "joints": joint_records,
            "link_masses_kg": {
                link_paths[path].GetName(): _link_mass(asset_id, link_paths[path].GetName()) for path in bodies
            },
            "inertia_floor_kg_m2": 1.1e-6,
            "source_mass_kg": asset_parameters.get("mass"),
            "mass_note": "Source USDA has no per-link mass or inertia; explicit link masses and box inertias are conversion approximations.",
            "joint_dynamics_note": "Source revolute joints retain their USD limits; damping/friction are small stability defaults because the source has no panel-joint damping.",
            "visual_mesh_count": sum(len(value["visuals"]) for value in exported_meshes.values()),
            "collision_mesh_count": sum(len(value["collisions"]) for value in exported_meshes.values()),
            "root_is_fixed": asset_id == "benchmark_door_000",
        }
        (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        shutil.move(str(destination), output)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Pinned local GenieSimAssets snapshot")
    parser.add_argument(
        "--output", type=Path, required=True, help="Output root under roboverse_data/icl/objects/geniesim_articulated"
    )
    parser.add_argument(
        "--asset-id", action="append", required=True, help="Articulated object asset ID; repeat as needed"
    )
    parser.add_argument("--texture-size", type=int, default=1024)
    args = parser.parse_args()
    if len(set(args.asset_id)) != len(args.asset_id):
        parser.error("Asset IDs must be unique")
    args.output.mkdir(parents=True, exist_ok=True)
    failures = []
    for asset_id in args.asset_id:
        output = args.output / asset_id
        if output.exists():
            existing = output / "provenance.json"
            try:
                data = json.loads(existing.read_text(encoding="utf-8"))
                if (
                    data.get("format") == _FORMAT
                    and data.get("source_revision") == REVISION
                    and data.get("asset_id") == asset_id
                ):
                    print(f"Already prepared {asset_id} -> {output}")
                    continue
            except (OSError, json.JSONDecodeError):
                pass
            failures.append((asset_id, f"Refusing to overwrite unverified output: {output}"))
            continue
        try:
            result = prepare_articulated_object(args.source, output, asset_id, texture_size=args.texture_size)
            print(f"Prepared {asset_id}: {result['joint_order']} -> {output}")
        except Exception as exc:
            failures.append((asset_id, f"{type(exc).__name__}: {exc}"))
            print(f"FAILED {asset_id}: {failures[-1][1]}")
    if failures:
        raise SystemExit(
            "Articulation preparation failed: " + "; ".join(f"{asset}: {error}" for asset, error in failures)
        )


if __name__ == "__main__":
    main()
