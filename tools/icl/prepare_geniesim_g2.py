"""Prepare the pinned AgiBot G2 Omnipicker model for all three ICL backends."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.icl.prepare_geniesim_g1 import REVISION as DRIVE_SOURCE_REVISION
from tools.icl.prepare_geniesim_g1 import _verify

SOURCE_REPOSITORY = "https://github.com/AgibotTech/genie_sim_robot_model"
SOURCE_REVISION = "6d5b2159806ae0e64b432b638e9ee288d8bddc8f"
DRIVE_SOURCE_REPOSITORY = "https://huggingface.co/datasets/agibot-world/GenieSimAssets"
_LEGACY_INCORRECT_DRIVE_REVISION = "e97579127f5ca5e493aad8adb51dc6b7d9ce5c7a"
DRIVE_SOURCE_LICENSE = "CC-BY-NC-SA-4.0"
PACKAGE_NAME = "genie_sim_robot_model"
MESH_PACKAGE_URI = "package://genie_sim_robot_model/robots/genie/g2/meshes/"
VARIANT = "G2_crs_omnipicker"
XACRO_VARIANT = "g2_crs_omnipicker"
CONTROL_JOINTS = {
    *(f"idx{20 + index:02d}_arm_l_joint{index}" for index in range(1, 8)),
    *(f"idx{60 + index:02d}_arm_r_joint{index}" for index in range(1, 8)),
    *(f"idx{index:02d}_body_joint{index}" for index in range(1, 6)),
    "idx11_head_joint1",
    "idx12_head_joint2",
    "idx13_head_joint3",
    "idx41_gripper_l_outer_joint1",
    "idx81_gripper_r_outer_joint1",
}
DEFAULT_DRIVE_USD = (
    Path(__file__).resolve().parents[2] / "datasets/icl/sources/geniesim-assets/robot/G2_omnipicker/robot.usda"
)
BENCHMARK_ROBOT_USD = "robot/G2_omnipicker/robot_fix.usda"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_revision(source: Path) -> str:
    top_level = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    if Path(top_level).resolve() != source.resolve():
        raise ValueError(f"Expected an independent robot-model Git checkout at {source}")
    status = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    if status.strip():
        raise ValueError(f"Robot-model checkout has uncommitted changes: {source}")
    result = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _verified_drive_usd(drive_usd: Path) -> tuple[Path, Path]:
    """Verify the drive USD and license against the pinned asset manifest."""
    drive_usd = Path(drive_usd).resolve()
    source_root = drive_usd.parents[2]
    verified = _verify(source_root, "robot/G2_omnipicker/robot.usda")
    if verified != drive_usd:
        raise ValueError(f"Expected the canonical G2 drive USD under {source_root}")
    return source_root, _verify(source_root, "LICENSE")


def _source_drive_parameters(drive_usd: Path, *, benchmark: bool = False) -> dict[str, dict[str, float | None]]:
    """Read source joint drives, optionally from the benchmark's fixed-base USD."""
    try:
        from pxr import Usd, UsdPhysics
    except ImportError as exc:
        raise RuntimeError("Reading Genie Sim G2 joint drives requires usd-core (pxr)") from exc
    if benchmark:
        source_root, _ = _verified_drive_usd(drive_usd)
        drive_usd = _verify(source_root, BENCHMARK_ROBOT_USD)
    stage = Usd.Stage.Open(str(drive_usd))
    if stage is None:
        raise ValueError(f"Cannot open pinned Genie Sim G2 robot USD: {drive_usd}")
    result = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        name = prim.GetName()
        if prim.IsA(UsdPhysics.RevoluteJoint):
            prefix, unit_scale = "angular", 180.0 / math.pi
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            prefix, unit_scale = "linear", 1.0
        else:
            continue
        stiffness = prim.GetAttribute(f"drive:{prefix}:physics:stiffness").Get()
        damping = prim.GetAttribute(f"drive:{prefix}:physics:damping").Get()
        max_force = prim.GetAttribute(f"drive:{prefix}:physics:maxForce").Get()
        if stiffness is None or damping is None:
            continue
        stiffness, damping = float(stiffness) * unit_scale, float(damping) * unit_scale
        effort = float(max_force) if max_force is not None else None
        if effort is not None and (not math.isfinite(effort) or effort <= 0):
            effort = None
        if not math.isfinite(stiffness) or not math.isfinite(damping) or stiffness < 0 or damping < 0:
            raise ValueError(f"Invalid source drive for {name}: kp={stiffness}, kd={damping}")
        result[name] = {
            "stiffness": stiffness,
            "damping": damping,
            "effort_limit_sim": effort,
        }
    if not CONTROL_JOINTS.issubset(result):
        raise ValueError(f"Pinned G2 USD has no drive parameters for: {sorted(CONTROL_JOINTS - set(result))}")
    return result


def _source_collision_policy(drive_usd: Path) -> dict:
    """Read the enabled collision shapes in the USD used by the benchmark.

    The separate robot-model Xacro enables some colliders that the actual
    benchmark USD explicitly disables, including the wheel cylinders.
    """
    from pxr import Usd, UsdPhysics

    source_root, _ = _verified_drive_usd(drive_usd)
    path = _verify(source_root, BENCHMARK_ROBOT_USD)
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Cannot open benchmark robot USD: {path}")
    layer_hashes = {}
    for layer in stage.GetUsedLayers():
        if not layer.realPath:
            continue
        layer_path = Path(layer.realPath).resolve()
        if not layer_path.is_relative_to(source_root):
            raise ValueError(f"Benchmark G2 USD references an external layer: {layer_path}")
        relative = layer_path.relative_to(source_root).as_posix()
        _verify(source_root, relative)
        layer_hashes[relative] = _sha256(layer_path)
    robot = stage.GetPrimAtPath("/genie")
    if not robot:
        raise ValueError("Benchmark robot USD has no /genie prim")
    links = {}
    for link in robot.GetChildren():
        enabled = disabled = 0
        for prim in Usd.PrimRange(link, Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                    enabled += 1
                else:
                    disabled += 1
        if enabled or disabled:
            links[link.GetName()] = {"enabled": enabled, "disabled": disabled}
    return {"robot_usd": BENCHMARK_ROBOT_USD, "layers_sha256": layer_hashes, "links": links}


def _source_control_limits(drive_usd: Path) -> dict[str, dict[str, float]]:
    """Read position, velocity and drive-effort limits from the benchmark USD."""
    from pxr import Usd, UsdPhysics

    source_root, _ = _verified_drive_usd(drive_usd)
    stage = Usd.Stage.Open(str(_verify(source_root, BENCHMARK_ROBOT_USD)))
    if stage is None:
        raise ValueError("Could not open the source benchmark G2 USD")
    result = {}
    for name in sorted(CONTROL_JOINTS):
        prim = stage.GetPrimAtPath(f"/genie/joints/{name}")
        if not prim or not prim.IsA(UsdPhysics.RevoluteJoint):
            raise ValueError(f"Benchmark G2 control joint {name!r} is not a revolute joint")
        joint = UsdPhysics.RevoluteJoint(prim)
        lower, upper = joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()
        velocity_attr = prim.GetAttribute("physxJoint:maxJointVelocity")
        velocity = velocity_attr.Get() if velocity_attr else None
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        force = drive.GetMaxForceAttr().Get() if drive else None
        values = (lower, upper, velocity, force)
        if any(value is None or not math.isfinite(float(value)) for value in values):
            raise ValueError(f"Benchmark G2 control joint {name!r} lacks finite source limits")
        lower, upper, velocity = (math.radians(float(value)) for value in (lower, upper, velocity))
        force = float(force)
        if lower >= upper or velocity <= 0 or force <= 0:
            raise ValueError(f"Benchmark G2 control joint {name!r} has invalid source limits")
        result[name] = {"lower": lower, "upper": upper, "velocity": velocity, "drive_force": force}
    return result


def _apply_source_control_limits(root: ET.Element, limits: dict[str, dict[str, float]]) -> None:
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    for name, values in limits.items():
        joint = joints.get(name)
        limit = joint.find("limit") if joint is not None else None
        if limit is None or joint.get("type") != "revolute":
            raise ValueError(f"Prepared G2 robot lacks controlled revolute joint {name!r}")
        for attr, key in (("lower", "lower"), ("upper", "upper"), ("velocity", "velocity"), ("effort", "drive_force")):
            limit.set(attr, f"{values[key]:.12g}")


def _apply_source_collision_policy(root: ET.Element, policy: dict) -> list[str]:
    """Remove Xacro collision shapes only when every source link shape is disabled."""
    removed = []
    for link in root.findall("link"):
        name = link.get("name")
        source = policy["links"].get(name)
        if source and source["enabled"] and source["disabled"] and link.findall("collision"):
            raise ValueError(f"Mixed enabled/disabled benchmark colliders cannot yet be mapped for {name!r}")
        if not source or source["enabled"]:
            continue
        collisions = link.findall("collision")
        for collision in collisions:
            link.remove(collision)
        if collisions:
            removed.append(name)
    return sorted(removed)


def _source_gripper_parameters(drive_usd: Path, *, benchmark: bool = False) -> dict:
    """Read the composed Genie Sim gripper limits, mimic and spherical loops.

    PhysX mimic semantics are ``joint + gearing * reference + offset = 0``;
    its documented defaults are gearing 1 and offset 0. USD joint limits and
    angular offsets are in degrees, while URDF and MJCF use radians.
    """
    from pxr import Usd, UsdPhysics

    source_root, _ = _verified_drive_usd(drive_usd)
    stage_path = _verify(source_root, BENCHMARK_ROBOT_USD) if benchmark else drive_usd
    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise ValueError(f"Cannot open pinned Genie Sim G2 robot USD: {drive_usd}")
    layer_hashes = {}
    for layer in stage.GetUsedLayers():
        if not layer.realPath:  # anonymous session layer
            continue
        layer_path = Path(layer.realPath).resolve()
        if not layer_path.is_relative_to(source_root):
            raise ValueError(f"G2 robot USD references a layer outside the pinned source: {layer_path}")
        relative = layer_path.relative_to(source_root).as_posix()
        verified = _verify(source_root, relative)
        layer_hashes[relative] = _sha256(verified)

    def body_name(prim, index: int) -> str:
        targets = prim.GetRelationship(f"physics:body{index}").GetTargets()
        if len(targets) != 1:
            raise ValueError(f"G2 joint {prim.GetPath()} needs one body{index} target")
        return targets[0].name

    def vector(prim, attribute: str) -> list[float]:
        value = prim.GetAttribute(attribute).Get()
        if value is None or len(value) != 3 or not all(math.isfinite(float(x)) for x in value):
            raise ValueError(f"G2 joint {prim.GetPath()} has an invalid {attribute}")
        return [float(x) for x in value]

    def quaternion(prim, attribute: str) -> list[float]:
        value = prim.GetAttribute(attribute).Get()
        if value is None:
            raise ValueError(f"G2 joint {prim.GetPath()} has no {attribute}")
        values = [float(value.GetReal()), *map(float, value.GetImaginary())]
        if not all(math.isfinite(x) for x in values) or sum(x * x for x in values) < 1e-12:
            raise ValueError(f"G2 joint {prim.GetPath()} has an invalid {attribute}")
        return values

    joints, loops = {}, {}
    for prim in stage.Traverse():
        name = prim.GetName()
        if "_gripper_" not in name:
            continue
        if prim.IsA(UsdPhysics.RevoluteJoint) and prim.GetPath().GetParentPath().pathString == "/genie/joints":
            joint = UsdPhysics.RevoluteJoint(prim)
            lower, upper = joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()
            armature = prim.GetAttribute("physxJoint:armature").Get()
            if lower is None or upper is None or armature is None:
                raise ValueError(f"G2 gripper joint {name} is missing limits or armature")
            lower, upper, armature = math.radians(float(lower)), math.radians(float(upper)), float(armature)
            if not all(math.isfinite(x) for x in (lower, upper, armature)) or lower >= upper or armature < 0:
                raise ValueError(f"G2 gripper joint {name} has invalid limits or armature")
            schema_op = prim.GetMetadata("apiSchemas")
            schemas = schema_op.GetAddedOrExplicitItems() if schema_op else []
            mimic_axes = [str(item).split(":", 1)[1] for item in schemas if str(item).startswith("PhysxMimicJointAPI:")]
            if len(mimic_axes) > 1:
                raise ValueError(f"G2 gripper joint {name} has multiple PhysX mimic axes")
            mimic = None
            if mimic_axes:
                prefix = f"physxMimicJoint:{mimic_axes[0]}:"
                targets = prim.GetRelationship(prefix + "referenceJoint").GetTargets()
                if len(targets) != 1:
                    raise ValueError(f"G2 mimic joint {name} needs one reference joint")
                gearing_attr = prim.GetAttribute(prefix + "gearing")
                offset_attr = prim.GetAttribute(prefix + "offset")
                gearing_value = gearing_attr.Get() if gearing_attr else None
                offset_value = offset_attr.Get() if offset_attr else None
                gearing = 1.0 if gearing_value is None else float(gearing_value)
                offset = 0.0 if offset_value is None else math.radians(float(offset_value))
                if not math.isfinite(gearing) or not math.isfinite(offset) or gearing == 0:
                    raise ValueError(f"G2 mimic joint {name} has invalid gearing or offset")
                mimic = {"joint": targets[0].name, "multiplier": -gearing, "offset": -offset}
            joints[name] = {
                "body0": body_name(prim, 0),
                "body1": body_name(prim, 1),
                "local_pos0": vector(prim, "physics:localPos0"),
                "local_pos1": vector(prim, "physics:localPos1"),
                "local_rot0_wxyz": quaternion(prim, "physics:localRot0"),
                "local_rot1_wxyz": quaternion(prim, "physics:localRot1"),
                "axis": str(joint.GetAxisAttr().Get()),
                "limits_rad": [lower, upper],
                "armature": armature,
                "mimic": mimic,
            }
        elif prim.IsA(UsdPhysics.SphericalJoint) and prim.GetPath().GetParentPath().pathString == "/genie/loop_joints":
            loops[name] = {
                "body0": body_name(prim, 0),
                "body1": body_name(prim, 1),
                "local_pos0": vector(prim, "physics:localPos0"),
                "local_pos1": vector(prim, "physics:localPos1"),
            }
    if len(joints) != 16 or len(loops) != 4 or sum(item["mimic"] is not None for item in joints.values()) != 2:
        raise ValueError("Pinned G2 USD does not have the expected 16 gripper joints, 4 loops and 2 mimic joints")
    return {"layers_sha256": dict(sorted(layer_hashes.items())), "joints": joints, "loops": loops}


def _apply_source_gripper_parameters(root: ET.Element, source: dict) -> None:
    """Overlay source USD gripper semantics onto the portable Xacro URDF."""
    from scipy.spatial.transform import Rotation

    joints = {
        joint.get("name"): joint
        for joint in root.findall("joint")
        if "_gripper_" in joint.get("name", "") and joint.get("type") == "revolute"
    }
    loops = {loop.get("name"): loop for loop in root.findall("loop_joint")}
    if set(joints) != set(source["joints"]) or set(loops) != set(source["loops"]):
        raise ValueError("G2 Xacro and source USD gripper joints do not match")
    for name, spec in source["joints"].items():
        joint = joints[name]
        if joint.find("parent").get("link") != spec["body0"] or joint.find("child").get("link") != spec["body1"]:
            raise ValueError(f"G2 Xacro and source USD joint bodies differ: {name}")
        origin = joint.find("origin")
        xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()]
        rpy = [float(x) for x in origin.get("rpy", "0 0 0").split()]
        w, x, y, z = spec["local_rot0_wxyz"]
        rotation_error = (Rotation.from_quat((x, y, z, w)).inv() * Rotation.from_euler("xyz", rpy)).magnitude()
        if (
            len(xyz) != 3
            or math.dist(xyz, spec["local_pos0"]) > 1e-6
            or math.dist(spec["local_pos1"], (0, 0, 0)) > 1e-6
            or abs(abs(spec["local_rot1_wxyz"][0]) - 1) > 1e-6
            or max(abs(x) for x in spec["local_rot1_wxyz"][1:]) > 1e-6
            or rotation_error > 1e-6
            or spec["axis"] != "Z"
            or joint.find("axis").get("xyz") != "0 0 1"
        ):
            raise ValueError(f"G2 Xacro and source USD joint frames differ: {name}")
        limit = joint.find("limit")
        limit.set("lower", f"{spec['limits_rad'][0]:.12g}")
        limit.set("upper", f"{spec['limits_rad'][1]:.12g}")
        for mimic in list(joint.findall("mimic")):
            joint.remove(mimic)
        if spec["mimic"] is not None:
            mimic = spec["mimic"]
            ET.SubElement(
                joint,
                "mimic",
                joint=mimic["joint"],
                multiplier=f"{mimic['multiplier']:.12g}",
                offset=f"{mimic['offset']:.12g}",
            )
    for name, spec in source["loops"].items():
        loop = loops[name]
        if loop.get("type") != "spherical":
            raise ValueError(f"G2 Xacro loop is not spherical: {name}")
        for index, tag in enumerate(("link1", "link2")):
            endpoint = loop.find(tag)
            if endpoint is None or endpoint.get("link") != spec[f"body{index}"]:
                raise ValueError(f"G2 Xacro and source USD loop bodies differ: {name}")
            endpoint.set("xyz", " ".join(f"{value:.12g}" for value in spec[f"local_pos{index}"]))


def _expand_xacro(source: Path, destination: Path) -> None:
    """Expand the upstream Xacro tree without requiring a ROS installation."""
    with tempfile.TemporaryDirectory(prefix="geniesim-g2-xacro-", dir=destination.parent) as temporary:
        package = Path(temporary) / PACKAGE_NAME
        variant_dir = package / "robots" / "genie" / "g2"
        variant_dir.mkdir(parents=True)
        shutil.copytree(source / "robots" / "genie" / "g2" / "xacro", variant_dir / "xacro")
        shutil.copy2(source / "robots" / "genie" / "g2" / f"{XACRO_VARIANT}.urdf.xacro", variant_dir)
        for path in package.rglob("*.xacro"):
            content = path.read_text(encoding="utf-8")
            content = content.replace("$(find genie_sim_robot_model)", package.as_posix())
            path.write_text(content, encoding="utf-8")

        entry = variant_dir / f"{XACRO_VARIANT}.urdf.xacro"
        command = [sys.executable, "-X", "utf8", "-c", "import xacro; xacro.main()", str(entry), "-o", str(destination)]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=False)
        if result.returncode:
            detail = "\n".join((result.stdout + result.stderr).splitlines()[-30:])
            raise RuntimeError(f"Cannot expand the official G2 Xacro:\n{detail}")


def _source_mesh_path(filename: str, mesh_root: Path) -> tuple[Path, Path]:
    if not filename.startswith(MESH_PACKAGE_URI):
        raise ValueError(f"Unexpected G2 mesh URI: {filename}")
    relative = Path(filename.removeprefix(MESH_PACKAGE_URI))
    source_mesh = (mesh_root / relative).resolve()
    if not source_mesh.is_relative_to(mesh_root.resolve()) or not source_mesh.is_file():
        raise FileNotFoundError(f"Missing or unsafe G2 mesh: {relative}")
    return source_mesh, relative


def _convert_visual_mesh(source: Path, destination_root: Path, relative: Path) -> list[tuple[Path, tuple[float, ...]]]:
    """Split Collada submeshes and retain their diffuse colors in URDF visuals."""
    import trimesh

    scene = trimesh.load_scene(source, process=False)
    if scene.is_empty or not scene.geometry:
        raise ValueError(f"Collada visual is empty: {source}")
    outputs = []
    for index, node in enumerate(scene.graph.nodes_geometry):
        transform, geometry_name = scene.graph.get(node)
        geometry = scene.geometry[geometry_name].copy()
        material = getattr(getattr(geometry, "visual", None), "material", None)
        if getattr(material, "image", None) is not None:
            raise ValueError(
                f"Textured Collada visual needs a texture-preserving converter: {source} ({geometry_name})"
            )
        color = getattr(material, "main_color", None)
        if color is None or len(color) != 4:
            raise ValueError(f"Collada submesh has no RGBA diffuse color: {source} ({geometry_name})")
        scale = 255.0 if max(float(channel) for channel in color) > 1.0 else 1.0
        rgba = tuple(float(channel) / scale for channel in color)
        geometry.apply_transform(transform)
        output_relative = relative.with_name(f"{relative.stem}_visual_{index}.stl")
        output_mesh = destination_root / output_relative
        output_mesh.parent.mkdir(parents=True, exist_ok=True)
        geometry.export(output_mesh)
        outputs.append((output_relative, rgba))
    return outputs


def _prepare_meshes(root: ET.Element, source: Path, destination: Path) -> list[str]:
    mesh_root = source / "robots" / "genie" / "g2" / "meshes"
    output_mesh_root = destination / "meshes"
    copied: set[str] = set()
    for link in root.findall("link"):
        for visual_index, visual in enumerate(list(link.findall("visual"))):
            reference = visual.find("geometry/mesh")
            if reference is None:
                continue
            source_mesh, relative = _source_mesh_path(reference.get("filename", ""), mesh_root)
            if source_mesh.suffix.lower() != ".dae":
                continue
            link.remove(visual)
            for part_index, (output_relative, rgba) in enumerate(
                _convert_visual_mesh(source_mesh, output_mesh_root, relative)
            ):
                item = ET.fromstring(ET.tostring(visual))
                item.set("name", f"{link.get('name')}_visual_{visual_index}_{part_index}")
                item.find("geometry/mesh").set("filename", (Path("meshes") / output_relative).as_posix())
                material = item.find("material")
                if material is None:
                    material = ET.SubElement(item, "material")
                material.set("name", f"{link.get('name')}_visual_mat_{visual_index}_{part_index}")
                for child in list(material):
                    material.remove(child)
                ET.SubElement(material, "color", rgba=" ".join(f"{channel:.9g}" for channel in rgba))
                link.append(item)
                copied.add(output_relative.as_posix())

    for reference in root.findall(".//mesh"):
        filename = reference.get("filename", "")
        if filename.startswith("meshes/"):
            continue
        source_mesh, relative = _source_mesh_path(filename, mesh_root)
        if source_mesh.suffix.lower() == ".dae":
            raise ValueError(f"Collada mesh is only supported in a visual element: {filename}")
        output_mesh = output_mesh_root / relative
        if relative.as_posix() not in copied:
            output_mesh.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_mesh, output_mesh)
            copied.add(relative.as_posix())
        reference.set("filename", (Path("meshes") / relative).as_posix())

    missing = [path for path in sorted(copied) if not (output_mesh_root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"G2 model has missing converted mesh files: {missing[:10]}")
    return sorted(copied)


def _inertia_triangle_violations(root: ET.Element) -> list[dict]:
    violations = []
    for link in root.findall("link"):
        inertia = link.find("inertial/inertia")
        if inertia is None:
            continue
        moments = [float(inertia.get(name, "0")) for name in ("ixx", "iyy", "izz")]
        largest = max(moments)
        if largest > sum(moments) - largest + 1e-9:
            violations.append({"link": link.get("name"), "principal_moments": moments})
    return violations


def _name_visuals(root: ET.Element) -> None:
    """Give each URDF visual a stable name to avoid MuJoCo's duplicate default names."""
    used: set[str] = set()
    for link in root.findall("link"):
        for index, visual in enumerate(link.findall("visual")):
            base = visual.get("name") or f"{link.get('name')}_visual_{index}"
            name, suffix = base, 1
            while name in used:
                name, suffix = f"{base}_{suffix}", suffix + 1
            visual.set("name", name)
            used.add(name)


def _add_mujoco_extensions(
    root: ET.Element,
    urdf_path: Path,
    output_xml: Path,
    violations: list[dict],
    source_drives: dict[str, dict[str, float | None]],
    source_gripper: dict,
) -> dict:
    import mujoco

    extension = root.find("mujoco")
    if extension is None:
        extension = ET.SubElement(root, "mujoco")
    compiler = extension.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(extension, "compiler")
    compiler.set("balanceinertia", "true")
    compiler.set("discardvisual", "false")
    compiler.set("fusestatic", "false")
    urdf_path.write_bytes(ET.tostring(root, encoding="utf-8", xml_declaration=True))
    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    mujoco.mj_saveLastXML(str(output_xml), model)

    mjcf = ET.parse(output_xml)
    mj_root = mjcf.getroot()
    inertia_floor_adjustments = _floor_small_mjcf_inertias(mj_root)
    for name, spec in source_gripper["joints"].items():
        joint = mj_root.find(f".//worldbody//joint[@name='{name}']")
        if joint is None:
            raise ValueError(f"Missing MuJoCo joint for the source G2 gripper: {name}")
        joint.set("armature", f"{spec['armature']:.12g}")
    equality = mj_root.find("equality")
    if equality is None:
        equality = ET.SubElement(mj_root, "equality")
    for loop in root.findall("loop_joint"):
        if loop.get("type") != "spherical":
            raise ValueError(f"Unsupported G2 loop joint: {loop.get('name')}")
        sites = []
        for tag in ("link1", "link2"):
            endpoint = loop.find(tag)
            if endpoint is None:
                raise ValueError(f"Incomplete G2 loop joint: {loop.get('name')}")
            link_name = endpoint.get("link")
            body = mj_root.find(f".//body[@name='{link_name}']")
            if body is None:
                raise ValueError(f"Missing MuJoCo body for G2 loop joint {loop.get('name')}: {link_name}")
            site_name = f"{loop.get('name')}_{tag}"
            ET.SubElement(body, "site", name=site_name, pos=endpoint.get("xyz", "0 0 0"), size="0.001")
            sites.append(site_name)
        ET.SubElement(equality, "connect", name=loop.get("name"), site1=sites[0], site2=sites[1])
    for joint in root.findall("joint"):
        mimic = joint.find("mimic")
        if mimic is None:
            continue
        name, source_name = joint.get("name"), mimic.get("joint")
        existing = [item for item in equality.findall("joint") if item.get("joint1") == name]
        if existing:
            continue
        offset = float(mimic.get("offset", "0"))
        multiplier = float(mimic.get("multiplier", "1"))
        ET.SubElement(
            equality,
            "joint",
            name=f"mimic_{name}",
            joint1=name,
            joint2=source_name,
            polycoef=f"{offset} {multiplier} 0 0 0",
        )
    for geom in mj_root.findall(".//worldbody//geom"):
        if geom.get("contype") != "0" or geom.get("conaffinity") != "0":
            geom.set("group", "3")
    actuators = mj_root.find("actuator")
    if actuators is None:
        actuators = ET.SubElement(mj_root, "actuator")
    for index in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        if not name:
            raise ValueError("G2 joint is missing a MuJoCo name")
        if not any(item.get("joint") == name for item in actuators):
            # Source USD also has zero-target drives on passive fingers and
            # mobile-base joints. Preserve those separately from the 24
            # policy-controlled joints rather than replacing them with zero gain.
            drive = source_drives.get(name)
            actuator = ET.SubElement(
                actuators,
                "position",
                name=name,
                joint=name,
                kp=f"{drive['stiffness']:.12g}" if drive else "0",
                kv=f"{drive['damping']:.12g}" if drive else "0",
            )
            if drive and drive["effort_limit_sim"] is not None:
                effort = drive["effort_limit_sim"]
                actuator.set("forcerange", f"{-effort:.12g} {effort:.12g}")
    mjcf.write(output_xml, encoding="utf-8", xml_declaration=True)
    verified = mujoco.MjModel.from_xml_path(str(output_xml))
    return {
        "links": len(root.findall("link")),
        "joints": len(root.findall("joint")),
        "movable_joints": sum(joint.get("type") != "fixed" for joint in root.findall("joint")),
        "mimic_constraints": sum(joint.find("mimic") is not None for joint in root.findall("joint")),
        "loop_constraints": len(root.findall("loop_joint")),
        "inertia_triangle_violations_balanced_by_mujoco": violations,
        "inertia_floor_adjustments": inertia_floor_adjustments,
        "mujoco": {
            "bodies": verified.nbody,
            "joints": verified.njnt,
            "geoms": verified.ngeom,
            "actuators": verified.nu,
        },
    }


def _floor_small_mjcf_inertias(mj_root: ET.Element, minimum: float = 1.1e-6) -> list[dict]:
    """Raise sub-micro inertial principal moments to Newton's stable floor."""
    adjustments = []
    for body in mj_root.findall(".//worldbody//body"):
        inertial = body.find("inertial")
        if inertial is None or inertial.get("diaginertia") is None:
            continue
        values = [float(value) for value in inertial.get("diaginertia", "").split()]
        if len(values) != 3 or not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError(f"Invalid principal inertia in MJCF body {body.get('name')!r}: {values}")
        floored = [max(value, minimum) for value in values]
        if floored != values:
            largest = max(floored)
            if largest > sum(floored) - largest + 1e-12:
                # This case is normally corrected by MuJoCo balanceinertia;
                # fail rather than changing the principal axis ratios silently.
                raise ValueError(f"Inertia floor violates the rigid-body triangle inequality: {body.get('name')}")
            inertial.set("diaginertia", " ".join(f"{value:.12g}" for value in floored))
            adjustments.append({"body": body.get("name"), "before": values, "after": floored})
    return adjustments


def correct_drive_provenance(*, output: Path, drive_usd: Path = DEFAULT_DRIVE_USD) -> dict:
    """Correct the legacy drive revision after verifying inputs and the prepared model."""
    output, drive_usd = Path(output).resolve(), Path(drive_usd).resolve()
    source_root, _ = _verified_drive_usd(drive_usd)
    provenance_path = output / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (
        provenance.get("format") != "roboverse.icl.geniesim_g2_robot.v2"
        or provenance.get("source_repository") != SOURCE_REPOSITORY
        or provenance.get("source_revision") != SOURCE_REVISION
        or provenance.get("robot_variant") != VARIANT
    ):
        raise ValueError(f"Unexpected prepared G2 robot provenance: {provenance_path}")
    drive = provenance.get("drive_source", {})
    if (
        drive.get("repository") != DRIVE_SOURCE_REPOSITORY
        or drive.get("robot_usd") != "robot/G2_omnipicker/robot.usda"
        or drive.get("robot_usd_sha256") != _sha256(drive_usd)
    ):
        raise ValueError("Prepared G2 drive source does not match the verified robot USD")
    if drive.get("revision") not in {_LEGACY_INCORRECT_DRIVE_REVISION, DRIVE_SOURCE_REVISION}:
        raise ValueError("Unexpected G2 drive revision; refusing to change its provenance")

    file_hashes = provenance.get("file_sha256")
    if not isinstance(file_hashes, dict) or not file_hashes:
        raise ValueError("Prepared G2 model has no file checksums")
    for relative, expected_hash in file_hashes.items():
        candidate = (output / relative).resolve()
        if not candidate.is_relative_to(output) or not candidate.is_file() or _sha256(candidate) != expected_hash:
            raise ValueError(f"Prepared G2 model checksum mismatch: {relative}")

    recorded = drive.get("parameters", {})
    source_parameters = _source_drive_parameters(drive_usd)
    if set(recorded) != CONTROL_JOINTS or set(drive.get("controlled_joints", [])) != CONTROL_JOINTS:
        raise ValueError("Prepared G2 drive provenance has a different controlled-joint set")
    for name in CONTROL_JOINTS:
        if name not in source_parameters or set(recorded[name]) != set(source_parameters[name]):
            raise ValueError(f"Prepared G2 drive parameters differ from the source: {name}")
        for key, expected in source_parameters[name].items():
            actual = recorded[name][key]
            if (expected is None) != (actual is None) or (
                expected is not None and not math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12)
            ):
                raise ValueError(f"Prepared G2 drive parameter differs from the source: {name}.{key}")

    if drive["revision"] == DRIVE_SOURCE_REVISION:
        return provenance
    drive["revision_correction"] = {
        "previous_revision": drive["revision"],
        "reason": "Legacy Git lookup resolved the parent workspace instead of the asset snapshot.",
        "verified_manifest_sha256": _sha256(source_root / "roboverse_source_manifest.json"),
        "verified_robot_usd_sha256": _sha256(drive_usd),
    }
    drive["revision"] = DRIVE_SOURCE_REVISION
    encoded = json.dumps(provenance, indent=2) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=".provenance-", suffix=".json", dir=output)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
        os.replace(temporary, provenance_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return provenance


def prepare(
    *, source: Path, output: Path, drive_usd: Path = DEFAULT_DRIVE_USD, benchmark_control_limits: bool = False
) -> dict:
    """Generate an attributed, portable G2 Omnipicker model and shared MJCF."""
    source, output, drive_usd = source.resolve(), output.resolve(), drive_usd.resolve()
    if _source_revision(source) != SOURCE_REVISION:
        raise ValueError(f"Expected {SOURCE_REPOSITORY} at pinned revision {SOURCE_REVISION}")
    _, drive_license_path = _verified_drive_usd(drive_usd)
    source_gripper = _source_gripper_parameters(drive_usd)
    benchmark_gripper = _source_gripper_parameters(drive_usd, benchmark=True)
    if any(source_gripper[key] != benchmark_gripper[key] for key in ("joints", "loops")):
        raise ValueError("G2 robot.usda gripper constraints differ from the benchmark robot_fix.usda")
    source_drives = _source_drive_parameters(drive_usd, benchmark=True)
    source_collisions = _source_collision_policy(drive_usd)
    source_limits = _source_control_limits(drive_usd) if benchmark_control_limits else None
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing G2 model: {output}")
    for required in ("LICENSE", "THIRD_PARTY_NOTICES"):
        if not (source / required).is_file():
            raise FileNotFoundError(f"Missing source attribution file: {required}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="geniesim-g2-", dir=output.parent) as temporary:
        destination = Path(temporary) / VARIANT
        destination.mkdir()
        urdf_path = destination / f"{VARIANT}.urdf"
        xml_path = destination / f"{VARIANT}.xml"
        _expand_xacro(source, urdf_path)
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        if root.tag != "robot" or root.find("link[@name='base_link']") is None:
            raise ValueError("Expanded file is not the official AgiBot G2 Omnipicker model")
        _apply_source_gripper_parameters(root, source_gripper)
        if source_limits is not None:
            _apply_source_control_limits(root, source_limits)
        disabled_collision_links = _apply_source_collision_policy(root, source_collisions)
        _name_visuals(root)
        mesh_files = _prepare_meshes(root, source, destination)
        violations = _inertia_triangle_violations(root)
        model_counts = _add_mujoco_extensions(root, urdf_path, xml_path, violations, source_drives, source_gripper)
        shutil.copy2(source / "LICENSE", destination / "LICENSE")
        shutil.copy2(source / "THIRD_PARTY_NOTICES", destination / "THIRD_PARTY_NOTICES")
        shutil.copy2(drive_license_path, destination / "LICENSE.GenieSimAssets")
        output_hashes = {
            path.relative_to(destination).as_posix(): _sha256(path)
            for path in sorted(destination.rglob("*"))
            if path.is_file()
        }
        provenance = {
            "format": "roboverse.icl.geniesim_g2_robot.v2",
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
            "source_license": "Apache-2.0 (AgiBot G2; Omnipicker, excludes Wuji assets)",
            "drive_source": {
                "repository": DRIVE_SOURCE_REPOSITORY,
                "revision": DRIVE_SOURCE_REVISION,
                "license": DRIVE_SOURCE_LICENSE,
                "robot_usd": BENCHMARK_ROBOT_USD,
                "robot_usd_sha256": _sha256(_verify(drive_usd.parents[2], BENCHMARK_ROBOT_USD)),
                "units": "angular stiffness and damping converted from per-degree to per-radian",
                "controlled_joints": sorted(CONTROL_JOINTS),
                "parameters": {name: source_drives[name] for name in sorted(CONTROL_JOINTS)},
                "uncontrolled_joint_policy": "source zero-target drives; mimic and loop constraints remain active",
                "uncontrolled_parameters": {name: source_drives[name] for name in sorted(source_drives.keys() - CONTROL_JOINTS)},
            },
            "gripper_constraint_source": {
                "repository": DRIVE_SOURCE_REPOSITORY,
                "revision": DRIVE_SOURCE_REVISION,
                "license": DRIVE_SOURCE_LICENSE,
                "units": "USD angular limits and mimic offsets converted from degrees to radians",
                **source_gripper,
            },
            "collision_source": {
                "repository": DRIVE_SOURCE_REPOSITORY,
                "revision": DRIVE_SOURCE_REVISION,
                **source_collisions,
                "disabled_xacro_collision_links": disabled_collision_links,
            },
            "robot_variant": VARIANT,
            "urdf_source": f"robots/genie/g2/{XACRO_VARIANT}.urdf.xacro",
            "visual_conversion": "Collada submeshes split to STL, preserving per-submesh diffuse RGBA in URDF",
            "collision_geometry": "Upstream G2 Xacro geometry, excluding links whose benchmark USD colliders are all disabled; enabled gripper geometry remains an approximation",
            "mujoco_model": "URDF import converted to shared MJCF; gripper mimic and spherical loop constraints follow the composed Genie Sim USD",
            "actuator_gains": "Controlled joints use source USD PD gains, angular values converted to per-radian units, and source effort limits",
            "inertia_policy": "MuJoCo balanceinertia plus a 1.1e-6 kg*m^2 principal-moment floor; shared MJCF is used by all three backends",
            "model": model_counts,
            "mesh_files": mesh_files,
            "file_sha256": output_hashes,
        }
        if source_limits is not None:
            provenance["control_limit_source"] = {
                "repository": DRIVE_SOURCE_REPOSITORY,
                "revision": DRIVE_SOURCE_REVISION,
                "robot_usd": BENCHMARK_ROBOT_USD,
                "units": "revolute position radians, velocity radians/second, drive force N*m",
                "effort_semantics": "URDF effort is the source drive maxForce; each controlled joint has one actuator",
                "limits": source_limits,
            }
        (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        shutil.move(str(destination), output)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Pinned genie_sim_robot_model checkout")
    parser.add_argument("--output", type=Path, required=True, help="New destination under roboverse_data/icl/robots")
    parser.add_argument(
        "--drive-usd",
        type=Path,
        default=DEFAULT_DRIVE_USD,
        help="Pinned GenieSimAssets G2 robot.usda; benchmark drives come from its sibling robot_fix.usda",
    )
    parser.add_argument(
        "--correct-drive-provenance",
        action="store_true",
        help="Verify and correct the legacy drive revision in an existing prepared G2 model",
    )
    parser.add_argument(
        "--benchmark-control-limits",
        action="store_true",
        help="Use benchmark USD position, velocity and drive-force limits for controlled joints",
    )
    args = parser.parse_args()
    if args.correct_drive_provenance:
        correct_drive_provenance(output=args.output, drive_usd=args.drive_usd)
        print(f"Verified G2 drive provenance: {args.output}")
        return
    if args.source is None:
        parser.error("--source is required when preparing a G2 model")
    provenance = prepare(
        source=args.source,
        output=args.output,
        drive_usd=args.drive_usd,
        benchmark_control_limits=args.benchmark_control_limits,
    )
    model = provenance["model"]
    print(
        f"Prepared {provenance['robot_variant']}: {model['links']} links, {model['joints']} joints, "
        f"{len(provenance['mesh_files'])} mesh files -> {args.output}"
    )


if __name__ == "__main__":
    main()
