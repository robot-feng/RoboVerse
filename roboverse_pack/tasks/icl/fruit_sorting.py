# Copyright (c) 2023-2026 AgiBot Inc.
# SPDX-License-Identifier: MPL-2.0
#
# Adapted from Genie Sim benchmark (https://github.com/AgibotTech/genie_sim).
# Changes: use MetaSim scenarios, state observations and independent task lifecycle.
# Full license: roboverse_pack/tasks/icl/LICENSE.geniesim-benchmark
"""A G1 tabletop fruit-sorting task with a state-based success condition."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

from metasim.constants import PhysicStateType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.lights import DistantLightCfg, DomeLightCfg
from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.task.registry import register_task
from metasim.utils.hf_util import LOCAL_DIR
from metasim.utils.math import quat_apply
from roboverse_pack.tasks.icl.geniesim_scene import geniesim_g1_cameras
from roboverse_pack.tasks.icl.g1_joint_task import CONTROL_JOINTS, G1JointPositionTask

SCENE_ROOT = Path(LOCAL_DIR) / "icl" / "scenes" / "fruit_sorting_1"
ROBOT_ROOT = Path(LOCAL_DIR) / "icl" / "robots" / "geniesim_g1_omnipicker_closed_v11"
APPLE_DELIVERY_SCENE_ROOT = Path(LOCAL_DIR) / "icl" / "scenes" / "g1_apple_scene_delivery_20261031_v4_r4"

# Source G1 initial state, mapped to the URDF's waist, head and arm joint names.
_LEFT_ARM = (-1.0751, 0.6109, 0.2793, -1.2846, 0.7295, 1.4957, -0.1868)
_RIGHT_ARM = (1.0734, -0.6109, -0.2793, 1.2846, -0.7313, -1.4957, 0.1868)
def build_geniesim_g1_robot(asset_root: Path) -> RobotCfg:
    """Build the prepared G1 Omnipicker config used by GenieSim ICL tasks."""
    urdf, mjcf = asset_root / "G1_omnipicker.urdf", asset_root / "G1_omnipicker.xml"
    provenance = json.loads((asset_root / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("robot_variant") != "G1_omnipicker" or not urdf.is_file() or not mjcf.is_file():
        raise ValueError("Prepare the AgiBot G1 Omnipicker model before building this task")
    drive_source = provenance.get("drive_source", {})
    drive_parameters = drive_source.get("parameters", {})
    active_joint_names = set(drive_source.get("active_joint_names") or ())
    asset_format = provenance.get("format")
    collision_geometry = provenance.get("collision_geometry", {})
    ground_alignment = provenance.get("collision_ground_alignment", {})
    if (
        asset_format not in {"roboverse.icl.geniesim_g1_robot.v4"}
        or drive_source.get("revision") != provenance.get("source_revision")
        or drive_source.get("robot_usd_sha256") != provenance.get("robot_usd_sha256")
        or active_joint_names != set(CONTROL_JOINTS)
    ):
        raise ValueError("Prepared G1 model is missing its pinned source drives or active-joint policy")
    if asset_format == "roboverse.icl.geniesim_g1_robot.v4":
        if (
            ground_alignment.get("link") != "base_link"
            or ground_alignment.get("method")
            != "shift source collision proxies to match the lowest composed chassis visual"
            or float(ground_alignment.get("vertical_shift_m", 0.0)) <= 0.0
        ):
            raise ValueError("Prepare the ground-aligned G1 v11 asset before building this task")
        records = collision_geometry.get("records", [])
        invalid_mesh_parts = [
            part["mesh"] for record in records for part in record.get("parts", []) if not part.get("watertight")
        ]
        if collision_geometry.get("source") != "robot.usda PhysicsCollisionAPI" or not records or invalid_mesh_parts:
            raise ValueError("Prepared G1 v4 model has missing or invalid source USD collision proxies")
        attachments = provenance.get("visual_attachment_transforms", {})
        if (
            attachments.get("source") != "robot.usda composed link visuals relative to link"
            or attachments.get("source_sha256") != provenance.get("robot_usd_sha256")
            or attachments.get("source_root") != "/G01"
        ):
            raise ValueError("Prepared G1 v4 model is missing its composed robot.usda visual attachment transforms")
        base_attachment = attachments.get("links", {}).get("base_link", {})
        base_xyz = np.asarray(base_attachment.get("xyz", ()), dtype=np.float64)
        if base_xyz.shape != (3,) or not np.isfinite(base_xyz).all() or base_xyz[2] < 0.2:
            raise ValueError("Prepared G1 v4 model is missing the authored base visual lift needed to clear the floor")
    closed = provenance.get("loop_constraints") == 4 and provenance.get("mimic_constraints") == 2
    if not closed:
        raise ValueError("G1 gripper loop and mimic constraints are required for fruit sorting")
    root = ET.parse(urdf).getroot()
    if asset_format == "roboverse.icl.geniesim_g1_robot.v4":
        for link_name, expected in provenance["visual_attachment_transforms"]["links"].items():
            expected_xyz = np.asarray(expected["xyz"], dtype=np.float64)
            expected_rpy = np.asarray(expected["rpy"], dtype=np.float64)
            for visual in root.findall(f"link[@name='{link_name}']/visual"):
                if visual.find("geometry/mesh") is None:
                    continue
                origin = visual.find("origin")
                actual_xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
                actual_rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
                if not np.allclose(actual_xyz, expected_xyz, atol=1e-8) or not np.allclose(
                    actual_rpy, expected_rpy, atol=1e-8
                ):
                    raise ValueError(f"Prepared G1 v4 visual does not match the composed USD attachment: {link_name}")
    joints = [item for item in root.findall("joint") if item.get("type") in ("revolute", "prismatic")]
    defaults = {joint.get("name"): 0.0 for joint in joints}
    defaults.update({
        "idx01_body_joint1": 0.42,
        "idx02_body_joint2": 0.8727,
        "idx11_head_joint1": 0.0,
        "idx12_head_joint2": 0.4363,
    })
    for side, offset, values in (("l", 20, _LEFT_ARM), ("r", 60, _RIGHT_ARM)):
        defaults.update({f"idx{offset + i:02d}_arm_{side}_joint{i}": value for i, value in enumerate(values, 1)})
    actuators = {}
    limits = {}
    for joint in joints:
        name, limit = joint.get("name"), joint.find("limit")
        if limit is None or name is None:
            raise ValueError("G1 movable joint is missing its name or limits")
        limits[name] = (float(limit.get("lower")), float(limit.get("upper")))
        source_drive = drive_parameters.get(name)
        if name in CONTROL_JOINTS and source_drive is None:
            raise ValueError(f"Prepared G1 source USD has no actuator parameters for controlled joint {name}")
        drive = source_drive if name in active_joint_names else None
        if drive is None:
            # Keep mimic/loop followers passive, but preserve their authored
            # URDF effort bound. Newton otherwise warns that the config
            # overrides the joint gains while silently retaining this limit.
            drive = {
                "stiffness": 0.0,
                "damping": 0.0,
                "effort_limit_sim": float(limit.get("effort")) if limit.get("effort") is not None else None,
            }
        actuators[name] = BaseActuatorCfg(
            stiffness=float(drive["stiffness"]),
            damping=float(drive["damping"]),
            effort_limit_sim=(float(drive["effort_limit_sim"]) if drive["effort_limit_sim"] is not None else None),
            velocity_limit=float(limit.get("velocity")),
        )
        if not limits[name][0] <= defaults[name] <= limits[name][1]:
            raise ValueError(f"Initial position is outside {name}'s source limits")
    robot = RobotCfg(
        name="agibot_g1",
        urdf_path=str(urdf),
        mjcf_path=str(mjcf),
        num_joints=len(joints),
        default_position=(-0.8, 0.0, -0.01),
        default_joint_positions=defaults,
        joint_limits=limits,
        fix_base_link=True,
        enabled_gravity=False,
        enabled_self_collisions=False,
        actuators=actuators,
        control_type={name: "position" for name in defaults},
    )
    if closed:
        robot.file_type["genesis"] = "mjcf"
    return robot


def build_scenario(
    *,
    simulator: str,
    scene_root: Path = SCENE_ROOT,
    robot_root: Path = ROBOT_ROOT,
    image_size: int = 640,
    headless: bool = True,
) -> ScenarioCfg:
    """Load the prepared source scene and one G1 robot through MetaSim."""
    if simulator not in ("mujoco", "genesis", "newton") or image_size < 16:
        raise ValueError("Choose a supported simulator and image size of at least 16")
    scene_root, robot_root = Path(scene_root), Path(robot_root)
    metadata = json.loads((scene_root / "scene.json").read_text(encoding="utf-8"))
    if metadata.get("format") != "roboverse.icl.fruit_scene.v1":
        raise ValueError("Not a prepared fruit-sorting scene")
    objects = []
    for name, entry in metadata["objects"].items():
        path = scene_root / name
        newton_urdf = entry.get("newton_urdf")
        if newton_urdf is not None and (
            not isinstance(newton_urdf, str)
            or not newton_urdf.lower().endswith(".urdf")
            or Path(newton_urdf).name != newton_urdf
            or "\\" in newton_urdf
        ):
            raise ValueError(f"Invalid Newton URDF filename for scene object {name!r}")
        urdf_path = path / (newton_urdf if simulator == "newton" and newton_urdf else "object.urdf")
        if not urdf_path.is_file():
            raise FileNotFoundError(f"Scene object {name!r} is missing its URDF: {urdf_path}")
        resources = [
            str(resource.resolve())
            for resource in path.iterdir()
            if resource.is_file() and resource.suffix.lower() in {".obj", ".mtl", ".png", ".jpg", ".jpeg"}
        ]
        objects.append(
            RigidObjCfg(
                name=name,
                urdf_path=str(urdf_path),
                mjcf_path=str(path / "object.xml"),
                extra_resources=resources,
                physics=PhysicStateType.GEOM if entry["fixed"] else PhysicStateType.RIGIDBODY,
                default_position=tuple(entry["position"]),
                default_orientation=tuple(entry["quaternion_wxyz"]),
            )
        )
    for primitive in metadata.get("primitives", []):
        # Older prepared scene manifests included this large source wall. Keep
        # loading compatible with them while framing the task around the robot
        # and the tabletop workspace.
        if primitive.get("name") == "back_wall":
            continue
        objects.append(
            PrimitiveCubeCfg(
                name=primitive["name"],
                size=primitive["size"],
                color=primitive["color"],
                physics=PhysicStateType.GEOM,
                default_position=tuple(primitive["position"]),
                default_orientation=tuple(primitive.get("orientation_wxyz", (1.0, 0.0, 0.0, 0.0))),
            )
        )
    robot = build_geniesim_g1_robot(robot_root)
    if simulator == "mujoco":
        # The source USD drives were authored for PhysX's implicit joint-drive
        # solver. Using those gains unchanged with MuJoCo's explicit 1/1980 s
        # step saturates the G1 joints on the first frame. Keep source geometry,
        # effort bounds, and position-target semantics, with solver-stable gains.
        for name, actuator in robot.actuators.items():
            if name == "idx01_body_joint1":
                gain_scale = 1e-3
            elif name in {"idx41_gripper_l_outer_joint1", "idx81_gripper_r_outer_joint1"}:
                gain_scale = 1e-4
            elif "_arm_" in name and name in CONTROL_JOINTS:
                gain_scale = 5e-5
            else:
                gain_scale = 2e-5
            actuator.stiffness *= gain_scale
            actuator.damping *= gain_scale
    robot_initial_state = metadata.get("robot_initial_state", {})
    if robot_initial_state:
        position = tuple(float(value) for value in robot_initial_state["position"])
        orientation = tuple(float(value) for value in robot_initial_state["orientation_wxyz"])
        if len(position) != 3 or len(orientation) != 4:
            raise ValueError("G1 scene robot_initial_state needs a position and WXYZ orientation")
        if not all(np.isfinite(value) for value in (*position, *orientation)):
            raise ValueError("G1 scene robot_initial_state must contain finite values")
        orientation_norm = float(np.linalg.norm(orientation))
        if orientation_norm < 1e-8:
            raise ValueError("G1 scene robot_initial_state orientation must be nonzero")
        robot.default_position = position
        robot.default_orientation = tuple(value / orientation_norm for value in orientation)
        joint_positions = robot_initial_state.get("joint_positions", {})
        unknown_joints = set(joint_positions) - set(robot.default_joint_positions)
        if unknown_joints:
            raise ValueError(f"G1 scene has unknown initial joints: {sorted(unknown_joints)}")
        checked_positions = {name: float(value) for name, value in joint_positions.items()}
        if not all(np.isfinite(value) for value in checked_positions.values()):
            raise ValueError("G1 scene joint initial positions must be finite")
        robot.default_joint_positions.update(checked_positions)
    overview = PinholeCameraCfg(
        name="overview",
        width=image_size,
        height=image_size,
        data_types=["rgb"],
        # Match the preferred rear-side three-quarter view: keep the robot to
        # the left of the tabletop while showing the tabletop and floor.
        pos=(1.7, -1.8, 1.9),
        look_at=(0.0, 0.15, 0.75),
        focal_length=20.0,
    )
    return ScenarioCfg(
        robots=[robot],
        objects=objects,
        cameras=[*geniesim_g1_cameras(robot_root, image_size), overview],
        simulator=simulator,
        headless=headless,
        num_envs=1,
        lights=[DomeLightCfg(intensity=450), DistantLightCfg(intensity=1100, polar=30, azimuth=45)],
        sim_params=SimParamCfg(
            dt=1 / 1980,
            njmax=1024,
            mujoco_actuator_limits_at_joint=simulator == "mujoco",
            mujoco_use_scenario_lights=True,
            newton_use_scenario_lights=True,
            newton_camera_background_rgb=(23, 36, 53),
        ),
        decimation=66,
    )


def _world_aabb(pose: torch.Tensor, bounds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return world-aligned AABB bounds for a batched local box and WXYZ pose."""
    if pose.ndim != 2 or pose.shape[1] != 7 or bounds.shape != (2, 3):
        raise ValueError("Expected pose (N,7) and local bounds (2,3)")
    corners = torch.stack([
        torch.stack([bounds[x, 0], bounds[y, 1], bounds[z, 2]]) for x in (0, 1) for y in (0, 1) for z in (0, 1)
    ])
    rotated = quat_apply(
        pose[:, None, 3:7].expand(-1, 8, -1).reshape(-1, 4),
        corners[None].expand(pose.shape[0], -1, -1).reshape(-1, 3),
    ).reshape(pose.shape[0], 8, 3)
    world = rotated + pose[:, None, :3]
    return world.amin(dim=1), world.amax(dim=1)


def point_inside_world_aabb(point: torch.Tensor, pose: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    """Test a point against the world-aligned bounds of a rotated local box."""
    if point.ndim != 2 or point.shape[1] != 3 or pose.shape != (point.shape[0], 7) or bounds.shape != (2, 3):
        raise ValueError("Expected point (N,3), pose (N,7), and local bounds (2,3)")
    lower, upper = _world_aabb(pose, bounds)
    return ((point >= lower) & (point <= upper)).all(dim=1)


@register_task("icl.fruit_sorting")
class FruitSortingTask(G1JointPositionTask):
    """Move the designated peach into the matching storage box."""

    scenario = None
    supported_simulators = ("mujoco", "genesis", "newton")
    max_episode_steps = 600

    def __init__(
        self,
        scenario: ScenarioCfg | None = None,
        device=None,
        *,
        simulator: str = "mujoco",
        scene_root: Path = SCENE_ROOT,
        robot_root: Path = ROBOT_ROOT,
        image_size: int = 256,
        headless: bool = True,
        metadata: dict | None = None,
    ):
        self._metadata = metadata or json.loads((Path(scene_root) / "scene.json").read_text(encoding="utf-8"))
        if scenario is None:
            scenario = build_scenario(
                simulator=simulator,
                scene_root=scene_root,
                robot_root=robot_root,
                image_size=image_size,
                headless=headless,
            )
        super().__init__(scenario, device)

    def _observe_outcome(self, state) -> None:
        target = state.objects[self._metadata["target"]].root_state
        box = state.objects[self._metadata["destination"]].root_state
        destination_metadata = self._metadata["objects"][self._metadata["destination"]]
        bounds = torch.as_tensor(
            destination_metadata.get("goal_bounds_m", destination_metadata["bounds_m"]),
            dtype=target.dtype,
            device=target.device,
        )
        inside = point_inside_world_aabb(target[:, :3], box[:, :7], bounds)
        self._inside_frames = torch.where(inside, self._inside_frames + 1, 0)
        self._success |= self._inside_frames >= 2
        self._failure |= (target[:, 2].abs() < 0.3) & ~self._success

@register_task("icl.geniesim_g1_apple_delivery")
class GenieSimG1AppleDeliveryTask(FruitSortingTask):
    """Pick up the supplied GenieSim apple and place it on the pink plate."""

    # This scene needs approach, grasp, lift, transport, place, and settle phases.
    max_episode_steps = 1800

    def __init__(self, scenario=None, device=None, *, simulator="mujoco", image_size=256, headless=True):
        super().__init__(
            scenario=scenario,
            device=device,
            simulator=simulator,
            scene_root=APPLE_DELIVERY_SCENE_ROOT,
            robot_root=ROBOT_ROOT,
            image_size=image_size,
            headless=headless,
        )
