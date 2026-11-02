# Copyright (c) 2026 AgiBot Inc.
# SPDX-License-Identifier: MPL-2.0
#
# Adapted from Genie Sim benchmark task execution (https://github.com/AgibotTech/genie_sim).
# Changes: execute imported source checker graphs through MetaSim task/state APIs.
# Full license: roboverse_pack/tasks/icl/LICENSE.geniesim-benchmark
"""Execute imported Genie Sim G1/G2 benchmark runs as RoboVerse tasks."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping
from copy import copy, deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces

from metasim.scenario.cameras import BaseCameraCfg, FisheyeCameraCfg, PinholeCameraCfg
from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv
from metasim.task.registry import register_task
from metasim.utils.fisheye import validate_fisheye_domain
from metasim.utils.hf_util import LOCAL_DIR
from metasim.utils.pinhole import apply_opencv_pinhole_distortion, opencv_pinhole_valid_mask
from roboverse_pack.tasks.icl.fruit_sorting import CONTROL_JOINTS, ROBOT_ROOT, build_geniesim_g1_robot
from roboverse_pack.tasks.icl.geniesim_checker import (
    CheckerResult,
    CheckerStatus,
    GenieSimChecker,
    checker_frame_from_states,
)
from roboverse_pack.tasks.icl.geniesim_scene import (
    build_run_scenario,
    geniesim_g1_cameras,
    geniesim_g2_cameras,
    geniesim_object_bounds,
    load_run_instance,
)

SPEC_PATH = Path(LOCAL_DIR) / "icl" / "geniesim_benchmark_specs_source_scene_v2" / "benchmark.json"
ASSETS_ROOT = Path(LOCAL_DIR) / "icl" / "objects" / "geniesim"
CACHE_ROOT = Path(LOCAL_DIR) / "icl" / "cache" / "geniesim_scene_wrappers_v6"
JOINT_POSITION_ACTION_SPEC = "g1_control_joint_position_target_v1"
G2_JOINT_POSITION_ACTION_SPEC = "g2_control_joint_position_target_v1"
CARTESIAN_ACTION_SPEC = "tcp_world_delta_xyz_rotvec_gripper_open_v1"
BIMANUAL_CARTESIAN_ACTION_SPEC = "bimanual_tcp_world_delta_xyz_rotvec_gripper_open_v1"
G2_ROBOT_ROOT = Path(LOCAL_DIR) / "icl" / "robots" / "geniesim_g2_omnipicker_v11"
G2_DRIVE_SOURCE_REVISION = "a0813aad7c16165daffbe6c2737754e0344809ee"
G2_ARM_JOINTS = tuple(
    [f"idx{20 + index:02d}_arm_l_joint{index}" for index in range(1, 8)]
    + [f"idx{60 + index:02d}_arm_r_joint{index}" for index in range(1, 8)]
)
G2_CONTROL_JOINTS = (
    *G2_ARM_JOINTS,
    *(f"idx{index:02d}_body_joint{index}" for index in range(1, 6)),
    "idx11_head_joint1",
    "idx12_head_joint2",
    "idx13_head_joint3",
    "idx41_gripper_l_outer_joint1",
    "idx81_gripper_r_outer_joint1",
)
G2_GRIPPER_CENTER_FROM_BASE = np.asarray(
    [[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.14308], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)


def _select_cameras(cameras: tuple[BaseCameraCfg, ...] | list[BaseCameraCfg], names: tuple[str, ...] | None):
    """Select configured views by name without changing their authored order."""
    if names is None:
        return tuple(cameras)
    if not names or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("camera_names must contain unique nonempty camera names")
    available = {camera.name for camera in cameras}
    missing = set(names) - available
    if missing:
        raise ValueError(f"Unknown camera preset(s): {', '.join(sorted(missing))}")
    requested = set(names)
    return tuple(camera for camera in cameras if camera.name in requested)


def _apply_camera_calibrations(cameras, calibrations):
    """Replace selected lens models while retaining each camera's mount pose."""
    if not calibrations:
        return tuple(cameras)
    if not isinstance(calibrations, Mapping):
        raise ValueError("camera_calibrations must map camera names to lens parameters")
    missing = set(calibrations) - {camera.name for camera in cameras}
    if missing:
        raise ValueError(f"Calibration references unselected cameras: {sorted(missing)}")
    allowed = {"projection_model", "calibration_size", "optical_center", "focal_length_px", "coefficients",
               "max_fov_rad", "calibration_source", "calibration_source_sha256", "max_fov_source"}
    result = []
    for camera in cameras:
        if camera.name not in calibrations:
            result.append(camera)
            continue
        values = dict(calibrations[camera.name])
        if set(values) - allowed:
            raise ValueError(f"Unknown camera calibration fields: {sorted(set(values) - allowed)}")
        for field in ("calibration_size", "optical_center", "focal_length_px", "coefficients"):
            if values.get(field) is not None:
                values[field] = tuple(values[field])
        values.setdefault("calibration_source", "user_supplied")
        size = values.get("calibration_size")
        if not size or len(size) != 2 or not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in size):
            raise ValueError("calibration_size must contain positive integer width and height")
        updated = FisheyeCameraCfg(
            name=camera.name, width=camera.width, height=max(1, round(camera.width * size[1] / size[0])),
            data_types=["rgb"], pos=camera.pos, look_at=camera.look_at, mount_to=camera.mount_to,
            mount_link=camera.mount_link, mount_pos=camera.mount_pos, mount_quat=camera.mount_quat, **values,
        )
        validate_fisheye_domain(updated)
        result.append(updated)
    return tuple(result)


def _single_tcp_arm(instance: Mapping[str, Any], problem_id: str, problem_index: int, requested: str | None) -> str:
    """Select an arm using source metadata and checker references, or an explicit override."""
    if requested is not None:
        if requested not in {"left", "right"}:
            raise ValueError("tcp_arm must be 'left' or 'right'")
        return requested

    instructions = instance.get("instructions", {}).get("instructions", [])
    item = instructions[problem_index] if isinstance(instructions, list) and problem_index < len(instructions) else {}
    item = item if isinstance(item, Mapping) else {}
    grippers = item.get("gripper", {})
    metadata_sides = (
        {
            str(value.get("side", "")).lower()
            for value in grippers.values()
            if isinstance(value, Mapping) and value.get("side")
        }
        if isinstance(grippers, Mapping)
        else set()
    )
    graph = instance.get("checker_graphs", {}).get(problem_id, {})
    graph_text = json.dumps(graph, ensure_ascii=False).lower()
    checker_sides = {side for side in ("left", "right") if f"{side}_gripper" in graph_text}
    if "both" in metadata_sides or len(metadata_sides) > 1:
        raise ValueError("This problem specifies both arms; use bimanual_tcp_delta or specify tcp_arm")
    if metadata_sides:
        side = next(iter(metadata_sides))
        if side not in {"left", "right"}:
            raise ValueError(f"Unsupported source gripper side {side!r}; specify tcp_arm")
        if len(checker_sides) == 1 and side not in checker_sides:
            raise ValueError("Source gripper metadata conflicts with the checker; specify tcp_arm explicitly")
        return side
    if len(checker_sides) > 1:
        raise ValueError("The source checker uses both arms; use bimanual_tcp_delta or specify tcp_arm")
    if checker_sides:
        return next(iter(checker_sides))
    instruction = str(item.get("instruction", "")).lower()
    mentioned = set(re.findall(r"\b(left|right)\s+(?:arm|hand|gripper)\b", instruction))
    if len(mentioned) > 1:
        raise ValueError("The instruction uses both arms; use bimanual_tcp_delta or specify tcp_arm")
    return next(iter(mentioned)) if mentioned else "right"


def build_geniesim_g2_robot(asset_root: Path = G2_ROBOT_ROOT) -> RobotCfg:
    """Build the pinned G2 Omnipicker with the shared constrained MJCF model."""
    import xml.etree.ElementTree as ET

    asset_root = Path(asset_root)
    provenance = json.loads((asset_root / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("format") != "roboverse.icl.geniesim_g2_robot.v2":
        raise ValueError(f"Unsupported prepared G2 robot asset: {asset_root}")
    if provenance.get("source_revision") != "6d5b2159806ae0e64b432b638e9ee288d8bddc8f":
        raise ValueError("Prepared G2 robot does not match the pinned AgiBot model revision")
    if provenance.get("robot_variant") != "G2_crs_omnipicker":
        raise ValueError(f"Expected G2 CRS Omnipicker, got {provenance.get('robot_variant')!r}")
    drive_source = provenance.get("drive_source", {})
    if drive_source.get("revision") != G2_DRIVE_SOURCE_REVISION:
        raise ValueError("Prepared G2 robot is missing drive settings from the pinned GenieSimAssets revision")
    urdf_path, mjcf_path = asset_root / "G2_crs_omnipicker.urdf", asset_root / "G2_crs_omnipicker.xml"
    if not urdf_path.is_file() or not mjcf_path.is_file():
        raise FileNotFoundError(f"Prepared G2 robot is incomplete: {asset_root}")

    defaults = {}
    limits = {}
    velocities = {}
    for joint in ET.parse(urdf_path).getroot().findall("joint"):
        if joint.get("type") not in {"revolute", "continuous", "prismatic"}:
            continue
        name = joint.get("name")
        limit = joint.find("limit")
        if not name:
            raise ValueError("Prepared G2 robot contains an unnamed movable joint")
        if joint.get("type") == "continuous":
            limits[name] = (-np.pi, np.pi)
            defaults[name] = 0.0
            velocities[name] = float(limit.get("velocity")) if limit is not None and limit.get("velocity") else None
        elif limit is not None:
            low, high = float(limit.get("lower", "0")), float(limit.get("upper", "0"))
            limits[name] = (low, high)
            defaults[name] = min(max(0.0, low), high)
            velocities[name] = float(limit.get("velocity")) if limit.get("velocity") else None
        else:
            raise ValueError(f"Prepared G2 joint has no position limits: {name}")
    missing = set(G2_CONTROL_JOINTS) - set(defaults)
    if missing:
        raise ValueError(f"Prepared G2 model is missing source control joints: {sorted(missing)}")
    drive_parameters = drive_source.get("parameters", {})
    missing_drives = set(G2_CONTROL_JOINTS) - set(drive_parameters)
    if missing_drives:
        raise ValueError(f"Prepared G2 model is missing source actuator parameters: {sorted(missing_drives)}")
    actuators = {
        name: BaseActuatorCfg(
            stiffness=float(drive_parameters[name]["stiffness"]),
            damping=float(drive_parameters[name]["damping"]),
            effort_limit_sim=(
                float(drive_parameters[name]["effort_limit_sim"])
                if drive_parameters[name].get("effort_limit_sim") is not None
                else None
            ),
            velocity_limit=velocities[name],
        )
        for name in G2_CONTROL_JOINTS
    }
    robot = RobotCfg(
        name="agibot_g2",
        urdf_path=str(urdf_path),
        mjcf_path=str(mjcf_path),
        num_joints=len(defaults),
        default_position=(0.0, 0.0, 0.0),
        default_joint_positions=defaults,
        joint_limits=limits,
        fix_base_link=True,
        enabled_gravity=False,
        enabled_self_collisions=False,
        actuators=actuators,
        control_type={name: "position" for name in G2_CONTROL_JOINTS},
    )
    robot.file_type.update({"mujoco": "mjcf", "genesis": "mjcf", "newton": "mjcf"})
    return robot


@register_task("icl.geniesim_benchmark")
class GenieSimBenchmarkTask(BaseTaskEnv):
    """Run one imported G1/G2 source run with its original checker as termination.

    The default action is a 20-value position target in ``CONTROL_JOINTS``
    order. Set ``action_mode="tcp_delta"`` to use SynthICL's seven-value
    world-frame TCP delta and gripper-open action. ``bimanual_tcp_delta`` takes
    two such actions in left-then-right order. Observations remain the
    standard MetaSim tensor state. G1 runs expose the existing 20-joint order;
    G2 runs expose the 24 independently controlled arm, waist, head and gripper
    joints from the source robot configuration.
    """

    scenario = None
    supported_simulators = ("mujoco", "genesis", "newton")
    max_episode_steps = 1000

    def __init__(
        self,
        scenario: ScenarioCfg | None = None,
        device=None,
        *,
        run_id: str,
        simulator: str = "mujoco",
        action_mode: str = "joint_position",
        tcp_arm: str | None = None,
        spec_path: Path = SPEC_PATH,
        assets_root: Path = ASSETS_ROOT,
        cache_root: Path = CACHE_ROOT,
        robot_root: Path | None = None,
        robot_variant: str | None = None,
        instance_index: int = 0,
        problem_index: int = 0,
        image_size: int = 640,
        headless: bool = True,
        cameras: tuple[BaseCameraCfg, ...] | None = None,
        camera_profile: str | None = None,
        camera_names: tuple[str, ...] | None = None,
        fisheye_projection: str = "calibrated",
        camera_calibrations: Mapping[str, Mapping[str, Any]] | None = None,
        vlm_score_fn: Callable[[Any, Mapping[str, Any]], Mapping[str, float]] | None = None,
        max_episode_steps: int = 1000,
        reset_settle_seconds: float = 2.0,
    ) -> None:
        if not run_id:
            raise ValueError("run_id must identify an imported Genie Sim run")
        if action_mode not in {"joint_position", "tcp_delta", "bimanual_tcp_delta"}:
            raise ValueError("action_mode must be 'joint_position', 'tcp_delta' or 'bimanual_tcp_delta'")
        if tcp_arm is not None and action_mode != "tcp_delta":
            raise ValueError("tcp_arm is only valid for the single-arm tcp_delta action mode")
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if not math.isfinite(float(reset_settle_seconds)) or reset_settle_seconds < 0:
            raise ValueError("reset_settle_seconds must be finite and non-negative")
        run, instance, evaluation = load_run_instance(Path(spec_path), run_id, instance_index)
        source_robot = evaluation.get("robot", {}).get("robot_id")
        if source_robot not in {"G1", "G2"}:
            raise ValueError(f"Unsupported Genie Sim robot {source_robot!r} in run {run_id!r}")
        if robot_variant not in {None, "G1", "G2"}:
            raise ValueError("robot_variant must be G1, G2, or None")
        runtime_robot = robot_variant or source_robot
        if camera_profile not in {None, "hardware", "benchmark"}:
            raise ValueError("camera_profile must be hardware, benchmark, or None")
        if fisheye_projection not in {"calibrated", "pinhole"}:
            raise ValueError("fisheye_projection must be calibrated or pinhole")
        if runtime_robot == "G1" and camera_profile == "benchmark" and cameras is None:
            raise ValueError("G1 benchmark wrist cameras are not implemented; use the head-camera profile")
        if scenario is None:
            if runtime_robot == "G1":
                resolved_robot_root = Path(robot_root) if robot_root is not None else ROBOT_ROOT
                robot = build_geniesim_g1_robot(resolved_robot_root)
                scene_cameras = cameras if cameras is not None else tuple(
                    geniesim_g1_cameras(resolved_robot_root, image_size=image_size, fisheye_projection=fisheye_projection,
                                        camera_names=camera_names)
                )
            else:
                resolved_robot_root = Path(robot_root) if robot_root is not None else G2_ROBOT_ROOT
                robot = build_geniesim_g2_robot(resolved_robot_root)
                profile = camera_profile or "hardware"
                scene_cameras = cameras if cameras is not None else tuple(
                    geniesim_g2_cameras(image_size=image_size, camera_profile=profile, fisheye_projection=fisheye_projection)
                )
            scene_cameras = _select_cameras(scene_cameras, camera_names)
            scene_cameras = _apply_camera_calibrations(scene_cameras, camera_calibrations)
            scenario, run, instance, evaluation = build_run_scenario(
                spec_path=Path(spec_path),
                run_id=run_id,
                simulator=simulator,
                robot=robot,
                assets_root=Path(assets_root),
                cache_root=Path(cache_root),
                instance_index=instance_index,
                cameras=scene_cameras,
                headless=headless,
                allow_robot_substitution=runtime_robot != source_robot,
            )
        elif scenario.simulator != simulator:
            raise ValueError(f"Scenario backend {scenario.simulator!r} does not match requested {simulator!r}")
        elif camera_names is not None or camera_calibrations:
            scenario = deepcopy(scenario)
            scenario.cameras = list(_apply_camera_calibrations(_select_cameras(scenario.cameras, camera_names), camera_calibrations))
        if scenario.num_envs != 1:
            raise ValueError("One Genie Sim checker graph is stateful; run one environment per task instance")
        robot_cfg = scenario.robots[0]
        if not robot_cfg.name.lower().endswith(runtime_robot.lower()):
            raise ValueError(f"Scenario robot {robot_cfg.name!r} does not match requested {runtime_robot!r}")
        self.robot_name = robot_cfg.name
        self.control_joint_names = CONTROL_JOINTS if runtime_robot == "G1" else G2_CONTROL_JOINTS
        missing = set(self.control_joint_names) - set(robot_cfg.default_joint_positions or {})
        if missing:
            raise ValueError(f"Prepared {runtime_robot} robot is missing benchmark control joints: {sorted(missing)}")
        missing_limits = set(self.control_joint_names) - set(robot_cfg.joint_limits or {})
        if missing_limits:
            raise ValueError(
                f"Prepared {runtime_robot} robot is missing benchmark joint limits: {sorted(missing_limits)}"
            )

        self.run_id = run_id
        self.source_robot = source_robot
        self.robot_type = runtime_robot
        self.action_mode = action_mode
        self._internal_target_clipping: dict[str, dict[str, float]] = {}
        self.run = run
        self.instance = instance
        self.evaluation = evaluation
        self.problem_index = problem_index
        self.checker = GenieSimChecker.from_instance(instance, problem_index=problem_index)
        self.tcp_arm = (
            _single_tcp_arm(instance, self.checker.problem_id, problem_index, tcp_arm)
            if action_mode == "tcp_delta"
            else None
        )
        self.object_bounds = geniesim_object_bounds(instance, Path(assets_root))
        from roboverse_pack.tasks.icl.geniesim_scene import (
            geniesim_checker_background_objects,
            geniesim_checker_subframes,
        )

        self.object_subframes = geniesim_checker_subframes(instance, Path(assets_root))
        self.checker_background_objects = geniesim_checker_background_objects(
            instance, evaluation.get("scene", {}).get("scene_usd")
        )
        self.vlm_score_fn = vlm_score_fn
        self._checker_result = CheckerResult(CheckerStatus.RUNNING, self.checker.problem_id, self.checker.instruction)
        self._checker_step = 0
        self.max_episode_steps = max_episode_steps
        self.reset_settle_seconds = float(reset_settle_seconds)
        self._last_reset_settle_steps = 0
        super().__init__(scenario, device)
        if self.handler.env_step_s is None or self.handler.env_step_s <= 0:
            self.close()
            raise RuntimeError("The active MetaSim backend did not report a valid environment step duration")
        self._delta_time = float(self.handler.env_step_s)
        self._object_joint_indices: dict[str, list[int]] = {}
        for obj in self.scenario.objects:
            object_urdf = getattr(obj, "urdf_path", None)
            provenance_path = Path(object_urdf).parent / "provenance.json" if object_urdf else None
            if provenance_path is None or not provenance_path.is_file():
                continue
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            if provenance.get("format") != "roboverse.icl.geniesim_articulated_object.v1":
                continue
            source_order = provenance.get("joint_order", [])
            backend_order = [name.split("/")[-1] for name in self.handler.get_joint_names(obj.name, sort=True)]
            if not source_order:
                raise ValueError(f"Prepared articulation {obj.name!r} has no source joint order")
            missing_object_joints = set(source_order) - set(backend_order)
            if missing_object_joints:
                self.close()
                raise ValueError(
                    f"Backend is missing source joints for articulated object {obj.name!r}: "
                    f"{sorted(missing_object_joints)} (backend order: {backend_order})"
                )
            self._object_joint_indices[obj.name] = [backend_order.index(name) for name in source_order]
        handler_joint_names = [name.split("/")[-1] for name in self.handler.get_joint_names(robot_cfg.name, sort=True)]
        missing_handler_joints = set(self.control_joint_names) - set(handler_joint_names)
        if missing_handler_joints:
            self.close()
            raise ValueError(f"Active backend is missing benchmark control joints: {sorted(missing_handler_joints)}")
        self._control_indices = [handler_joint_names.index(name) for name in self.control_joint_names]
        self._cartesian_controller = None
        self._bimanual_controllers = None
        if action_mode in {"tcp_delta", "bimanual_tcp_delta"}:
            from roboverse_pack.tasks.icl.control import G1CartesianController, G2CartesianController

            try:
                controller_cls = G1CartesianController if runtime_robot == "G1" else G2CartesianController
                if action_mode == "tcp_delta":
                    controller_kwargs = {"arm_side": self.tcp_arm}
                    if runtime_robot == "G1":
                        # Keep the waist available as IK redundancy, matching
                        # the absolute TCP waypoint replay path. Freezing it
                        # can leave source task poses just outside the arm-only
                        # orientation tolerance even when the full robot can
                        # reach them cleanly.
                        controller_kwargs["allow_torso_motion"] = True
                    self._cartesian_controller = controller_cls(
                        robot_cfg, self.control_joint_names, **controller_kwargs
                    )
                else:
                    self._bimanual_controllers = tuple(
                        controller_cls(robot_cfg, self.control_joint_names, arm_side=side) for side in ("left", "right")
                    )
            except Exception:
                self.close()
                raise

    def _get_initial_states(self) -> list[dict]:
        """Use imported source object and robot start states for reset."""
        robot = self.scenario.robots[0]
        return [
            {
                "objects": {
                    obj.name: {
                        "pos": torch.as_tensor(obj.default_position),
                        "rot": torch.as_tensor(obj.default_orientation),
                    }
                    for obj in self.scenario.objects
                },
                "robots": {
                    robot.name: {
                        "pos": torch.as_tensor(robot.default_position),
                        "rot": torch.as_tensor(robot.default_orientation),
                        "dof_pos": dict(robot.default_joint_positions),
                    }
                },
            }
        ]

    def _action_space(self) -> spaces.Box:
        """Return the declared joint-target or SynthICL TCP-delta action range."""
        if self.action_mode in {"tcp_delta", "bimanual_tcp_delta"}:
            low = np.asarray([-0.021] * 3 + [-0.21] * 3 + [0.0], dtype=np.float32)
            high = np.asarray([0.021] * 3 + [0.21] * 3 + [1.0], dtype=np.float32)
            if self.action_mode == "bimanual_tcp_delta":
                low, high = np.tile(low, 2), np.tile(high, 2)
            return spaces.Box(low=low, high=high, dtype=np.float32)
        limits = self.scenario.robots[0].joint_limits
        low = np.asarray([limits[name][0] for name in self.control_joint_names], dtype=np.float32)
        high = np.asarray([limits[name][1] for name in self.control_joint_names], dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def position_action(self, values: Any) -> list[dict[str, dict[str, dict[str, float]]]]:
        """Build a MetaSim named-joint action from one robot control vector."""
        command = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
        if command.shape != (len(self.control_joint_names),) or not torch.isfinite(command).all():
            raise ValueError(f"Expected {len(self.control_joint_names)} finite joint position targets")
        limits = self.scenario.robots[0].joint_limits
        targets = {}
        for name, value in zip(self.control_joint_names, command.tolist(), strict=True):
            low, high = limits[name]
            # State tensors are commonly float32, so a value at a double-
            # precision limit can round a few ulps outside it on round-trip.
            if value < low - 1e-6 or value > high + 1e-6:
                raise ValueError(f"Position target for {name!r} is outside its joint limits")
            targets[name] = min(max(value, low), high)
        return [{self.robot_name: {"dof_pos_target": targets}}]

    def _process_action(self, actions):
        """Map the declared action vector to the backend-independent named targets."""
        if self.action_mode in {"tcp_delta", "bimanual_tcp_delta"}:
            states = self.handler.get_states(mode="tensor")
            current_controls = (
                states
                .robots[self.robot_name]
                .joint_pos[0, self._control_indices]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            cartesian_action = torch.as_tensor(actions, dtype=torch.float64).reshape(-1).detach().cpu().numpy()
            if self.action_mode == "tcp_delta":
                joint_targets = self._cartesian_controller.joint_targets(current_controls, cartesian_action)
            else:
                if cartesian_action.shape != (14,) or not np.isfinite(cartesian_action).all():
                    raise ValueError("Bimanual Cartesian action must contain 14 finite values")
                joint_targets = current_controls.copy()
                for index, controller in enumerate(self._bimanual_controllers):
                    targets = controller.joint_targets(current_controls, cartesian_action[index * 7 : (index + 1) * 7])
                    controlled = [
                        *controller.arm_input_indices,
                        self.control_joint_names.index(controller.gripper_name),
                    ]
                    joint_targets[controlled] = targets[controlled]
            # Untouched joints retain their measured position. Physics can
            # transiently overshoot a joint limit, in which case copying that
            # measurement verbatim would make an otherwise valid Cartesian
            # command fail position_action(). Record the correction: it is a
            # target constraint, not a claim that the measured state is valid.
            self._internal_target_clipping = {}
            limits = self.scenario.robots[0].joint_limits
            for index, name in enumerate(self.control_joint_names):
                value = float(joint_targets[index])
                low, high = limits[name]
                bounded = float(np.clip(value, low, high))
                if np.isfinite(value) and bounded != value:
                    self._internal_target_clipping[name] = {
                        "unclipped_target": value,
                        "applied_target": bounded,
                        "measured_position": float(current_controls[index]),
                    }
                    joint_targets[index] = bounded
            return self.position_action(joint_targets)
        self._internal_target_clipping = {}
        return self.position_action(actions)

    def _observe_checker(self, states, *, action_step: int, completed_sim_steps: int, delta_time: float) -> None:
        vlm_scores = self.vlm_score_fn(states, self.instance) if self.vlm_score_fn is not None else None
        frame = checker_frame_from_states(
            states,
            robot_name=self.robot_name,
            step=action_step,
            object_bounds=self.object_bounds,
            object_joint_indices=self._object_joint_indices,
            object_subframes=self.object_subframes,
            static_objects=self.checker_background_objects,
            gripper_links=(
                {"left_gripper": "gripper_l_base_link", "right_gripper": "gripper_r_base_link"}
                if self.robot_type == "G2"
                else None
            ),
            gripper_link_transforms=(
                {
                    "left_gripper": G2_GRIPPER_CENTER_FROM_BASE,
                    "right_gripper": G2_GRIPPER_CENTER_FROM_BASE,
                }
                if self.robot_type == "G2"
                else None
            ),
            delta_time=delta_time,
            vlm_scores=vlm_scores,
        )
        self._checker_result = self.checker.step(frame)
        self._checker_step = action_step
        self._checker_sim_step = completed_sim_steps

    def _reward(self, env_states) -> torch.Tensor:
        """Give a sparse one-time success reward."""
        return torch.tensor(
            [float(self._checker_result.status == CheckerStatus.SUCCESS)], dtype=torch.float32, device=self.device
        )

    def _observation(self, env_states):
        """Return calibrated RGB sensor images while keeping backend state tensors immutable."""
        distorted = {
            camera.name: camera
            for camera in self.scenario.cameras
            if isinstance(camera, PinholeCameraCfg) and camera.opencv_distortion is not None
        }
        if not distorted or not env_states.cameras:
            return env_states
        observations = copy(env_states)
        observations.cameras = dict(env_states.cameras)
        for name, camera_cfg in distorted.items():
            camera_state = observations.cameras.get(name)
            if camera_state is None or camera_state.rgb is None:
                continue
            corrected_state = copy(camera_state)
            corrected_state.rgb = apply_opencv_pinhole_distortion(camera_state.rgb, camera_cfg)
            distortion_mask = opencv_pinhole_valid_mask(camera_cfg, device=camera_state.rgb.device)
            distortion_mask = distortion_mask.unsqueeze(0).expand(camera_state.rgb.shape[0], -1, -1)
            corrected_state.valid_mask = (
                distortion_mask
                if camera_state.valid_mask is None
                else camera_state.valid_mask & distortion_mask
            )
            observations.cameras[name] = corrected_state
        return observations

    def _terminated(self, env_states) -> torch.Tensor:
        """End when the imported graph reaches success or a source failure state."""
        done = self._checker_result.status in {CheckerStatus.SUCCESS, CheckerStatus.FAILURE}
        return torch.tensor([done], dtype=torch.bool, device=self.device)

    def _time_out(self, env_states) -> torch.Tensor:
        return super()._time_out(env_states)

    def _prepare_callbacks(self) -> None:
        super()._prepare_callbacks()
        self.reset_callback.append(self._reset_checker)

    def _reset_checker(self, env_ids: list[int]) -> None:
        if 0 in env_ids:
            self._internal_target_clipping = {}
            self.checker.reset()
            self._checker_step = 0
            self._checker_sim_step = 0
            self._checker_result = CheckerResult(
                CheckerStatus.RUNNING, self.checker.problem_id, self.checker.instruction
            )

    def _settle_reset_state(self, observation):
        """Advance source-like settling while holding the reset joint pose."""
        steps = math.ceil(self.reset_settle_seconds / self._delta_time)
        if steps <= 0:
            return observation
        targets = observation.robots[self.robot_name].joint_pos[0, self._control_indices]
        self.handler.set_dof_targets(self.position_action(targets))
        for _ in range(steps):
            self.handler.simulate()
        self._last_reset_settle_steps = steps
        return self._observation(self.handler.get_states(mode="tensor"))

    def reset(self, states=None, env_ids: list[int] | None = None, seed: int | None = None):
        """Reset, reproduce GenieSim's scene settling, then initialize the checker."""
        observation, info = super().reset(states=states, env_ids=env_ids, seed=seed)
        selected_envs = list(range(self.handler.num_envs)) if env_ids is None else env_ids
        self._last_reset_settle_steps = 0
        if states is None and 0 in selected_envs and self.reset_settle_seconds > 0:
            observation = self._settle_reset_state(observation)
        if info is not None:
            info["geniesim_reset_settle_seconds"] = self._last_reset_settle_steps * self._delta_time
            info["geniesim_reset_settle_steps"] = self._last_reset_settle_steps
        if env_ids is None or 0 in env_ids:
            self._observe_checker(observation, action_step=0, completed_sim_steps=0, delta_time=0.0)
        return observation, info

    def step(self, actions):
        """Step MetaSim and sample the source checker before actions 30, 60, ... .

        Source PiEnv and VecEnv update their checker every 30 action calls.
        They use wall-clock deltas; deterministic simulation instead advances
        timed rules by the elapsed simulated action time between samples.
        """
        action_step = int(self._episode_steps[0].item()) + 1
        if action_step % 30 == 0:
            states = self.handler.get_states(mode="tensor")
            self._observe_checker(
                states,
                action_step=action_step,
                completed_sim_steps=action_step - 1,
                delta_time=(action_step - 1 - getattr(self, "_checker_sim_step", 0)) * self._delta_time,
            )
        observation, reward, terminated, timeout, info = super().step(actions)
        if info is None:
            raise RuntimeError("MetaSim task step did not return an info mapping")
        info.update({
            "geniesim_run_id": self.run_id,
            "geniesim_checker_action_step": self._checker_step,
            "geniesim_checker_completed_sim_steps": getattr(self, "_checker_sim_step", 0),
            "geniesim_checker_time_source": "simulated_elapsed",
            "geniesim_source_robot": self.source_robot,
            "geniesim_runtime_robot": self.robot_type,
            "geniesim_robot_adaptation": self.run.get("robot_adaptation"),
            "geniesim_internal_target_clipping": self._internal_target_clipping.copy(),
            "geniesim_reset_pose_validation": self.run.get("reset_pose_validation"),
            "geniesim_problem_id": self._checker_result.problem_id,
            "geniesim_instruction": self._checker_result.instruction,
            "geniesim_checker_status": self._checker_result.status.value,
            "geniesim_action_spec": (
                BIMANUAL_CARTESIAN_ACTION_SPEC
                if self.action_mode == "bimanual_tcp_delta"
                else CARTESIAN_ACTION_SPEC
                if self.action_mode == "tcp_delta"
                else G2_JOINT_POSITION_ACTION_SPEC
                if self.robot_type == "G2"
                else JOINT_POSITION_ACTION_SPEC
            ),
            "geniesim_tcp_arm": (
                self.tcp_arm
                if self.action_mode == "tcp_delta"
                else "both"
                if self.action_mode == "bimanual_tcp_delta"
                else None
            ),
            "task_failure": torch.tensor(
                [self._checker_result.status == CheckerStatus.FAILURE], dtype=torch.bool, device=self.device
            ),
        })
        return observation, reward, terminated, timeout, info
