"""Record source-camera scene probes and action replays for Genie Sim benchmark tasks.

The default probe holds the imported start pose. A recorded action sequence can
also be replayed with its original action period, preserving checker outcomes
and exceptions; rendering by itself does not establish task or physics parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[2]
from tools.icl.import_geniesim_specs import source_content_sha256
from tools.icl.video_utils import RGBVideoWriter

CATALOG_PATH = ROOT / "roboverse_pack/tasks/icl/benchmark_catalog.json"
SPEC_PATH = ROOT / "roboverse_data/icl/geniesim_benchmark_specs_source_scene_v2/benchmark.json"
ASSET_ROOT = ROOT / "roboverse_data/icl/objects/geniesim"
G1_ROBOT_ROOT = ROOT / "roboverse_data/icl/robots/geniesim_g1_omnipicker_closed_v11"
G2_ROBOT_ROOT = ROOT / "roboverse_data/icl/robots/geniesim_g2_omnipicker_v11"
BACKENDS = ("mujoco", "genesis", "newton")
_ACTION_SEQUENCE_FORMAT = "roboverse.icl.benchmark_action_sequence.v1"
_PROFILE_CAMERAS = {
    "hardware": ("head_rgb", "head_fisheye_left", "head_fisheye_right"),
    "benchmark": ("head_rgb", "hand_left", "hand_right"),
}


def _selected_camera_names(profile: str, names: tuple[str, ...] | None) -> tuple[str, ...]:
    """Validate a nonempty, ordered camera selection for the requested profile."""
    if profile not in _PROFILE_CAMERAS:
        raise ValueError(f"Unknown camera profile: {profile!r}")
    selected = _PROFILE_CAMERAS[profile] if names is None else tuple(names)
    if not selected or any(not isinstance(name, str) or not name for name in selected):
        raise ValueError("camera names must be nonempty")
    if len(set(selected)) != len(selected):
        raise ValueError("camera names must not contain duplicates")
    unknown = set(selected) - set(_PROFILE_CAMERAS[profile])
    if unknown:
        raise ValueError(f"Camera names unavailable for {profile!r} profile: {sorted(unknown)}")
    return selected


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _camera_config_info(path: Path | None, camera_names: tuple[str, ...]) -> dict | None:
    """Capture an exact, self-contained calibration file identity for provenance."""
    if path is None:
        return None
    path = Path(path).resolve()
    raw = path.read_bytes()

    def reject_nonfinite(value):
        raise ValueError(f"Non-finite JSON number in camera config: {value}")

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key in camera config: {key}")
            result[key] = value
        return result

    def validate_finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Non-finite JSON number in camera config")
        if isinstance(value, dict):
            for item in value.values():
                validate_finite(item)
        elif isinstance(value, list):
            for item in value:
                validate_finite(item)

    payload = json.loads(raw, parse_constant=reject_nonfinite, object_pairs_hook=unique_pairs)
    validate_finite(payload)
    if not isinstance(payload, dict) or set(payload) != {"format", "cameras"}:
        raise ValueError("Camera config must contain only 'format' and 'cameras'")
    if payload["format"] != "roboverse.icl.geniesim_camera_calibration.v1":
        raise ValueError("Unsupported camera config format")
    cameras = payload["cameras"]
    if not isinstance(cameras, dict) or not cameras or not all(isinstance(item, dict) for item in cameras.values()):
        raise ValueError("Camera config cameras must be a nonempty object of calibration objects")
    if set(cameras) - set(camera_names):
        raise ValueError(f"Camera config includes unselected cameras: {sorted(set(cameras) - set(camera_names))}")
    from metasim.scenario.cameras import FisheyeCameraCfg
    from metasim.utils.fisheye import validate_fisheye_domain

    mandatory = {"projection_model", "calibration_size", "optical_center", "coefficients", "max_fov_rad"}
    allowed = mandatory | {"focal_length_px", "calibration_source", "calibration_source_sha256", "max_fov_source"}
    for name, camera in cameras.items():
        if set(camera) - allowed or mandatory - set(camera):
            raise ValueError(f"Invalid calibration fields for {name!r}: missing {sorted(mandatory - set(camera))}, "
                             f"unknown {sorted(set(camera) - allowed)}")
        source = camera.get("calibration_source")
        if source is not None and (not isinstance(source, str) or not source.strip()):
            raise ValueError(f"calibration_source for {name!r} must be nonempty text")
        for field in ("calibration_size", "optical_center", "coefficients", "focal_length_px"):
            if field not in camera:
                continue
            values = camera[field]
            if not isinstance(values, list) or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
                raise ValueError(f"{field} for {name!r} must be a numeric array")
        fov = camera["max_fov_rad"]
        if isinstance(fov, bool) or not isinstance(fov, (int, float)):
            raise ValueError(f"max_fov_rad for {name!r} must be numeric")
        values = {key: value for key, value in camera.items() if key != "calibration_source"}
        validate_fisheye_domain(FisheyeCameraCfg(name=name, **values))
    return {"path": str(path), "sha256": _digest(raw), "content": payload}


def _robot_asset_info(robot_type: str, robot_root: Path | None) -> dict:
    """Identify the selected portable robot by its manifest and model hashes."""
    if robot_type not in {"g1_op", "g2_op"}:
        raise ValueError(f"Unsupported source robot {robot_type!r}")
    root = Path(robot_root or (G1_ROBOT_ROOT if robot_type == "g1_op" else G2_ROBOT_ROOT)).resolve()
    stem = "G1_omnipicker" if robot_type == "g1_op" else "G2_crs_omnipicker"
    files = {"provenance": root / "provenance.json", "urdf": root / f"{stem}.urdf", "mjcf": root / f"{stem}.xml"}
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(f"Prepared robot asset is incomplete: {path}")
    return {"root": str(root), "model_version": root.name,
            "files_sha256": {name: _digest(path.read_bytes()) for name, path in files.items()}}


def _robot_variants(run: dict, requested: str | None) -> tuple[str, str]:
    """Resolve source and runtime robots without changing the source task identity."""
    try:
        source = {"g1_op": "G1", "g2_op": "G2"}[run["robot"]]
    except KeyError as exc:
        raise ValueError("Source run has no supported robot variant") from exc
    if requested not in {None, "G1", "G2"}:
        raise ValueError("robot_variant must be G1, G2, or None")
    return source, requested or source


def _camera_protocol(task) -> list[dict]:
    """Record the camera models and mounts actually requested by the task."""
    from metasim.scenario.cameras import FisheyeCameraCfg
    from metasim.utils.fisheye import fisheye_rays
    from metasim.utils.pinhole import opencv_pinhole_valid_mask

    cameras = []
    for camera in task.scenario.cameras:
        pixel_validity = None
        if isinstance(camera, FisheyeCameraCfg):
            _, valid = fisheye_rays(camera)
            pixel_validity = {
                "basis": "calibrated_ray_domain",
                "valid_pixels": int(np.count_nonzero(valid)),
                "total_pixels": int(valid.size),
                "valid_fraction": float(np.mean(valid)),
            }
        elif getattr(camera, "opencv_distortion", None) is not None:
            valid = opencv_pinhole_valid_mask(camera).numpy()
            pixel_validity = {
                "basis": "opencv_pinhole_inverse_map",
                "valid_pixels": int(np.count_nonzero(valid)),
                "total_pixels": int(valid.size),
                "valid_fraction": float(np.mean(valid)),
            }
        cameras.append({
            "name": camera.name,
            "projection": type(camera).__name__,
            "width": camera.width,
            "height": camera.height,
            "mount_to": getattr(camera, "mount_to", None),
            "mount_link": getattr(camera, "mount_link", None),
            "mount_pos": getattr(camera, "mount_pos", None),
            "mount_quat": getattr(camera, "mount_quat", None),
            "focal_length": getattr(camera, "focal_length", None),
            "horizontal_aperture": getattr(camera, "horizontal_aperture", None),
            "intrinsics": getattr(camera, "intrinsics", None),
            "opencv_distortion": getattr(camera, "opencv_distortion", None),
            "distortion_source": getattr(camera, "distortion_source", None),
            "projection_model": getattr(camera, "projection_model", None),
            "calibration_size": getattr(camera, "calibration_size", None),
            "optical_center": getattr(camera, "optical_center", None),
            "focal_length_px": getattr(camera, "focal_length_px", None),
            "coefficients": getattr(camera, "coefficients", None),
            "max_fov_rad": getattr(camera, "max_fov_rad", None),
            "calibration_source": getattr(camera, "calibration_source", None),
            "calibration_source_sha256": getattr(camera, "calibration_source_sha256", None),
            "max_fov_source": getattr(camera, "max_fov_source", None),
            "geometric_pixel_validity": pixel_validity,
        })
    return cameras


def _camera_observation_protocol(observation) -> list[dict]:
    """Capture the camera tensors and calibration fields returned by the task."""
    protocol = []
    for name, camera in observation.cameras.items():
        rgb = camera.rgb
        valid_mask = camera.valid_mask
        entry = {
            "name": name,
            "rgb_shape": list(rgb.shape[1:]) if rgb is not None else None,
            "rgb_dtype": str(rgb.dtype) if rgb is not None else None,
            "intrinsics": camera.intrinsics[0].detach().cpu().tolist() if camera.intrinsics is not None else None,
            "position": camera.pos[0].detach().cpu().tolist() if camera.pos is not None else None,
            "quat_world": camera.quat_world[0].detach().cpu().tolist() if camera.quat_world is not None else None,
            "valid_pixels": int(valid_mask[0].sum().item()) if valid_mask is not None else None,
            "valid_fraction": float(valid_mask[0].float().mean().item()) if valid_mask is not None else None,
        }
        protocol.append(entry)
    return protocol


def _physics_protocol(task) -> dict:
    """Record solver settings that affect the actually configured task."""
    params = getattr(task.scenario, "sim_params", None)
    return {
        "physics_dt": getattr(task.handler, "physics_dt", None),
        "environment_step_seconds": getattr(task.handler, "env_step_s", None),
        "reset_settle_seconds": getattr(task, "reset_settle_seconds", 0.0),
        "decimation": getattr(task.scenario, "decimation", None),
        "mujoco_integrator": getattr(params, "mujoco_integrator", None),
        "mujoco_actuator_limits_at_joint": getattr(params, "mujoco_actuator_limits_at_joint", None),
        "newton_mujoco_enable_multiccd": getattr(params, "newton_mujoco_enable_multiccd", None),
    }


def _runtime_file_hashes(simulator: str) -> dict[str, str]:
    files = [
        Path(__file__).resolve(),
        ROOT / "roboverse_pack/tasks/icl/control.py",
        ROOT / "roboverse_pack/tasks/icl/geniesim_benchmark.py",
        ROOT / "roboverse_pack/tasks/icl/geniesim_checker.py",
        ROOT / "roboverse_pack/tasks/icl/geniesim_scene.py",
        ROOT / "roboverse_pack/tasks/icl/fruit_sorting.py",
        ROOT / "packages/metasim/metasim/scenario/simulator_params.py",
        ROOT / "packages/metasim/metasim/scenario/cameras.py",
        ROOT / "packages/metasim/metasim/utils/fisheye.py",
        ROOT / "packages/metasim/metasim/utils/pinhole.py",
        ROOT / "packages/metasim/metasim/sim/base.py",
        ROOT / "packages/metasim/metasim/task/base.py",
        ROOT / f"packages/metasim/metasim/sim/{simulator}/{simulator}.py",
    ]
    compat = ROOT / f"packages/metasim/metasim/sim/{simulator}/_{simulator}_compat.py"
    if compat.is_file():
        files.append(compat)
    return {str(path.relative_to(ROOT)): _digest(path.read_bytes()) for path in files}


def _decode_video(path: Path) -> dict:
    """Decode every frame and report encoded geometry without hiding failures."""
    import av

    result = {"frames": 0, "width": None, "height": None, "error": None}
    try:
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                size = (frame.width, frame.height)
                if result["width"] is not None and size != (result["width"], result["height"]):
                    raise ValueError("Video frame dimensions changed during decoding")
                result["width"], result["height"] = size
                result["frames"] += 1
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def load_action_sequence(path: Path, *, run_id: str, instance_index: int, problem_index: int) -> dict:
    """Read an explicit, finite action sequence with task identity and cadence."""
    sequence = json.loads(Path(path).read_text(encoding="utf-8"))
    if sequence.get("format") != _ACTION_SEQUENCE_FORMAT:
        raise ValueError("Unsupported benchmark action sequence format")
    if (sequence.get("run_id"), sequence.get("instance_index"), sequence.get("problem_index")) != (
        run_id,
        instance_index,
        problem_index,
    ):
        raise ValueError("Action sequence run, instance or problem does not match the requested task")
    if sequence.get("action_mode") not in {"joint_position", "tcp_delta", "bimanual_tcp_delta", "tcp_waypoint"}:
        raise ValueError("Unsupported benchmark action mode")
    if sequence.get("robot_variant") not in {None, "G1", "G2"}:
        raise ValueError("Action sequence robot_variant must be G1 or G2")
    tcp_arm = sequence.get("tcp_arm")
    if tcp_arm is not None and (sequence["action_mode"] not in {"tcp_delta", "tcp_waypoint"} or tcp_arm not in {"left", "right"}):
        raise ValueError("tcp_arm is only valid for single-arm Cartesian actions and must be left or right")
    fingerprint = sequence.get("benchmark_source_content_sha256")
    if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError("Action sequence requires a benchmark_source_content_sha256 fingerprint")
    period = sequence.get("step_seconds")
    if not isinstance(period, (int, float)) or isinstance(period, bool) or not math.isfinite(period) or period <= 0:
        raise ValueError("Action sequence step_seconds must be positive and finite")
    if not isinstance(sequence.get("source"), dict) or not sequence["source"]:
        raise ValueError("Action sequence source metadata is required")
    actions = sequence.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("Action sequence must contain at least one action")
    for index, action in enumerate(actions):
        if (
            not isinstance(action, list)
            or not action
            or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                for value in action
            )
        ):
            raise ValueError(f"Action {index} must be a nonempty list of finite numbers")
    return sequence


def _tcp_waypoint_subtarget_pose(
    controller,
    controls: np.ndarray,
    waypoint,
    *,
    max_translation_step: float = 0.008,
    max_rotation_step: float = 0.1,
) -> tuple[np.ndarray, bool]:
    """Create one bounded, latched TCP subtarget on the way to an absolute waypoint.

    A waypoint is ``[x, y, z, rx, ry, rz, gripper]`` where the first three
    values are an absolute world position, the rotation vector is an absolute
    world orientation, and gripper is in the controller's 0-closed / 1-open
    convention. The returned boolean is true when this subtarget is the final
    waypoint. The recorder holds each subtarget until the measured TCP reaches
    it before creating the next one, so actuator lag cannot make the target
    move away faster than the robot can follow.
    """
    from scipy.spatial.transform import Rotation

    target = np.asarray(waypoint, dtype=np.float64)
    if target.shape != (7,) or not np.isfinite(target).all():
        raise ValueError("TCP waypoint must contain seven finite values")
    if not math.isfinite(max_translation_step) or max_translation_step <= 0:
        raise ValueError("max_translation_step must be positive and finite")
    if not math.isfinite(max_rotation_step) or max_rotation_step <= 0:
        raise ValueError("max_rotation_step must be positive and finite")
    current_position, current_rotation = controller.tcp_pose(controls)
    position_delta = target[:3] - current_position
    distance = float(np.linalg.norm(position_delta))
    position_fraction = min(1.0, max_translation_step / distance) if distance > 0 else 1.0
    subtarget_position = current_position + position_fraction * position_delta
    target_rotation = Rotation.from_rotvec(target[3:6])
    rotation_delta = (target_rotation * current_rotation.inv()).as_rotvec()
    rotation_distance = float(np.linalg.norm(rotation_delta))
    rotation_fraction = min(1.0, max_rotation_step / rotation_distance) if rotation_distance > 0 else 1.0
    subtarget_rotation = Rotation.from_rotvec(rotation_fraction * rotation_delta) * current_rotation
    is_final = position_fraction == 1.0 and rotation_fraction == 1.0
    return np.r_[subtarget_position, subtarget_rotation.as_rotvec()], is_final


def _tcp_waypoint_joint_targets(
    controller,
    controls: np.ndarray,
    waypoint,
    *,
    max_translation_step: float = 0.008,
    max_rotation_step: float = 0.1,
    target_pose: np.ndarray | None = None,
):
    """Convert an absolute world TCP waypoint or latched subtarget to joint targets.

    A latched subtarget remains fixed until reached, but a slow robot may lag far
    behind it. Bound the command from the measured TCP pose on every physics
    step so the controller's Cartesian-delta guard remains valid without moving
    the latched target away from the robot.
    """
    from scipy.spatial.transform import Rotation

    target = np.asarray(waypoint, dtype=np.float64)
    if target.shape != (7,) or not np.isfinite(target).all():
        raise ValueError("TCP waypoint must contain seven finite values")
    current_position, current_rotation = controller.tcp_pose(controls)
    if target_pose is None:
        target_pose, _ = _tcp_waypoint_subtarget_pose(
            controller, controls, target, max_translation_step=max_translation_step
        )
    else:
        target_pose = np.asarray(target_pose, dtype=np.float64)
        if target_pose.shape != (6,) or not np.isfinite(target_pose).all():
            raise ValueError("Latched TCP target pose must contain six finite values")
    step_pose, _ = _tcp_waypoint_subtarget_pose(
        controller,
        controls,
        np.r_[target_pose, target[6]],
        max_translation_step=max_translation_step,
        max_rotation_step=max_rotation_step,
    )
    position_delta = step_pose[:3] - current_position
    rotation_delta = (Rotation.from_rotvec(step_pose[3:6]) * current_rotation.inv()).as_rotvec()
    return controller.joint_targets(controls, np.r_[position_delta, rotation_delta, target[6]])


def record_action_replay(
    *,
    simulator: str,
    run_id: str,
    output: Path,
    actions_path: Path,
    instance_index: int = 0,
    problem_index: int = 0,
    image_size: int = 640,
    fps: int = 30,
    camera_profile: str = "hardware",
    camera_names: tuple[str, ...] | None = None,
    camera_config: Path | None = None,
    robot_root: Path | None = None,
    robot_variant: str | None = None,
) -> dict:
    """Replay a fixed action stream, retaining video and diagnostics on failure."""
    if simulator not in BACKENDS or image_size < 64 or fps < 1:
        raise ValueError("Choose a supported simulator, image_size >= 64 and fps >= 1")
    camera_names = _selected_camera_names(camera_profile, camera_names)
    calibration = _camera_config_info(camera_config, camera_names)
    action_bytes = Path(actions_path).read_bytes()
    sequence = load_action_sequence(
        actions_path, run_id=run_id, instance_index=instance_index, problem_index=problem_index
    )
    if json.loads(action_bytes) != sequence:
        raise ValueError("Action file changed while it was being loaded")
    if not math.isclose(1.0 / fps, sequence["step_seconds"], rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("Video fps must match the action period to preserve playback timing")
    final_output = Path(output).resolve()
    if final_output.exists():
        raise FileExistsError(f"Choose an output directory without prior videos: {final_output}")
    if not SPEC_PATH.is_file():
        raise FileNotFoundError(f"Prepare imported Genie Sim task specs first: {SPEC_PATH}")
    spec_bytes = SPEC_PATH.read_bytes()
    pack = json.loads(spec_bytes)
    source_metadata = pack.get("source", {})
    source_fingerprint = source_metadata.get("content_sha256")
    if not source_fingerprint or source_fingerprint != source_content_sha256(pack):
        raise ValueError("Benchmark source fingerprint is missing or invalid")
    if sequence["benchmark_source_content_sha256"] != source_fingerprint:
        raise ValueError("Action sequence was recorded for a different benchmark source snapshot")
    runtime_hashes = _runtime_file_hashes(simulator)

    from roboverse_pack.tasks.icl.geniesim_benchmark import GenieSimBenchmarkTask
    from roboverse_pack.tasks.icl.geniesim_scene import load_run_instance

    run, instance, _ = load_run_instance(SPEC_PATH, run_id, instance_index)
    source_robot, runtime_robot = _robot_variants(run, robot_variant)
    action_robot = sequence.get("robot_variant") or source_robot
    if action_robot != runtime_robot:
        raise ValueError(f"Action sequence is for {action_robot}, but runtime robot is {runtime_robot}")
    robot_asset = _robot_asset_info("g1_op" if runtime_robot == "G1" else "g2_op", robot_root)
    expected_robot_hashes = sequence.get("robot_asset_sha256")
    if expected_robot_hashes is not None and expected_robot_hashes != robot_asset["files_sha256"]:
        raise ValueError("Action sequence was recorded for a different robot asset")
    expected_dims = {
        "joint_position": 20 if runtime_robot == "G1" else 24,
        "tcp_delta": 7,
        "bimanual_tcp_delta": 14,
        "tcp_waypoint": 7,
    }
    dim = expected_dims[sequence["action_mode"]]
    if any(len(action) != dim for action in sequence["actions"]):
        raise ValueError(f"The action sequence requires {dim} values per action")
    task_action_mode = "joint_position" if sequence["action_mode"] == "tcp_waypoint" else sequence["action_mode"]
    task = GenieSimBenchmarkTask(
        run_id=run_id,
        simulator=simulator,
        action_mode=task_action_mode,
        tcp_arm=sequence.get("tcp_arm") if task_action_mode == "tcp_delta" else None,
        spec_path=SPEC_PATH,
        robot_root=Path(robot_asset["root"]),
        robot_variant=robot_variant,
        assets_root=ASSET_ROOT,
        instance_index=instance_index,
        problem_index=problem_index,
        image_size=image_size,
        max_episode_steps=4096 if sequence["action_mode"] == "tcp_waypoint" else len(sequence["actions"]) + 1,
        camera_profile=camera_profile,
        camera_names=camera_names,
        **({"camera_calibrations": calibration["content"]["cameras"]} if calibration else {}),
    )
    writers: dict[str, RGBVideoWriter] = {}
    final_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix="rvreplay_", dir=final_output.parent)
    staging = Path(temporary.name) / "record"
    staging.mkdir()
    records = []
    frame_shapes: dict[str, list[int]] = {}
    try:
        if not math.isclose(task.handler.env_step_s, sequence["step_seconds"], rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("Action sequence period does not match the backend's actual environment step")
        observation, reset_info = task.reset()
        if set(observation.cameras) != set(camera_names):
            raise ValueError("Observed cameras do not match the requested camera names")

        def capture(state) -> None:
            if set(state.cameras) != set(camera_names):
                raise ValueError("Observed cameras changed during action replay")
            for name, camera in state.cameras.items():
                rgb = camera.rgb[0].detach().cpu().numpy()
                if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.std() < 1:
                    raise ValueError(f"Invalid or blank RGB frame from {name!r}")
                frame_shapes[name] = list(rgb.shape)
                if name not in writers:
                    writers[name] = RGBVideoWriter(staging / f"{simulator}_{name}.mp4", rgb.shape[1], rgb.shape[0], fps)
                writers[name].append(rgb)

        def object_states(state) -> dict:
            return {name: value.root_state[0, :7].detach().cpu().tolist() for name, value in state.objects.items()}

        initial_objects = object_states(observation)
        capture(observation)
        waypoint_controller = None
        waypoint_protocol = None
        if sequence["action_mode"] == "tcp_waypoint":
            from scipy.spatial.transform import Rotation

            from roboverse_pack.tasks.icl.control import G1CartesianController, G2CartesianController

            controller_class = G1CartesianController if runtime_robot == "G1" else G2CartesianController
            controller_kwargs = {"arm_side": sequence.get("tcp_arm", "right")}
            if runtime_robot == "G1":
                # The source G1 has two actuated waist joints; include them in
                # IK redundancy so extended reaches do not drive the wrist to
                # a hard stop while the torso is artificially frozen.
                controller_kwargs["allow_torso_motion"] = True
            waypoint_controller = controller_class(task.scenario.robots[0], task.control_joint_names, **controller_kwargs)
            waypoint_protocol = {
                "position": "absolute_world_meters",
                "orientation": "absolute_world_rotation_vector_radians",
                "feedback": "measured_tcp_pose_each_environment_step",
                "max_translation_step_m": 0.02,
                "max_rotation_step_rad": 0.2,
                "subtarget_arrival_tolerance_m": 0.002,
                "subtarget_orientation_tolerance_rad": 0.02,
                "subtarget_progression": "hold_until_measured_arrival",
                "max_steps_per_waypoint": 240,
                "ik_joints": list(waypoint_controller.ik_names) if runtime_robot == "G1" else list(
                    waypoint_controller.arm_names
                ),
            }
        outcome, exception = "sequence_exhausted", None
        sim_step_index = 0
        for index, action in enumerate(sequence["actions"]):
            if waypoint_controller is not None:
                subtarget_pose = None
                subtarget_is_final = False
                waypoint_reached = False
                final_position_error = math.inf
                final_orientation_error = math.inf
                for waypoint_step in range(240):
                    waypoint_target_clipping = {}
                    try:
                        current_controls = (
                            observation.robots[task.robot_name]
                            .joint_pos[0, task._control_indices]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float64)
                        )
                        if subtarget_pose is None:
                            subtarget_pose, subtarget_is_final = _tcp_waypoint_subtarget_pose(
                                waypoint_controller,
                                current_controls,
                                action,
                                max_translation_step=0.02,
                                max_rotation_step=0.2,
                            )
                        step_action = _tcp_waypoint_joint_targets(
                            waypoint_controller,
                            current_controls,
                            action,
                            max_translation_step=0.02,
                            max_rotation_step=0.2,
                            target_pose=subtarget_pose,
                        )
                        limits = task.scenario.robots[0].joint_limits
                        for control_index, joint_name in enumerate(task.control_joint_names):
                            value = float(step_action[control_index])
                            lower, upper = limits[joint_name]
                            bounded = float(np.clip(value, lower, upper))
                            if bounded != value:
                                waypoint_target_clipping[joint_name] = {
                                    "unclipped_target": value,
                                    "applied_target": bounded,
                                    "measured_position": float(current_controls[control_index]),
                                }
                                step_action[control_index] = bounded
                        observation, reward, terminated, timeout, info = task.step(step_action)
                        sim_step_index += 1
                        if waypoint_target_clipping:
                            info["geniesim_internal_target_clipping"] = waypoint_target_clipping
                    except Exception as exc:
                        outcome = "action_exception"
                        exception = {
                            "action_index": index,
                            "waypoint_step": waypoint_step,
                            "type": type(exc).__name__,
                            "message": str(exc),
                        }
                        break

                    measured_controls = (
                        observation.robots[task.robot_name]
                        .joint_pos[0, task._control_indices]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
                    measured_position, measured_rotation = waypoint_controller.tcp_pose(measured_controls)
                    target_rotation = Rotation.from_rotvec(subtarget_pose[3:6])
                    final_rotation = Rotation.from_rotvec(action[3:6])
                    final_position_error = float(np.linalg.norm(np.asarray(action[:3]) - measured_position))
                    final_orientation_error = float(np.linalg.norm((final_rotation * measured_rotation.inv()).as_rotvec()))
                    subtarget_position_error = float(np.linalg.norm(subtarget_pose[:3] - measured_position))
                    subtarget_orientation_error = float(
                        np.linalg.norm((target_rotation * measured_rotation.inv()).as_rotvec())
                    )
                    records.append({
                        "action_index": index,
                        "waypoint_step": waypoint_step,
                        "sim_step_index": sim_step_index - 1,
                        "time_seconds": sim_step_index * sequence["step_seconds"],
                        "checker_status": info["geniesim_checker_status"],
                        "internal_target_clipping": info.get("geniesim_internal_target_clipping", {}),
                        "waypoint_target_clipping": waypoint_target_clipping,
                        "subtarget_pose": subtarget_pose.tolist(),
                        "subtarget_is_final": subtarget_is_final,
                        "subtarget_position_error_m": subtarget_position_error,
                        "subtarget_orientation_error_rad": subtarget_orientation_error,
                        "waypoint_position_error_m": final_position_error,
                        "waypoint_orientation_error_rad": final_orientation_error,
                        "reward": float(reward[0]),
                        "objects": object_states(observation),
                        "commanded_joint_positions": step_action.tolist(),
                        "control_joint_positions": measured_controls.tolist(),
                        "joint_position_tracking_error": (step_action - measured_controls).tolist(),
                    })
                    capture(observation)
                    if bool(terminated[0]):
                        outcome = f"checker_{info['geniesim_checker_status']}"
                        break
                    if bool(timeout[0]):
                        outcome = "timeout"
                        break

                    arrived = subtarget_position_error <= 0.002 and subtarget_orientation_error <= 0.02
                    if arrived:
                        if subtarget_is_final:
                            waypoint_reached = True
                            break
                        subtarget_pose = None
                if outcome != "sequence_exhausted":
                    break
                if not waypoint_reached:
                    outcome = "waypoint_timeout"
                    exception = {
                        "action_index": index,
                        "type": "WaypointTimeout",
                        "message": f"TCP waypoint not reached within 240 simulation steps; position error="
                        f"{final_position_error:.4f} m, orientation error={final_orientation_error:.4f} rad",
                    }
                    break
                continue

            waypoint_target_clipping = {}
            try:
                observation, reward, terminated, timeout, info = task.step(action)
                sim_step_index += 1
            except Exception as exc:
                outcome = "action_exception"
                exception = {"action_index": index, "type": type(exc).__name__, "message": str(exc)}
                break
            records.append({
                "action_index": index,
                "sim_step_index": sim_step_index - 1,
                "time_seconds": sim_step_index * sequence["step_seconds"],
                "checker_status": info["geniesim_checker_status"],
                "internal_target_clipping": info.get("geniesim_internal_target_clipping", {}),
                "waypoint_target_clipping": waypoint_target_clipping,
                "reward": float(reward[0]),
                "objects": object_states(observation),
                "control_joint_positions": observation
                .robots[task.robot_name]
                .joint_pos[0, task._control_indices]
                .detach()
                .cpu()
                .tolist(),
            })
            capture(observation)
            if bool(terminated[0]):
                outcome = f"checker_{info['geniesim_checker_status']}"
                break
            if bool(timeout[0]):
                outcome = "timeout"
                break
        for writer in writers.values():
            writer.close()
        writers.clear()
        video_files = sorted(path.name for path in staging.glob("*.mp4"))
        video_decode = {name: _decode_video(staging / name) for name in video_files}
        video_sha256 = {name: _digest((staging / name).read_bytes()) for name in video_files}
        (staging / "actions.json").write_bytes(action_bytes)
        runtime_end_hashes = _runtime_file_hashes(simulator)
        try:
            robot_asset_after = _robot_asset_info("g1_op" if runtime_robot == "G1" else "g2_op", Path(robot_asset["root"]))
        except OSError:
            robot_asset_after = None
        try:
            source_file_hash_after = _digest(SPEC_PATH.read_bytes())
        except OSError:
            source_file_hash_after = None
        try:
            calibration_after = _camera_config_info(camera_config, camera_names)
        except (OSError, ValueError, UnicodeDecodeError):
            calibration_after = None
        report = {
            "format": "roboverse.icl.benchmark_action_replay.v1",
            "run_id": run_id,
            "instance": f"{instance['task']}/{instance['instance_id']}",
            "instance_index": instance_index,
            "problem_index": problem_index,
            "simulator": simulator,
            "source_content_sha256": source_fingerprint,
            "source_robot": source_robot,
            "runtime_robot": runtime_robot,
            "robot_adaptation": task.run.get("robot_adaptation"),
            "action_robot_variant": action_robot,
            "action_robot_variant_inferred_from_source": sequence.get("robot_variant") is None,
            "source_snapshot_file_sha256": _digest(spec_bytes),
            "source_snapshot_changed_during_replay": source_file_hash_after != _digest(spec_bytes),
            "robot_asset": robot_asset,
            "robot_asset_changed_during_replay": robot_asset_after != robot_asset,
            "action_sequence_robot_asset_pinned": expected_robot_hashes is not None,
            "runtime_file_sha256": runtime_hashes,
            "runtime_files_changed_during_replay": runtime_hashes != runtime_end_hashes,
            "runtime_file_sha256_after_replay": runtime_end_hashes,
            "action_source": sequence["source"],
            "actions_sha256": _digest(action_bytes),
            "action_mode": sequence["action_mode"],
            "tcp_arm": sequence.get("tcp_arm"),
            "waypoint_protocol": waypoint_protocol,
            "step_seconds": float(task.handler.env_step_s),
            "reset_settle": {
                "requested_seconds": task.reset_settle_seconds,
                "actual_seconds": reset_info.get("geniesim_reset_settle_seconds", 0.0),
                "simulator_steps": reset_info.get("geniesim_reset_settle_steps", 0),
                "excluded_from_episode_step_count": True,
            },
            "video_fps": fps,
            "camera_profile": camera_profile,
            "camera_names": list(camera_names),
            "camera_config": calibration,
            "camera_config_changed_during_replay": calibration_after != calibration,
            "camera_protocol": _camera_protocol(task),
            "camera_observation_protocol": _camera_observation_protocol(observation),
            "physics_protocol": _physics_protocol(task),
            "camera_image_size_argument": image_size,
            "camera_frame_shapes": frame_shapes,
            "video_encoded_dimensions": {
                name: [shape[1] + shape[1] % 2, shape[0] + shape[0] % 2] for name, shape in frame_shapes.items()
            },
            "video_files": video_files,
            "video_decode": video_decode,
            "video_sha256": video_sha256,
            "initial_objects": initial_objects,
            "steps": records,
            "outcome": outcome,
            "exception": exception,
            "validation_scope": "actual replay and imported checker outcome; not proof of source-checker or cross-backend parity",
        }
        (staging / "action_replay_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        shutil.move(str(staging), str(final_output))
        return report
    finally:
        for writer in writers.values():
            writer.close()
        try:
            task.close()
        finally:
            temporary.cleanup()


def record_one(
    *,
    simulator: str,
    run_id: str,
    output: Path,
    instance_index: int = 0,
    problem_index: int = 0,
    image_size: int = 640,
    frames: int = 45,
    fps: int = 30,
    camera_profile: str = "hardware",
    camera_names: tuple[str, ...] | None = None,
    camera_config: Path | None = None,
    robot_root: Path | None = None,
    robot_variant: str | None = None,
    motion_mode: str = "hold",
) -> dict:
    """Record each configured task camera while holding the source start pose."""
    if simulator not in BACKENDS:
        raise ValueError(f"Unsupported simulator: {simulator}")
    if image_size < 64 or frames < 2 or fps < 1:
        raise ValueError("image_size must be >=64, frames >=2 and fps >=1")
    if motion_mode not in {"hold", "head_pan"}:
        raise ValueError("motion_mode must be hold or head_pan")
    camera_names = _selected_camera_names(camera_profile, camera_names)
    calibration = _camera_config_info(camera_config, camera_names)
    if not SPEC_PATH.is_file():
        raise FileNotFoundError(f"Prepare imported Genie Sim task specs first: {SPEC_PATH}")
    spec_bytes = SPEC_PATH.read_bytes()
    pack = json.loads(spec_bytes)
    source_metadata = pack.get("source", {})
    source_fingerprint = source_metadata.get("content_sha256")
    if not source_fingerprint or source_fingerprint != source_content_sha256(pack):
        raise ValueError("Benchmark source fingerprint is missing or invalid; re-import or correct its provenance")

    from roboverse_pack.tasks.icl.geniesim_benchmark import GenieSimBenchmarkTask
    from roboverse_pack.tasks.icl.geniesim_scene import load_run_instance

    run, instance, _ = load_run_instance(SPEC_PATH, run_id, instance_index)
    source_robot, runtime_robot = _robot_variants(run, robot_variant)
    robot_asset = _robot_asset_info("g1_op" if runtime_robot == "G1" else "g2_op", robot_root)
    runtime_hashes = _runtime_file_hashes(simulator)
    final_output = Path(output).resolve()
    report_path = final_output / "scene_video_report.json"
    if report_path.is_file():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            existing.get("format") == "roboverse.icl.benchmark_video_probe.v1"
            and existing.get("simulator") == simulator
            and existing.get("run_id") == run_id
            and existing.get("source_content_sha256") == source_fingerprint
            and existing.get("robot_asset") == robot_asset
            and existing.get("runtime_file_sha256") == runtime_hashes
            and existing.get("runtime_files_changed_during_recording") is False
            and existing.get("source_snapshot_file_sha256") == _digest(spec_bytes)
            and existing.get("source_snapshot_changed_during_recording") is False
            and existing.get("robot_asset_changed_during_recording") is False
            and existing.get("source_robot_variant") == source_robot
            and existing.get("runtime_robot") == runtime_robot
            and existing.get("instance_index") == instance_index
            and existing.get("problem_index") == problem_index
            and existing.get("camera_profile") == camera_profile
            and existing.get("camera_names") == list(camera_names)
            and existing.get("camera_config") == calibration
            and existing.get("camera_config_changed_during_recording") is False
            and existing.get("motion_mode", "hold") == motion_mode
            and existing.get("capture") == {"fps": fps, "frames": frames, "camera_image_size_argument": image_size}
            and existing.get("video_files")
            and set(existing["video_files"]) == {f"{simulator}_{name}.mp4" for name in camera_names}
            and len(existing["video_files"]) == len(camera_names)
            and all(
                (final_output / name).is_file()
                and _digest((final_output / name).read_bytes()) == existing.get("video_sha256", {}).get(name)
                and existing.get("video_decode", {}).get(name, {}).get("error") is None
                and existing.get("video_decode", {}).get(name, {}).get("frames") == frames
                for name in existing["video_files"]
            )
        ):
            return existing
        raise FileExistsError(f"Output has a different or incomplete prior report: {output}")
    if final_output.exists():
        raise FileExistsError(f"Output directory exists but has no valid report: {final_output}")
    task = GenieSimBenchmarkTask(
        run_id=run_id,
        simulator=simulator,
        spec_path=SPEC_PATH,
        robot_root=Path(robot_asset["root"]),
        robot_variant=robot_variant,
        assets_root=ASSET_ROOT,
        cache_root=ROOT / "roboverse_data/icl/cache/geniesim_scene_wrappers_v6",
        instance_index=instance_index,
        problem_index=problem_index,
        image_size=image_size,
        max_episode_steps=max(frames + 2, 120),
        camera_profile=camera_profile,
        camera_names=camera_names,
        **({"camera_calibrations": calibration["content"]["cameras"]} if calibration else {}),
    )
    writers: dict[str, RGBVideoWriter] = {}
    final_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix="rvvid_", dir=final_output.parent)
    output = Path(temporary.name) / "record"
    output.mkdir()
    report_path = output / "scene_video_report.json"
    initial_objects: dict[str, list[float]] = {}
    final_objects: dict[str, list[float]] = {}
    frame_stats: dict[str, list[float]] = {}
    frame_shapes: dict[str, list[int]] = {}
    statuses: list[str] = []
    timeout_seen = False
    task_failure_seen = False
    head_positions: list[float] = []
    try:
        observation, reset_info = task.reset()
        if set(observation.cameras) != set(camera_names):
            raise ValueError("Observed cameras do not match the requested camera names")
        initial_objects = {
            name: state.root_state[0, :7].detach().cpu().tolist() for name, state in observation.objects.items()
        }
        target = [task.scenario.robots[0].default_joint_positions[name] for name in task.control_joint_names]
        head_joint = "idx11_head_joint1"
        head_index = task.control_joint_names.index(head_joint) if head_joint in task.control_joint_names else None
        if motion_mode == "head_pan" and head_index is None:
            raise ValueError("Head-pan probe requires idx11_head_joint1")
        head_target = None
        if head_index is not None:
            head_limit = task.scenario.robots[0].joint_limits[head_joint]
            direction = 1.0 if head_limit[1] - target[head_index] >= target[head_index] - head_limit[0] else -1.0
            head_target = float(np.clip(target[head_index] + direction * 0.08, *head_limit))
            head_positions.append(float(observation.robots[task.robot_name].joint_pos[0, task._control_indices[head_index]]))
        for index in range(frames):
            if index:
                command = target.copy()
                if motion_mode == "head_pan" and frames // 4 <= index < 3 * frames // 4:
                    command[head_index] = head_target
                observation, _, _, timeout, info = task.step(command)
                if head_index is not None:
                    head_positions.append(float(observation.robots[task.robot_name].joint_pos[0, task._control_indices[head_index]]))
                timeout_seen |= bool(timeout[0])
                task_failure_seen |= bool(info.get("task_failure", [False])[0])
                statuses.append(info.get("geniesim_checker_status", "unknown"))
            if set(observation.cameras) != set(camera_names):
                raise ValueError("Observed cameras changed during recording")
            for name, camera in observation.cameras.items():
                rgb = camera.rgb[0].detach().cpu().numpy()
                if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3 or not np.isfinite(rgb).all():
                    raise ValueError(f"Camera {name!r} returned invalid RGB {rgb.shape} {rgb.dtype}")
                frame_std = float(rgb.std())
                if frame_std < 1.0:
                    raise RuntimeError(f"Camera {name!r} rendered a blank frame (std={frame_std:.3f})")
                frame_stats.setdefault(name, []).append(frame_std)
                frame_shapes[name] = [frames, *rgb.shape]
                if name not in writers:
                    writers[name] = RGBVideoWriter(output / f"{simulator}_{name}.mp4", rgb.shape[1], rgb.shape[0], fps)
                writers[name].append(rgb)
        final_objects = {
            name: state.root_state[0, :7].detach().cpu().tolist() for name, state in observation.objects.items()
        }
        for writer in writers.values():
            writer.close()
        writers.clear()

        video_files = sorted(path.name for path in output.glob("*.mp4"))
        video_decode = {name: _decode_video(output / name) for name in video_files}
        video_sha256 = {name: _digest((output / name).read_bytes()) for name in video_files}
        aliases = []
        for obj in task.scenario.objects:
            urdf_path = getattr(obj, "urdf_path", None)
            manifest_path = Path(urdf_path).parent / "instance.json" if urdf_path else None
            if manifest_path and manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("asset_alias") is not None:
                    aliases.append({"scene_object": obj.name, **manifest["asset_alias"]})
        runtime_end_hashes = _runtime_file_hashes(simulator)
        try:
            robot_asset_after = _robot_asset_info("g1_op" if runtime_robot == "G1" else "g2_op", Path(robot_asset["root"]))
        except OSError:
            robot_asset_after = None
        try:
            source_file_hash_after = _digest(SPEC_PATH.read_bytes())
        except OSError:
            source_file_hash_after = None
        try:
            calibration_after = _camera_config_info(camera_config, camera_names)
        except (OSError, ValueError, UnicodeDecodeError):
            calibration_after = None
        report = {
            "format": "roboverse.icl.benchmark_video_probe.v1",
            "source_revision": source_metadata.get("revision", "unknown"),
            "source_content_sha256": source_fingerprint,
            "source_snapshot_file_sha256": _digest(spec_bytes),
            "source_snapshot_changed_during_recording": source_file_hash_after != _digest(spec_bytes),
            "robot_asset": robot_asset,
            "robot_asset_changed_during_recording": robot_asset_after != robot_asset,
            "runtime_file_sha256": runtime_hashes,
            "runtime_files_changed_during_recording": runtime_hashes != runtime_end_hashes,
            "runtime_file_sha256_after_recording": runtime_end_hashes,
            "run_id": run_id,
            "source_robot": run["robot"],
            "source_robot_variant": source_robot,
            "runtime_robot": runtime_robot,
            "robot_adaptation": task.run.get("robot_adaptation"),
            "camera_protocol": _camera_protocol(task),
            "camera_observation_protocol": _camera_observation_protocol(observation),
            "physics_protocol": _physics_protocol(task),
            "reset_settle": {
                "requested_seconds": task.reset_settle_seconds,
                "actual_seconds": reset_info.get("geniesim_reset_settle_seconds", 0.0),
                "simulator_steps": reset_info.get("geniesim_reset_settle_steps", 0),
                "excluded_from_episode_step_count": True,
            },
            "motion_mode": motion_mode,
            "head_pan": ({
                "joint": head_joint,
                "requested_target": head_target if motion_mode == "head_pan" else target[head_index],
                "observed_min": min(head_positions),
                "observed_max": max(head_positions),
                "max_deviation_from_start": max(abs(value - head_positions[0]) for value in head_positions),
                "finite": all(math.isfinite(value) for value in head_positions),
            } if head_positions else None),
            "source_subtask": run["subtask"],
            "instance": f"{instance['task']}/{instance['instance_id']}",
            "instance_index": instance_index,
            "problem_index": problem_index,
            "simulator": simulator,
            "output_directory": str(final_output),
            "camera_profile": camera_profile,
            "camera_names": list(camera_names),
            "camera_config": calibration,
            "camera_config_changed_during_recording": calibration_after != calibration,
            "capture": {"fps": fps, "frames": frames, "camera_image_size_argument": image_size},
            "camera_frame_shapes": frame_shapes,
            "video_encoded_dimensions": {
                name: [shape[2] + shape[2] % 2, shape[1] + shape[1] % 2] for name, shape in frame_shapes.items()
            },
            "camera_pixel_std_range": {name: [min(values), max(values)] for name, values in frame_stats.items()},
            "video_files": video_files,
            "video_decode": video_decode,
            "video_sha256": video_sha256,
            "compatibility_asset_aliases": aliases,
            "initial_object_root_states": initial_objects,
            "final_object_root_states": final_objects,
            "checker_statuses_observed": sorted(set(statuses)),
            "timeout_seen": timeout_seen,
            "task_failure_seen": task_failure_seen,
            "task_success_claimed": False,
            "validation_scope": "configured start pose / nonblank camera renders / held or head-pan targets; not task completion or parity",
        }
        report["report_payload_sha256"] = _digest(
            json.dumps(report, ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if final_output.exists():
            raise FileExistsError(f"Output directory appeared during recording: {final_output}")
        shutil.move(str(output), str(final_output))
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return report
    finally:
        for writer in writers.values():
            writer.close()
        try:
            task.close()
        finally:
            temporary.cleanup()


def build_unique_scene_plan(
    pack: dict,
    *,
    simulators: tuple[str, ...],
    robot_variants: tuple[str, ...],
    scene_start: int = 0,
    scene_count: int | None = None,
) -> dict:
    """Select one canonical source run per referenced scene and expand the test matrix."""
    if scene_start < 0 or (scene_count is not None and scene_count <= 0):
        raise ValueError("scene_start must be non-negative and scene_count must be positive")
    if not simulators or len(set(simulators)) != len(simulators) or any(sim not in BACKENDS for sim in simulators):
        raise ValueError("simulators must be a nonempty subset without duplicates")
    if not robot_variants or len(set(robot_variants)) != len(robot_variants) or any(
        robot not in {"G1", "G2"} for robot in robot_variants
    ):
        raise ValueError("robot_variants must be a nonempty subset of G1 and G2 without duplicates")
    runs = pack.get("runs")
    instances = pack.get("instances")
    if not isinstance(runs, list) or not isinstance(instances, dict):
        raise ValueError("Benchmark snapshot is missing runs or scene instances")
    canonical = {}
    for run in sorted(runs, key=lambda item: item.get("id", "")):
        run_id = run.get("id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("Benchmark run has no ID")
        for index, instance_key in enumerate(run.get("instances", [])):
            if not isinstance(instance_key, str) or not re.fullmatch(r"[A-Za-z0-9_-]+/[0-9]+", instance_key):
                raise ValueError(f"Invalid source scene key {instance_key!r}")
            if instance_key not in instances:
                raise ValueError(f"Run {run_id!r} references unknown scene {instance_key!r}")
            canonical.setdefault(instance_key, {"run_id": run_id, "instance_index": index})
    scene_keys = sorted(canonical)
    if scene_start >= len(scene_keys):
        raise ValueError("scene_start is outside the available referenced scenes")
    selected = scene_keys[scene_start : None if scene_count is None else scene_start + scene_count]
    jobs = []
    for key in selected:
        for robot in sorted(robot_variants):
            for simulator in sorted(simulators):
                jobs.append({
                    "scene": key,
                    **canonical[key],
                    "problem_index": 0,
                    "robot_variant": robot,
                    "simulator": simulator,
                    "output_relative": f"{key}/{robot}/{simulator}",
                })
    return {
        "format": "roboverse.icl.geniesim_unique_scene_matrix_plan.v1",
        "selection_rule": "lexicographically first source run ID and its original instance index per scene key",
        "referenced_scene_count": len(scene_keys),
        "scene_start": scene_start,
        "scene_count": len(selected),
        "robot_variants": sorted(robot_variants),
        "simulators": sorted(simulators),
        "jobs": jobs,
    }


def _atomic_json(path: Path, value: dict) -> None:
    """Replace one JSON state file atomically without truncating its prior version."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _matrix_report_error(report: dict, output: Path, job: dict, plan: dict) -> str | None:
    """Check one scene probe, its videos and the matrix's frozen provenance."""
    expected_cameras = set(plan["camera_names"])
    payload = {key: value for key, value in report.items() if key != "report_payload_sha256"}
    if report.get("report_payload_sha256") != _digest(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")):
        return "probe report checksum is invalid"
    if (
        report.get("format") != "roboverse.icl.benchmark_video_probe.v1"
        or report.get("run_id") != job["run_id"]
        or report.get("instance_index") != job["instance_index"]
        or report.get("problem_index") != job["problem_index"]
        or report.get("runtime_robot") != job["robot_variant"]
        or report.get("simulator") != job["simulator"]
        or report.get("source_content_sha256") != plan["source_content_sha256"]
        or report.get("source_snapshot_file_sha256") != plan["source_snapshot_file_sha256"]
        or report.get("robot_asset") != plan["robot_assets"].get(job["robot_variant"])
        or report.get("runtime_file_sha256") != plan["runtime_file_sha256"].get(job["simulator"])
        or report.get("camera_profile") != "hardware"
        or report.get("camera_names") != plan["camera_names"]
        or report.get("camera_config") != plan["camera_config"]
        or report.get("motion_mode") != "head_pan"
        or report.get("capture") != plan["capture"]
    ):
        return "report identity, configuration or frozen provenance does not match the matrix plan"
    if any(report.get(key) is not False for key in (
        "runtime_files_changed_during_recording", "source_snapshot_changed_during_recording",
        "robot_asset_changed_during_recording",
        "camera_config_changed_during_recording",
    )):
        return "source, model or runtime files changed during the probe"
    cameras = report.get("camera_protocol", [])
    if not isinstance(cameras, list) or len(cameras) != len(expected_cameras) or not all(isinstance(camera, dict) for camera in cameras) or {
        camera.get("name") for camera in cameras
    } != expected_cameras:
        return "probe did not expose exactly the requested cameras"
    actual_cameras = {camera["name"]: camera for camera in cameras}
    if plan["camera_config"] is not None:
        for name, override in plan["camera_config"]["content"]["cameras"].items():
            actual = actual_cameras[name]
            if actual.get("projection") != "FisheyeCameraCfg" or any(
                actual.get(field) != value for field, value in override.items()
            ):
                return f"camera calibration override was not applied to {name}"
    protocol = report.get("physics_protocol") or {}
    if not isinstance(protocol, dict) or not isinstance(protocol.get("environment_step_seconds"), (int, float)) or not math.isclose(
        protocol["environment_step_seconds"], 1 / plan["capture"]["fps"], rel_tol=0.0, abs_tol=1e-8
    ):
        return "video playback rate does not match the actual environment step"
    if report.get("timeout_seen") or report.get("task_failure_seen"):
        return "probe reported task failure or timeout"
    head = report.get("head_pan") or {}
    if not head.get("finite") or head.get("max_deviation_from_start", 0) < 1e-4:
        return "head-pan command produced no observable finite head movement"
    files = report.get("video_files", [])
    decode, hashes = report.get("video_decode", {}), report.get("video_sha256", {})
    if len(files) != len(expected_cameras) or set(files) != {
        f"{job['simulator']}_{name}.mp4" for name in expected_cameras
    }:
        return "probe did not produce exactly the requested camera videos"
    for name in files:
        path = output / name
        if not path.is_file() or _digest(path.read_bytes()) != hashes.get(name):
            return f"video missing or changed: {name}"
        result = decode.get(name, {})
        if result.get("error") is not None or result.get("frames") != plan["capture"]["frames"]:
            return f"video did not decode the expected number of frames: {name}"
        actual_decode = _decode_video(path)
        if actual_decode != result:
            return f"video decode no longer matches the recorded inspection: {name}"
    return None


def record_unique_scene_matrix(
    *,
    output_root: Path,
    simulators: tuple[str, ...] = BACKENDS,
    robot_variants: tuple[str, ...] = ("G1", "G2"),
    scene_start: int = 0,
    scene_count: int | None = None,
    image_size: int = 256,
    frames: int = 30,
    fps: int = 30,
    camera_names: tuple[str, ...] | None = None,
    camera_config: Path | None = None,
    resume: bool = False,
) -> dict:
    """Run a resumable isolated load/head-motion/video probe for each selected scene."""
    if image_size < 64 or frames < 5 or fps < 1:
        raise ValueError("Unique-scene matrix requires image_size >= 64, frames >= 5 and fps >= 1")
    camera_names = _selected_camera_names("hardware", camera_names)
    calibration = _camera_config_info(camera_config, camera_names)
    spec_bytes = SPEC_PATH.read_bytes()
    pack = json.loads(spec_bytes)
    fingerprint = pack.get("source", {}).get("content_sha256")
    if not fingerprint or fingerprint != source_content_sha256(pack):
        raise ValueError("Benchmark source fingerprint is missing or invalid")
    plan = build_unique_scene_plan(
        pack, simulators=simulators, robot_variants=robot_variants, scene_start=scene_start, scene_count=scene_count
    )
    plan.update({
        "source_content_sha256": fingerprint,
        "source_snapshot_file_sha256": _digest(spec_bytes),
        "capture": {"fps": fps, "frames": frames, "camera_image_size_argument": image_size},
        "camera_profile": "hardware",
        "camera_names": list(camera_names),
        "camera_config": calibration,
        "motion_mode": "head_pan",
        "robot_assets": {robot: _robot_asset_info("g1_op" if robot == "G1" else "g2_op", None) for robot in sorted(robot_variants)},
        "runtime_file_sha256": {sim: _runtime_file_hashes(sim) for sim in sorted(simulators)},
        "validation_scope": "one canonical source run per scene and independent G1/G2 short head-motion loads; not all task variants or task success",
    })
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    plan_path, status_path = output_root / "matrix_plan.json", output_root / "matrix_status.json"
    if plan_path.exists():
        if not resume or json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise FileExistsError("Matrix plan exists; use --resume with the exact same source, models and settings")
    else:
        if status_path.exists():
            raise FileExistsError("Matrix status exists without its original plan")
        _atomic_json(plan_path, plan)
    plan_hash = _digest(plan_path.read_bytes())
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if not resume or status.get("plan_sha256") != plan_hash:
            raise ValueError("Existing matrix status does not match the frozen plan")
    else:
        status = {"format": "roboverse.icl.geniesim_unique_scene_matrix_status.v1", "plan_sha256": plan_hash, "jobs": {}}
        _atomic_json(status_path, status)
    for job in plan["jobs"]:
        job_id = job["output_relative"]
        current = status["jobs"].setdefault(job_id, {"attempts": []})
        previous_output = current["attempts"][-1].get("output") if current["attempts"] else None
        previous_destination = output_root / previous_output if previous_output else None
        previous_report_path = previous_destination / "scene_video_report.json" if previous_destination else None
        if resume and previous_report_path and previous_report_path.is_file():
            try:
                previous = json.loads(previous_report_path.read_text(encoding="utf-8"))
                if _matrix_report_error(previous, previous_destination, job, plan) is None:
                    current["status"] = "recorded"
                    _atomic_json(status_path, status)
                    continue
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                pass
        if _digest(SPEC_PATH.read_bytes()) != plan["source_snapshot_file_sha256"] or any(
            _runtime_file_hashes(sim) != plan["runtime_file_sha256"][sim] for sim in plan["simulators"]
        ):
            raise RuntimeError("Source snapshot or runtime code changed during matrix execution; start a new plan")
        if any(
            _robot_asset_info("g1_op" if robot == "G1" else "g2_op", None) != plan["robot_assets"][robot]
            for robot in plan["robot_variants"]
        ):
            raise RuntimeError("Robot assets changed during matrix execution; start a new plan")
        if _camera_config_info(camera_config, camera_names) != calibration:
            raise RuntimeError("Camera configuration changed during matrix execution; start a new plan")
        logs = output_root / "logs" / job_id
        logs.mkdir(parents=True, exist_ok=True)
        attempt = len(current["attempts"]) + 1
        destination = output_root / job_id / f"attempt_{attempt:04d}"
        report_path = destination / "scene_video_report.json"
        stdout_path, stderr_path = logs / f"attempt_{attempt:04d}.stdout.log", logs / f"attempt_{attempt:04d}.stderr.log"
        command = [
            sys.executable, str(Path(__file__).resolve()), "--sim", job["simulator"], "--run-id", job["run_id"],
            "--instance-index", str(job["instance_index"]), "--problem-index", str(job["problem_index"]),
            "--robot-variant", job["robot_variant"], "--camera-profile", "hardware", "--motion-mode", "head_pan",
            "--camera-names", ",".join(camera_names),
            "--image-size", str(image_size), "--frames", str(frames), "--fps", str(fps), "--output", str(destination),
        ]
        if calibration is not None:
            command.extend(["--camera-config", calibration["path"]])
        entry = {"attempt": attempt, "output": str(destination.relative_to(output_root)),
                 "stdout": str(stdout_path.relative_to(output_root)), "stderr": str(stderr_path.relative_to(output_root))}
        current["attempts"].append(entry)
        current["status"] = "running"
        _atomic_json(status_path, status)
        with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
            try:
                process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
            except OSError as exc:
                print(f"Could not start probe: {type(exc).__name__}: {exc}", file=stderr, flush=True)
                returncode = -1
            else:
                try:
                    returncode = process.wait()
                except BaseException:
                    process.terminate()
                    process.wait()
                    raise
        entry["returncode"] = returncode
        reason = None
        if returncode != 0:
            reason = f"probe process exited with status {returncode}; inspect the retained logs"
        elif not report_path.is_file():
            reason = "probe exited without a report"
        else:
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                reason = _matrix_report_error(report, destination, job, plan)
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                reason = f"could not validate probe output: {exc}"
        entry["error"] = reason
        current["status"] = "recorded" if reason is None else "failed"
        _atomic_json(status_path, status)
        print(json.dumps({"job": job_id, "status": current["status"], "error": reason}, ensure_ascii=False), flush=True)
    return status


def record_semantic_matrix(
    *,
    output_root: Path,
    simulators: tuple[str, ...],
    image_size: int,
    frames: int,
    camera_profile: str,
    camera_names: tuple[str, ...] | None = None,
    camera_config: Path | None = None,
    resume: bool,
) -> dict:
    """Record one canonical source run for every semantic task and backend."""
    if not CATALOG_PATH.is_file():
        raise FileNotFoundError(CATALOG_PATH)
    camera_names = _selected_camera_names(camera_profile, camera_names)
    calibration = _camera_config_info(camera_config, camera_names)
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    matrix = []
    for semantic_task in catalog["semantic_tasks"]:
        run_id = semantic_task["run_ids"][0]
        for simulator in simulators:
            output = output_root / semantic_task["robot"] / semantic_task["subtask"] / simulator
            # Let record_one validate any existing report's source, robot,
            # camera and capture settings before it can be reused.
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--sim",
                simulator,
                "--run-id",
                run_id,
                "--image-size",
                str(image_size),
                "--frames",
                str(frames),
                "--camera-profile",
                camera_profile,
                "--camera-names",
                ",".join(camera_names),
                "--output",
                str(output),
            ]
            if calibration is not None:
                command.extend(["--camera-config", calibration["path"]])
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
            status = "recorded" if completed.returncode == 0 else "failed"
            entry = {"run_id": run_id, "simulator": simulator, "status": status, "output": str(output)}
            if completed.returncode:
                entry["returncode"] = completed.returncode
                entry["stderr_tail"] = completed.stderr[-4000:]
                entry["stdout_tail"] = completed.stdout[-4000:]
            matrix.append(entry)
            output_root.mkdir(parents=True, exist_ok=True)
            (output_root / "matrix_status.json").write_text(
                json.dumps({"format": "roboverse.icl.benchmark_video_matrix.v1", "results": matrix}, indent=2) + "\n",
                encoding="utf-8",
            )
    result = {
        "format": "roboverse.icl.benchmark_video_matrix.v1",
        "semantic_task_count": len(catalog["semantic_tasks"]),
        "simulators": list(simulators),
        "camera_profile": camera_profile,
        "camera_names": list(camera_names),
        "camera_config": calibration,
        "results": matrix,
        "recorded_count": sum(item["status"] == "recorded" for item in matrix),
        "failed_count": sum(item["status"] == "failed" for item in matrix),
        "behavioral_parity_claimed": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "matrix_status.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", choices=BACKENDS)
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--instance-index", type=int, default=0)
    parser.add_argument("--problem-index", type=int, default=0)
    parser.add_argument("--robot-root", type=Path, help="select a prepared G1/G2 robot asset for a single run")
    parser.add_argument("--robot-variant", choices=("G1", "G2"), help="select the runtime robot for a single run")
    parser.add_argument("--camera-profile", choices=("hardware", "benchmark"), default="hardware")
    parser.add_argument(
        "--camera-names", help="comma-separated camera names; default is all cameras in the selected profile"
    )
    parser.add_argument("--camera-config", type=Path, help="JSON calibration overrides for selected cameras")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--frames", type=int, default=45)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--motion-mode", choices=("hold", "head_pan"), default="hold")
    parser.add_argument(
        "--actions-json", type=Path, help="replay a provenance-bearing action sequence instead of holding"
    )
    parser.add_argument("--all-semantic-tasks", action="store_true")
    parser.add_argument("--all-scenes", action="store_true", help="probe each unique source scene with both robots")
    parser.add_argument("--scene-start", type=int, default=0, help="zero-based index in sorted unique scenes")
    parser.add_argument("--scene-count", type=int, help="limit the number of scenes in this segment")
    parser.add_argument("--robot-variants", default="G1,G2", help="comma-separated G1/G2 subset for the scene matrix")
    parser.add_argument("--simulators", default=",".join(BACKENDS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--matrix-output", type=Path)
    args = parser.parse_args()
    try:
        selected_cameras = _selected_camera_names(
            args.camera_profile, None if args.camera_names is None else tuple(name.strip() for name in args.camera_names.split(","))
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.all_semantic_tasks and args.all_scenes:
        parser.error("Choose either --all-semantic-tasks or --all-scenes")
    if args.all_scenes:
        if args.actions_json is not None or args.robot_root is not None or args.robot_variant is not None:
            parser.error("--all-scenes selects its own action and robot variants")
        if args.camera_profile != "hardware":
            parser.error("--all-scenes requires the unified hardware head-camera profile")
        if args.matrix_output is None:
            parser.error("--matrix-output is required with --all-scenes")
        result = record_unique_scene_matrix(
            output_root=args.matrix_output,
            simulators=tuple(value for value in args.simulators.split(",") if value),
            robot_variants=tuple(value for value in args.robot_variants.split(",") if value),
            scene_start=args.scene_start,
            scene_count=args.scene_count,
            image_size=args.image_size,
            frames=args.frames,
            fps=args.fps,
            camera_names=selected_cameras,
            camera_config=args.camera_config,
            resume=args.resume,
        )
        print(json.dumps({"jobs": len(result["jobs"])}, ensure_ascii=False), flush=True)
        return
    if args.all_semantic_tasks:
        if args.actions_json is not None or args.robot_root is not None or args.robot_variant is not None:
            parser.error("--actions-json, --robot-root and --robot-variant cannot be combined with --all-semantic-tasks")
        simulators = tuple(value for value in args.simulators.split(",") if value)
        if not simulators or any(value not in BACKENDS for value in simulators):
            parser.error("--simulators must be a comma-separated subset of mujoco,genesis,newton")
        if args.matrix_output is None:
            parser.error("--matrix-output is required with --all-semantic-tasks")
        result = record_semantic_matrix(
            output_root=args.matrix_output.resolve(),
            simulators=simulators,
            image_size=args.image_size,
            frames=args.frames,
            camera_profile=args.camera_profile,
            camera_names=selected_cameras,
            camera_config=args.camera_config,
            resume=args.resume,
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return
    if not args.sim or not args.run_id or args.output is None:
        parser.error("A single probe requires --sim, --run-id and --output")
    if args.actions_json is not None:
        if args.motion_mode != "hold":
            parser.error("--motion-mode cannot be combined with --actions-json")
        report = record_action_replay(
            simulator=args.sim,
            run_id=args.run_id,
            output=args.output,
            actions_path=args.actions_json,
            instance_index=args.instance_index,
            problem_index=args.problem_index,
            image_size=args.image_size,
            fps=args.fps,
            camera_profile=args.camera_profile,
            camera_names=selected_cameras,
            camera_config=args.camera_config,
            robot_root=args.robot_root,
            robot_variant=args.robot_variant,
        )
        print(json.dumps({"outcome": report["outcome"], "steps": len(report["steps"])}, ensure_ascii=False), flush=True)
        if report["outcome"] == "action_exception":
            raise SystemExit(1)
        return
    record_one(
        simulator=args.sim,
        run_id=args.run_id,
        output=args.output,
        instance_index=args.instance_index,
        problem_index=args.problem_index,
        image_size=args.image_size,
        frames=args.frames,
        fps=args.fps,
        camera_profile=args.camera_profile,
        camera_names=selected_cameras,
        camera_config=args.camera_config,
        robot_root=args.robot_root,
        robot_variant=args.robot_variant,
        motion_mode=args.motion_mode,
    )


if __name__ == "__main__":
    main()
