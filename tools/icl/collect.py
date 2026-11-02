"""Collect one physical ICL demonstration or pair two recorded demonstrations."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import h5py
import numpy as np
import torch

from metasim.utils.demo_utils import validate_demo_time_base
from metasim.utils.trajectory import _detach, episode_from_states, read_header, save_episode
from roboverse_learn.il.icl.data import (
    ACTION_SPEC,
    preprocess_trajectory,
    read_metadata,
    validate_splits,
    write_pair,
    write_trajectory,
)
from roboverse_pack.tasks.icl.control import FrankaCartesianController
from roboverse_pack.tasks.icl.push_to_region import (
    CONTROL_PERIOD,
    GOAL_CENTERS,
    PickPlaceRegionTask,
    PushToRegionTask,
    build_scenario,
)

PHASES = ["start", "approach", "lower", "push", "lift", "settled"]
PICK_PHASES = ["start", "approach", "grasp", "lift", "place", "settled"]
CAMERAS = {"images_front": "front", "images_side": "side", "images_wrist": "overhead"}
CAMERA_ROLES = {"images_front": "external_front", "images_side": "external_side", "images_wrist": "external_overhead"}


class DemonstrationFailure(RuntimeError):
    """The expert could not complete the requested physical task."""


def _capture_valid_masks(state, cameras, previous: dict[str, np.ndarray | None]) -> None:
    """Capture static CameraState masks, rejecting cameras whose masks change over time."""
    for key, camera in cameras.items():
        tensor = state.cameras[camera].valid_mask
        mask = None if tensor is None else tensor[0].detach().cpu().numpy().copy()
        if mask is not None and (mask.dtype != np.bool_ or mask.ndim != 2):
            raise ValueError(f"Invalid camera valid_mask for {camera}")
        if key in previous:
            prior = previous[key]
            if (prior is None) != (mask is None) or (prior is not None and not np.array_equal(prior, mask)):
                raise ValueError(f"Camera valid_mask changed during recording for {camera}")
        else:
            previous[key] = mask


def collect(args: argparse.Namespace) -> Path:
    """Run a single closed-loop expert rollout and export it only if it succeeds."""
    jitter = getattr(args, "position_jitter", 0.003)
    if not np.isfinite(jitter) or not 0 <= jitter <= 0.03:
        raise ValueError("position jitter must be between 0 and 0.03 meters")
    rng = np.random.default_rng(args.seed)
    xy = (0.5 + float(rng.uniform(-jitter, jitter)), float(rng.uniform(-jitter, jitter)))
    scenario = build_scenario(simulator=args.sim, object_id=args.object_id, image_size=args.image_size, object_xy=xy)
    asset_info = json.loads(
        (Path(scenario.objects[0].urdf_path).parent / "provenance.json").read_text(encoding="utf-8")
    )
    asset_fingerprint = hashlib.sha256(
        (
            asset_info["source_files_sha256"]["textured.obj"] + asset_info["source_files_sha256"]["collision.ply"]
        ).encode()
    ).hexdigest()
    task = getattr(args, "task", "push")
    if task not in ("push", "pick_place"):
        raise ValueError(f"Unsupported task: {task}")
    task_id = "icl.push_to_region" if task == "push" else "icl.pick_place_region"
    origin = f"{args.sim}_{args.object_id}_{args.goal}_{args.seed}"
    if task == "pick_place":
        origin = f"pick_{origin}"
    raw_dir = args.output / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    record_path = raw_dir / f"{origin}.npz"
    hdf5_path = raw_dir / f"{origin}.hdf5"
    result_path = raw_dir / f"{origin}.json"
    if any(path.exists() for path in (record_path, hdf5_path, result_path)):
        raise FileExistsError(f"A recording for {origin} already exists in {raw_dir}")

    env = (PushToRegionTask if task == "push" else PickPlaceRegionTask)(scenario, goal=args.goal)
    success = False
    reason = ""
    failure_kind = ""
    physical_states = []
    joint_actions = []
    actions = []
    waypoint_indices = [0]
    assigned = []
    images = {key: [] for key in CAMERAS}
    camera_masks: dict[str, np.ndarray | None] = {}
    robot_states = []
    try:
        validate_demo_time_base(env.handler, expected_dt=CONTROL_PERIOD / 15, decimation=15)
        obs, _ = env.reset()
        joint_names = env.handler.get_joint_names("franka", sort=True)
        controller = FrankaCartesianController(scenario.robots[0], joint_names)
        _, desired_rotation = controller.tcp_pose(obs.robots["franka"].joint_pos[0].cpu().numpy())

        def snapshot(state) -> None:
            physical = _detach(state)
            q = state.robots["franka"].joint_pos[0].detach().cpu().numpy()
            position, rotation = controller.tcp_pose(q)
            finger_indices = [controller.joint_names.index(name) for name in scenario.robots[0].ee_joint_names]
            openness = float(np.clip(np.mean(q[finger_indices]) / 0.04, 0.0, 1.0))
            state_vector = np.r_[position, rotation.as_rotvec(), openness].astype(np.float32)
            camera_frames = {}
            for key, camera in CAMERAS.items():
                frame = state.cameras[camera].rgb[0].detach().cpu().numpy()
                if frame.dtype != np.uint8 or frame.shape != (args.image_size, args.image_size, 3):
                    raise ValueError(f"Unexpected RGB frame for {camera}: {frame.shape} {frame.dtype}")
                camera_frames[key] = frame.copy()
            _capture_valid_masks(state, CAMERAS, camera_masks)
            physical_states.append(physical)
            robot_states.append(state_vector)
            for key, frame in camera_frames.items():
                images[key].append(frame)

        def step(action: np.ndarray, phase: int):
            nonlocal obs
            q = obs.robots["franka"].joint_pos[0].detach().cpu().numpy()
            joints = controller.joint_targets(q, action)
            next_obs, _, _, timeout, _ = env.step([{"franka": {"dof_pos_target": joints}}])
            snapshot(next_obs)
            joint_actions.append(
                torch.tensor([[joints[name.split("/")[-1]] for name in joint_names]], dtype=torch.float32)
            )
            actions.append(action.astype(np.float32))
            assigned.append(phase)
            obs = next_obs
            if bool(timeout[0]):
                raise DemonstrationFailure("Task timed out before the rollout completed")
            return obs

        snapshot(obs)
        if task == "pick_place":

            def move(target, gripper: float, phase: int) -> None:
                target = np.asarray(target, dtype=float)
                for _ in range(180):
                    q = obs.robots["franka"].joint_pos[0].detach().cpu().numpy()
                    position, orientation = controller.tcp_pose(q)
                    delta = target - position
                    distance = np.linalg.norm(delta)
                    if distance < 0.012:
                        return
                    rotation_delta = (desired_rotation * orientation.inv()).as_rotvec()
                    rotation_delta *= min(1.0, 0.15 / max(np.linalg.norm(rotation_delta), 1e-10))
                    step(np.r_[delta * min(1.0, 0.015 / distance), rotation_delta, gripper], phase)
                raise DemonstrationFailure(f"Expert could not reach pick-and-place phase {phase}")

            def hold(count: int, gripper: float, phase: int) -> None:
                for _ in range(count):
                    step(np.r_[np.zeros(6), gripper], phase)

            hold(15, 1.0, 1)
            move([*xy, 0.18], 1.0, 1)
            move([*xy, 0.032], 1.0, 1)
            waypoint_indices.append(len(actions))
            hold(35, 0.0, 2)
            waypoint_indices.append(len(actions))
            move([*xy, 0.18], 0.0, 3)
            if not bool(env._lifted[0]):
                raise DemonstrationFailure("Object was not lifted during grasp")
            waypoint_indices.append(len(actions))
            goal_xy = GOAL_CENTERS[args.goal]
            move([*goal_xy, 0.18], 0.0, 4)
            move([*goal_xy, 0.04], 0.0, 4)
            hold(30, 1.0, 4)
            waypoint_indices.append(len(actions))
            move([*goal_xy, 0.18], 1.0, 5)
            for settle_step in range(75):
                step(np.r_[np.zeros(6), 1.0], 5)
                if settle_step >= 11 and bool(env._terminated(obs)[0]):
                    break
            waypoint_indices.append(len(actions))
        else:
            _collect_push(
                obs, env, controller, desired_rotation, asset_info, xy, args.goal, step, actions, waypoint_indices
            )
        success = bool(env._terminated(obs)[0])
        if not success:
            reason = "Final physical state did not satisfy the task's success checker"
            failure_kind = "task_failed"
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        failure_kind = "task_failed" if isinstance(exc, DemonstrationFailure) else "execution_error"
    finally:
        try:
            if physical_states:
                record = episode_from_states(
                    env.handler,
                    physical_states,
                    joint_actions,
                    seed=args.seed,
                    info={
                        "task": task_id,
                        "goal": args.goal,
                        "success": success,
                        "failure": reason,
                        "failure_kind": failure_kind,
                    },
                )
                save_episode(record, record_path)
            result_path.write_text(
                json.dumps(
                    {
                        "origin_id": origin,
                        "success": success,
                        "failure": reason,
                        "failure_kind": failure_kind,
                        "steps": len(actions),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        finally:
            env.close()

    if not success:
        raise RuntimeError(f"Demonstration {origin} failed: {reason}; see {result_path}")
    actions.append(np.array([0, 0, 0, 0, 0, 0, actions[-1][6]], dtype=np.float32))
    assigned.append(5)
    metadata = {
        "origin_id": origin,
        "task": task_id,
        "goal": args.goal,
        "split": args.split,
        "object_id": args.object_id,
        "asset_sha256": asset_fingerprint,
        "simulator": args.sim,
        "control_period": CONTROL_PERIOD,
        "camera_roles": CAMERA_ROLES,
        "camera_valid_masks": {
            key: {"source": "metasim.CameraState.valid_mask"} for key, mask in camera_masks.items() if mask is not None
        },
        "camera_configs": {
            camera.name: {
                "pos": camera.pos,
                "look_at": camera.look_at,
                "width": camera.width,
                "height": camera.height,
                "horizontal_fov": camera.horizontal_fov,
            }
            for camera in scenario.cameras
        },
        "demonstration_kind": "physical_closed_loop",
        "image_processing": "unsegmented_rgb",
        "success": True,
        "physical_record": record_path.name,
        "action_spec": ACTION_SPEC,
        "phase_schema": PHASES if task == "push" else PICK_PHASES,
        "seed": args.seed,
        "position_jitter": jitter,
        "image_size": args.image_size,
        "initial_object_xy": xy,
    }
    return write_trajectory(
        hdf5_path,
        metadata=metadata,
        images={key: np.stack(frames) for key, frames in images.items()},
        robot_states=np.stack(robot_states),
        actions=np.stack(actions),
        waypoints_idx=np.array(waypoint_indices, dtype=np.int64),
        assigned_wps=np.array(assigned, dtype=np.int64),
        valid_masks={key: mask for key, mask in camera_masks.items() if mask is not None},
    )


def _collect_push(obs, env, controller, desired_rotation, asset_info, xy, goal, step, actions, waypoint_indices):
    """Execute the push expert; the state is obtained from the handler after each step."""
    direction = 1.0 if goal == "left" else -1.0
    bounds = np.asarray(asset_info["bounds_m"], dtype=float)
    extents = bounds[1] - bounds[0]
    round_object = bool(np.all(extents > 0) and np.max(extents) / np.min(extents) < 1.2)
    if round_object:
        push_direction = np.asarray(GOAL_CENTERS[goal]) - np.asarray(xy)
        push_direction /= np.linalg.norm(push_direction)
        approach_xy = np.asarray(xy) - 0.11 * push_direction
        push_xy = np.asarray(xy) + 0.14 * push_direction
    else:
        approach_xy = np.asarray(xy) + [0.0, -direction * 0.11]
        push_xy = np.asarray(xy) + [0.0, direction * 0.12]
    stages = (
        ("approach", np.r_[approach_xy, 0.13]),
        ("lower", np.r_[approach_xy, 0.025]),
        ("push", np.r_[push_xy, 0.025]),
        ("lift", np.r_[push_xy, 0.13]),
    )
    lift_target = None
    for phase, (label, target) in enumerate(stages, start=1):
        if label == "lift" and lift_target is not None:
            target = lift_target
        start_step = len(actions)
        reached = False
        for _ in range(200):
            q = obs.robots["franka"].joint_pos[0].detach().cpu().numpy()
            position, orientation = controller.tcp_pose(q)
            if label == "push":
                object_xy = obs.objects["object"].root_state[0, :2].detach().cpu().numpy()
                margin = 0.035 if round_object else 0.05
                if np.all(np.abs(object_xy - GOAL_CENTERS[goal]) < margin):
                    reached = True
                    lift_target = np.r_[position[:2], 0.13]
                    break
            delta = target - position
            distance = np.linalg.norm(delta)
            if label != "push" and distance < 0.02:
                reached = True
                break
            if label == "push" and distance < 0.01:
                break
            rotation_delta = (desired_rotation * orientation.inv()).as_rotvec()
            rotation_norm = np.linalg.norm(rotation_delta)
            if rotation_norm > 0.15:
                rotation_delta *= 0.15 / rotation_norm
            speed = 0.01 if round_object and label == "push" else 0.02
            action = np.r_[delta * min(1.0, speed / distance), rotation_delta, 0.0]
            obs = step(action, phase)
        if not reached or len(actions) == start_step:
            raise DemonstrationFailure(f"Expert did not reach phase {label!r}")
        waypoint_indices.append(len(actions))
    for settle_step in range(75):
        obs = step(np.zeros(7), 5)
        if settle_step >= 11 and bool(env._terminated(obs)[0]):
            break
    waypoint_indices.append(len(actions))


def batch(args: argparse.Namespace) -> dict:
    """Run a prevalidated explicit collection plan in isolated processes."""
    jobs = json.loads(args.plan.read_text(encoding="utf-8"))
    jitter = getattr(args, "position_jitter", 0.003)
    if not np.isfinite(jitter) or not 0 <= jitter <= 0.03:
        raise ValueError("position jitter must be between 0 and 0.03 meters")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("A collection plan must be a nonempty JSON array")
    allowed = {"sim", "goal", "split", "object_id", "seed"}
    existing = sorted(args.output.rglob("*.hdf5"))
    validate_splits(existing)
    split_by_asset = {read_metadata(path)["asset_sha256"]: read_metadata(path)["split"] for path in existing}
    origins = set()
    for job in jobs:
        if not isinstance(job, dict) or not allowed <= set(job) or set(job) - allowed - {"task"}:
            raise ValueError(f"Each job must contain {sorted(allowed)} and may include task")
        if job.get("task", "push") not in ("push", "pick_place"):
            raise ValueError(f"Invalid task in {job}")
        if job["sim"] not in ("mujoco", "genesis", "newton") or job["goal"] not in ("left", "right"):
            raise ValueError(f"Invalid simulator or goal in {job}")
        if job["split"] not in ("train", "validation", "test") or type(job["seed"]) is not int or job["seed"] < 0:
            raise ValueError(f"Invalid split or seed in {job}")
        scenario = build_scenario(simulator=job["sim"], object_id=job["object_id"], image_size=args.image_size)
        info = json.loads((Path(scenario.objects[0].urdf_path).parent / "provenance.json").read_text(encoding="utf-8"))
        fingerprint = hashlib.sha256(
            (info["source_files_sha256"]["textured.obj"] + info["source_files_sha256"]["collision.ply"]).encode()
        ).hexdigest()
        if fingerprint in split_by_asset and split_by_asset[fingerprint] != job["split"]:
            raise ValueError(f"Asset {job['object_id']} is assigned to multiple splits")
        split_by_asset[fingerprint] = job["split"]
        origin = f"{job['sim']}_{job['object_id']}_{job['goal']}_{job['seed']}"
        if job.get("task", "push") == "pick_place":
            origin = f"pick_{origin}"
        if origin in origins:
            raise ValueError(f"Duplicate job: {origin}")
        origins.add(origin)

    logs = args.output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    results = []
    for job in jobs:
        origin = f"{job['sim']}_{job['object_id']}_{job['goal']}_{job['seed']}"
        if job.get("task", "push") == "pick_place":
            origin = f"pick_{origin}"
        raw = args.output / "raw"
        trajectory, record, result = (raw / f"{origin}{suffix}" for suffix in (".hdf5", ".npz", ".json"))
        if any(path.exists() for path in (trajectory, record, result)):
            status = "incomplete"
            if result.is_file():
                result_info = json.loads(result.read_text(encoding="utf-8"))
                status = (
                    "success" if result_info.get("success") is True else result_info.get("failure_kind", "task_failed")
                )
            if status == "success":
                if not trajectory.is_file() or not record.is_file():
                    status = "incomplete"
                else:
                    meta = read_metadata(trajectory)
                    if any(meta[key] != job[key] for key in ("goal", "split", "object_id", "seed")):
                        raise ValueError(f"Existing recording does not match job: {origin}")
                    task_id = (
                        "icl.pick_place_region" if job.get("task", "push") == "pick_place" else "icl.push_to_region"
                    )
                    if meta["task"] != task_id:
                        raise ValueError(f"Existing task does not match job: {origin}")
                    if meta.get("position_jitter", 0.003) != jitter:
                        raise ValueError(f"Existing position jitter does not match job: {origin}")
                    with h5py.File(trajectory, "r") as h5:
                        if any(
                            h5["trajectory"][key].shape[1:3] != (args.image_size, args.image_size) for key in CAMERAS
                        ):
                            raise ValueError(f"Existing image size does not match job: {origin}")
                    header = read_header(record)
                    if (
                        header["info"].get("success") is not True
                        or header["info"].get("goal") != job["goal"]
                        or header["info"].get("task") != task_id
                        or header["provenance"].get("simulator") != job["sim"]
                    ):
                        raise ValueError(f"Existing physical record does not match job: {origin}")
            results.append({"origin_id": origin, "status": status, "existing": True})
            continue
        log = logs / f"{origin}.log"
        attempt = 1
        while log.exists():
            log = logs / f"{origin}.retry-{attempt}.log"
            attempt += 1
        command = [sys.executable, str(Path(__file__).resolve()), "collect"]
        for key in ("sim", "goal", "split", "object_id", "seed"):
            command.extend((f"--{key.replace('_', '-')}", str(job[key])))
        command.extend(("--image-size", str(args.image_size), "--output", str(args.output)))
        command.extend(("--task", job.get("task", "push")))
        command.extend(("--position-jitter", str(jitter)))
        with log.open("x", encoding="utf-8") as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        if completed.returncode == 0 and trajectory.is_file() and record.is_file() and result.is_file():
            status = "success"
        elif result.is_file() and json.loads(result.read_text(encoding="utf-8")).get("success") is False:
            status = json.loads(result.read_text(encoding="utf-8")).get("failure_kind", "task_failed")
        else:
            status = "execution_error"
        results.append({
            "origin_id": origin,
            "status": status,
            "existing": False,
            "log": str(log),
            "exit_code": completed.returncode,
        })
        print(f"{origin}: {status}", flush=True)
    return {"jobs": results, "success": sum(item["status"] == "success" for item in results), "total": len(results)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("collect", help="Record one independent physical rollout")
    run.add_argument("--sim", choices=("mujoco", "genesis", "newton"), required=True)
    run.add_argument("--task", choices=("push", "pick_place"), default="push")
    run.add_argument("--goal", choices=("left", "right"), required=True)
    run.add_argument("--object-id", default="008_pudding_box")
    run.add_argument("--split", choices=("train", "validation", "test"), required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--image-size", type=int, default=256)
    run.add_argument("--position-jitter", type=float, default=0.003, help="Initial XY jitter in meters (0 to 0.03)")
    run.add_argument("--output", type=Path, required=True, help="Dataset library root")
    pair = subparsers.add_parser("pair", help="Create a linked SynthICL pair")
    pair.add_argument("--reference", type=Path, required=True)
    pair.add_argument("--query", type=Path, required=True)
    pair.add_argument("--output", type=Path, required=True)
    preprocess = subparsers.add_parser("preprocess", help="Create a traceable uniform-size copy of one raw trajectory")
    preprocess.add_argument("--source", type=Path, required=True)
    preprocess.add_argument("--output", type=Path, required=True)
    preprocess.add_argument("--height", type=int, default=256)
    preprocess.add_argument("--width", type=int, default=256)
    preprocess.add_argument("--mode", choices=("stretch", "letterbox", "center_crop"), default="letterbox")
    preprocess.add_argument("--batch-size", type=int, default=8)
    check = subparsers.add_parser("check-splits", help="Check a library for origin and asset leakage")
    check.add_argument("--library", type=Path, required=True)
    run_batch = subparsers.add_parser("batch", help="Execute an explicit split-safe collection plan")
    run_batch.add_argument("--plan", type=Path, required=True, help="JSON array of collection jobs")
    run_batch.add_argument("--output", type=Path, required=True)
    run_batch.add_argument("--image-size", type=int, default=256)
    run_batch.add_argument("--position-jitter", type=float, default=0.003)
    args = parser.parse_args()
    if args.command == "collect":
        if args.seed < 0:
            parser.error("--seed must be nonnegative")
        print(f"Saved {collect(args)}")
    elif args.command == "pair":
        print(f"Saved {write_pair(args.output, reference=args.reference, query=args.query)}")
    elif args.command == "preprocess":
        output = preprocess_trajectory(args.source, args.output, target_hw=(args.height, args.width),
                                       mode=args.mode, batch_size=args.batch_size)
        print(f"Saved {output}")
    elif args.command == "batch":
        report = batch(args)
        report_dir = args.output / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"collection-{uuid.uuid4().hex}.json"
        payload = {
            "plan": json.loads(args.plan.read_text(encoding="utf-8")),
            "image_size": args.image_size,
            "position_jitter": args.position_jitter,
            **report,
        }
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=report_dir, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        try:
            temporary.rename(report_path)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Saved batch report {report_path}")
        print(json.dumps(report, indent=2))
        if report["success"] != report["total"]:
            raise SystemExit(1)
    else:
        paths = sorted(args.library.rglob("*.hdf5"))
        if not paths:
            parser.error("No raw .hdf5 episodes found")
        validate_splits(paths)
        print(f"Split validation passed for {len(paths)} episodes")


if __name__ == "__main__":
    main()
