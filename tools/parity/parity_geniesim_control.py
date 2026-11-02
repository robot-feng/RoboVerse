"""Measure a shared GenieSim G1 joint trajectory across simulation backends.

This is a diagnostic motion, not a benchmark solution. Each backend is run in
its own process, with the same named position targets and simulated time. The
recorded link poses are backend observations, not forward-kinematics estimates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BACKENDS = ("mujoco", "genesis", "newton")
RUN_ID = "g1op_s2r_bimanual_chip_handover"


def position_targets(initial: np.ndarray, names: tuple[str, ...], step: int, steps: int) -> np.ndarray:
    """A shared smooth out-and-back command in two arm joints, in radians."""
    if not 0 <= step <= steps or steps < 2:
        raise ValueError("Expected 0 <= step <= steps and at least two steps")
    targets = np.asarray(initial, dtype=np.float64).copy()
    amplitude = 0.08 * np.sin(np.pi * step / steps) ** 2
    for name, direction in (("idx23_arm_l_joint3", 1), ("idx63_arm_r_joint3", -1)):
        targets[names.index(name)] += direction * amplitude
    return targets


def _sample(task, observation, step: int, command: np.ndarray) -> dict:
    robot = observation.robots[task.robot_name]
    q = robot.joint_pos[0, task._control_indices].detach().cpu().numpy()
    if robot.body_state is None or robot.body_names is None:
        raise RuntimeError("Backend did not supply robot body poses")
    links = {}
    for side in ("l", "r"):
        name = f"gripper_{side}_center_link"
        matches = [i for i, candidate in enumerate(robot.body_names) if candidate.split("/")[-1] == name]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {name} body, found {len(matches)}")
        links[name] = robot.body_state[0, matches[0], :7].detach().cpu().tolist()
    return {
        "step": step,
        "time_s": step * task.handler.env_step_s,
        "target": command.tolist(),
        "joint_pos": q.tolist(),
        "gripper_link_poses": links,
        "object_poses": {
            name: state.root_state[0, :7].detach().cpu().tolist() for name, state in observation.objects.items()
        },
    }


def record_backend(simulator: str, run_id: str, output: Path, steps: int, image_size: int) -> dict:
    """Capture measured joint, link, object and video trajectories for one backend."""
    from roboverse_pack.tasks.icl.geniesim_benchmark import SPEC_PATH, GenieSimBenchmarkTask
    from tools.icl.video_utils import RGBVideoWriter

    shared_files = (
        ROOT / "roboverse_pack/tasks/icl/control.py",
        ROOT / "roboverse_pack/tasks/icl/geniesim_scene.py",
        ROOT / "roboverse_pack/tasks/icl/geniesim_benchmark.py",
        ROOT / "roboverse_pack/tasks/icl/geniesim_checker.py",
    )
    def shared_hashes() -> dict[str, str]:
        return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in shared_files}

    source_hash = hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest()
    code_hashes = shared_hashes()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    output.mkdir(parents=True)
    task = None
    writer = None
    try:
        task = GenieSimBenchmarkTask(
            run_id=run_id, simulator=simulator, action_mode="joint_position", image_size=image_size,
            max_episode_steps=steps + 5,
        )
        if task.source_robot != "G1":
            raise ValueError("This diagnostic currently requires a G1 benchmark task")
        obs, _ = task.reset()
        # build_run_scenario stores the canonical source reset (including
        # bounded gripper clipping) in the robot config. Use that identical
        # float64 target vector on each backend; starting from each backend's
        # float32 observation would make the supposedly shared trajectory
        # differ by backend rounding. Record that reset-state discrepancy
        # separately instead of hiding it in the target comparison.
        initial = np.asarray(
            [task.scenario.robots[0].default_joint_positions[name] for name in task.control_joint_names],
            dtype=np.float64,
        )
        observed_initial = (
            obs.robots[task.robot_name].joint_pos[0, task._control_indices].detach().cpu().numpy().copy()
        )
        reset_position_delta = observed_initial - initial
        fps = round(1.0 / task.handler.env_step_s)
        if abs(fps * task.handler.env_step_s - 1) > 1e-8:
            raise ValueError("Recording fps cannot represent this backend's action interval")
        samples = []
        for step in range(steps + 1):
            command = position_targets(initial, task.control_joint_names, step, steps)
            if step:
                obs, _, _, _, info = task.step(command)
            else:
                info = {}
            samples.append(_sample(task, obs, step, command))
            rgb = obs.cameras["head_rgb"].rgb[0].detach().cpu().numpy()
            if writer is None:
                writer = RGBVideoWriter(output / f"{simulator}_head_rgb.mp4", rgb.shape[1], rgb.shape[0], fps)
            writer.append(rgb)
        writer.close()
        writer = None
        result = {
            "format": "roboverse.icl.geniesim_control_probe.v1",
            "run_id": run_id,
            "simulator": simulator,
            "simulator_version": importlib.metadata.version(
                {"mujoco": "mujoco", "genesis": "genesis-world", "newton": "newton"}[simulator]
            ),
            "source_snapshot_sha256": source_hash,
            "source_snapshot_changed_during_run": source_hash != hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest(),
            "shared_code_sha256": code_hashes,
            "shared_code_changed_during_run": code_hashes != shared_hashes(),
            "robot_mjcf_sha256": hashlib.sha256(Path(task.scenario.robots[0].mjcf_path).read_bytes()).hexdigest(),
            "control_joint_names": list(task.control_joint_names),
            "physics_step_s": task.scenario.sim_params.dt,
            "env_step_s": task.handler.env_step_s,
            "video_fps": fps,
            "reset_pose_validation": task.run.get("reset_pose_validation"),
            "trajectory_initial_target_source": "configured_source_reset_pose",
            "reset_joint_position_error_rad": float(np.max(np.abs(reset_position_delta))),
            "reset_joint_position_error_joint": task.control_joint_names[int(np.argmax(np.abs(reset_position_delta)))],
            "checker_status_last": info.get("geniesim_checker_status"),
            "task_success_claimed": False,
            "samples": samples,
        }
        (output / "trajectory.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        if writer is not None:
            writer.close()
        if task is not None:
            task.close()


def _orientation_difference(first, second) -> float:
    """Shortest angular separation between scalar-first pose quaternions."""
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Pose quaternions must contain four finite values")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        raise ValueError("Pose quaternion has zero norm")
    return float(2 * np.arccos(np.clip(abs(np.dot(a / na, b / nb)), 0, 1)))


def compare_trajectories(reference: dict, actual: dict) -> dict:
    """Compare same-timestamp measured states; reject incomparable trajectories."""
    if (reference["run_id"], reference["control_joint_names"], len(reference["samples"])) != (
        actual["run_id"], actual["control_joint_names"], len(actual["samples"])
    ):
        raise ValueError("Trajectories use different tasks, control joints, or sample counts")
    digest_checks = {}
    if any(item.get("source_snapshot_changed_during_run") or item.get("shared_code_changed_during_run")
           for item in (reference, actual)):
        raise ValueError("Source snapshot or shared code changed during a recording")
    first_code, second_code = reference.get("shared_code_sha256"), actual.get("shared_code_sha256")
    if first_code and second_code and first_code != second_code:
        raise ValueError("Trajectories use different shared code")
    digest_checks["shared_code_sha256"] = "matched" if first_code and second_code else "not_recorded"
    for key in ("source_snapshot_sha256", "robot_mjcf_sha256"):
        left_digest, right_digest = reference.get(key), actual.get(key)
        if left_digest and right_digest and left_digest != right_digest:
            raise ValueError(f"Trajectories have different {key}")
        digest_checks[key] = "matched" if left_digest and right_digest else "not_recorded"
    joint_error, target_error, link_error, link_rotation_error, object_error = [], [], [], [], []
    for left, right in zip(reference["samples"], actual["samples"], strict=True):
        if left["step"] != right["step"] or abs(left["time_s"] - right["time_s"]) > 1e-8:
            raise ValueError("Trajectories have different simulation timestamps")
        target_error.append(float(np.max(np.abs(np.asarray(left["target"]) - right["target"]))))
        joint_error.append(float(np.max(np.abs(np.asarray(left["joint_pos"]) - right["joint_pos"]))))
        if left["gripper_link_poses"].keys() != right["gripper_link_poses"].keys():
            raise ValueError("Trajectories have different gripper links")
        if left["object_poses"].keys() != right["object_poses"].keys():
            raise ValueError("Trajectories have different objects")
        link_error.extend(
            float(np.linalg.norm(np.asarray(pose[:3]) - right["gripper_link_poses"][name][:3]))
            for name, pose in left["gripper_link_poses"].items()
        )
        link_rotation_error.extend(
            _orientation_difference(pose[3:7], right["gripper_link_poses"][name][3:7])
            for name, pose in left["gripper_link_poses"].items()
        )
        object_error.extend(
            float(np.linalg.norm(np.asarray(pose[:3]) - right["object_poses"][name][:3]))
            for name, pose in left["object_poses"].items()
        )
    if max(target_error) > 1e-8:
        raise ValueError("Trajectories were not driven by the same targets")
    return {
        "reference": reference["simulator"], "actual": actual["simulator"],
        "max_joint_difference_rad": max(joint_error),
        "max_gripper_position_difference_m": max(link_error),
        "max_gripper_orientation_difference_rad": max(link_rotation_error),
        "max_object_position_difference_m": max(object_error, default=0.0),
        "sample_count": len(joint_error),
        "provenance_comparison": digest_checks,
        "task_success_claimed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", choices=BACKENDS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()
    if bool(args.sim) == args.all or args.steps < 2 or args.image_size < 64:
        parser.error("Choose --sim or --all; steps must be >=2 and image-size >=64")
    if args.sim:
        record_backend(args.sim, args.run_id, args.output.resolve(), args.steps, args.image_size)
        return
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    args.output.mkdir(parents=True)
    results = {}
    for simulator in BACKENDS:
        command = [sys.executable, str(Path(__file__).resolve()), "--sim", simulator, "--run-id", args.run_id,
                   "--output", str(args.output / simulator), "--steps", str(args.steps),
                   "--image-size", str(args.image_size)]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        (args.output / f"{simulator}.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"{simulator} exited {completed.returncode}; see {args.output / f'{simulator}.log'}")
        results[simulator] = json.loads((args.output / simulator / "trajectory.json").read_text(encoding="utf-8"))
    comparisons = [compare_trajectories(results["mujoco"], results[name]) for name in ("genesis", "newton")]
    (args.output / "comparison.json").write_text(json.dumps(comparisons, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
