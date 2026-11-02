"""Record a deterministic G1 apple-to-plate manipulation attempt for review.

This is a physics rollout diagnostic, not a general-purpose policy. It records
the actual per-step joint targets, robot/object state trace, and rendered views
so a candidate scripted demonstration can be inspected before acceptance.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import cv2
import mujoco
import numpy as np

from roboverse_pack.tasks.icl.control import G1CartesianController
from roboverse_pack.tasks.icl.fruit_sorting import (
    APPLE_DELIVERY_SCENE_ROOT,
    CONTROL_JOINTS,
    GenieSimG1AppleDeliveryTask,
    build_scenario,
)

CAMERA_NAMES = ("overview", "head_rgb", "head_fisheye_left", "head_fisheye_right")


def record_demo(*, simulator: str, output: Path, image_size: int = 640) -> dict:
    """Run and record one pick-and-place attempt, keeping partial evidence on failure."""
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    writers = {}
    frame_shapes = {}
    actions = []
    trace = []
    report = {
        "format": "roboverse.icl.g1_apple_delivery_rollout.v1",
        "task": "icl.geniesim_g1_apple_delivery",
        "simulator": simulator,
        "robot": "G1_omnipicker",
        "camera_names": list(CAMERA_NAMES),
        "image_size": image_size,
        "drive_gain_scales": None,
        "outcome": "initializing",
        "numerical_joint_limit_clip_count": 0,
    }
    env = None
    state = None
    controller = None

    def controls() -> np.ndarray:
        return env.control_state(state)[0].detach().cpu().numpy().copy()

    def tcp_position() -> np.ndarray:
        return controller.tcp_pose(controls())[0]

    def apple_pose() -> np.ndarray:
        return state.objects["apple"].root_state[0, :7].detach().cpu().numpy().copy()

    def capture() -> None:
        for name in CAMERA_NAMES:
            image = state.cameras[name].rgb[0].detach().cpu().numpy()
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 or image.std() < 1:
                raise RuntimeError(f"Invalid or blank RGB frame from {name!r}: {image.shape} {image.dtype}")
            if name not in writers:
                height, width = image.shape[:2]
                frame_shapes[name] = list(image.shape)
                writers[name] = cv2.VideoWriter(
                    str(output / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height)
                )
                if not writers[name].isOpened():
                    raise RuntimeError(f"Could not open video writer for {name!r}")
            writers[name].write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def step_target(target: np.ndarray, phase: str) -> np.ndarray:
        nonlocal state
        limits = np.asarray([env.scenario.robots[0].joint_limits[name] for name in CONTROL_JOINTS], dtype=float)
        clipped = np.clip(target, limits[:, 0], limits[:, 1])
        clip_delta = np.abs(clipped - target)
        if float(clip_delta.max(initial=0.0)) > 1e-3:
            raise ValueError("Controller produced a target more than 1e-3 rad outside a joint limit")
        report["numerical_joint_limit_clip_count"] += int(np.count_nonzero(clip_delta))
        target = clipped
        state, reward, success, timeout, info = env.step(env.position_action(target))
        actions.append({"phase": phase, "dof_pos_target": np.asarray(target, dtype=float).tolist()})
        capture()
        apple = apple_pose()
        row = {
            "step": len(actions),
            "phase": phase,
            "tcp_position_m": tcp_position().tolist(),
            "apple_root_position_m": apple[:3].tolist(),
            "success": bool(success[0]),
            "timeout": bool(timeout[0]),
            "task_failure": bool(info.get("task_failure", [False])[0]),
        }
        trace.append(row)
        if row["timeout"]:
            raise RuntimeError(f"Task ended during {phase}: failure={row['task_failure']}")
        return success.detach().cpu().numpy()

    def hold(gripper: float, frames: int, phase: str) -> None:
        for _ in range(frames):
            q = controls()
            target = q.copy()
            grip_index = CONTROL_JOINTS.index("idx41_gripper_l_outer_joint1")
            low, high = env.scenario.robots[0].joint_limits["idx41_gripper_l_outer_joint1"]
            target[grip_index] = low + gripper * (high - low)
            step_target(target, phase)

    def move_to(goal: np.ndarray, gripper: float, phase: str, max_chunks: int = 60) -> None:
        goal = np.asarray(goal, dtype=float)
        for _ in range(max_chunks):
            q = controls()
            current = controller.tcp_pose(q)[0]
            delta = goal - current
            distance = float(np.linalg.norm(delta))
            if distance < 0.006:
                return
            delta *= min(1.0, 0.018 / max(distance, 1e-9))
            target = controller.joint_targets(q, np.r_[delta, 0.0, 0.0, 0.0, gripper])
            for _ in range(7):
                done = step_target(target, phase)
                if bool(done[0]):
                    return
        residual = float(np.linalg.norm(goal - tcp_position()))
        raise RuntimeError(f"{phase} did not reach its waypoint; residual={residual:.4f} m")

    try:
        scenario = build_scenario(simulator=simulator, scene_root=APPLE_DELIVERY_SCENE_ROOT, image_size=image_size)
        if simulator == "mujoco":
            report["drive_gain_scales"] = {
                "arm": 5e-5,
                "head_and_torso_pitch": 2e-5,
                "torso_yaw": 1e-3,
                "gripper_leader": 1e-4,
                "source_asset_gains_preserved": True,
            }
        env = GenieSimG1AppleDeliveryTask(scenario=scenario, simulator=simulator, image_size=image_size)
        state, _ = env.reset()
        controller = G1CartesianController(
            env.scenario.robots[0], CONTROL_JOINTS, arm_side="left", allow_torso_motion=True
        )
        if set(state.cameras) != set(CAMERA_NAMES):
            raise RuntimeError(f"Unexpected cameras: {sorted(state.cameras)}")
        report["step_seconds"] = env.handler.env_step_s
        report["initial_apple_pose"] = apple_pose().tolist()
        report["initial_tcp_position_m"] = tcp_position().tolist()
        capture()

        # Estimate the grasp center from the actual prepared G1 finger links.
        q = controls()
        q_open = q.copy()
        q_open[CONTROL_JOINTS.index("idx41_gripper_l_outer_joint1")] = 0.78
        tcp_open = controller.tcp_pose(q_open)[0]
        centers = []
        for name in ("gripper_l_inner_link4", "gripper_l_outer_link4"):
            body = mujoco.mj_name2id(controller.model, mujoco.mjtObj.mjOBJ_BODY, name)
            local = controller.data.xpos[body].copy()
            centers.append(controller.base_position + controller.base_rotation.apply(local))
        tool_offset = np.mean(centers, axis=0) - tcp_open
        initial_apple = apple_pose()
        apple_center = initial_apple[:3] + np.array([0.0, 0.0, 0.04])
        grasp_tcp = apple_center - tool_offset
        pregrasp = grasp_tcp + np.array([0.0, 0.0, 0.11])
        report.update({
            "tool_offset_m": tool_offset.tolist(),
            "pregrasp_tcp_position_m": pregrasp.tolist(),
            "grasp_tcp_position_m": grasp_tcp.tolist(),
        })

        hold(0.95, 35, "open_gripper")
        move_to(pregrasp, 0.95, "approach")
        # Contact first occurs before the TCP reaches the nominal fingertip
        # center. Close at the first reachable contact height so the open
        # fingertip cannot continue nudging the apple across the tabletop.
        contact_tcp = grasp_tcp + np.array([0.0, 0.0, 0.065])
        report["contact_tcp_position_m"] = contact_tcp.tolist()
        move_to(contact_tcp, 0.95, "contact_approach")
        hold(0.05, 70, "close_gripper")
        report["apple_after_close_pose"] = apple_pose().tolist()
        move_to(grasp_tcp, 0.05, "descend_closed", max_chunks=120)
        hold(0.05, 35, "grasp_settle")
        move_to(grasp_tcp + np.array([0.0, 0.0, 0.14]), 0.05, "lift")
        apple_after_lift = apple_pose()
        report["apple_after_lift_pose"] = apple_after_lift.tolist()
        if apple_after_lift[2] < initial_apple[2] + 0.04:
            raise RuntimeError("Apple did not rise with the closed gripper; rejecting this rollout as an unsuccessful grasp")

        plate = state.objects["pink_plate"].root_state[0, :3].detach().cpu().numpy().copy()
        desired_apple_position = plate + np.array([0.0, 0.0, 0.04])
        place_tcp = tcp_position() + desired_apple_position - apple_after_lift[:3]
        report["desired_apple_position_on_plate"] = desired_apple_position.tolist()
        report["place_tcp_position_m"] = place_tcp.tolist()
        move_to(place_tcp + np.array([0.0, 0.0, 0.12]), 0.05, "transport")
        move_to(place_tcp, 0.05, "place")
        hold(0.95, 50, "release")
        hold(0.95, 90, "settle")
        report["final_apple_pose"] = apple_pose().tolist()
        report["success"] = bool(env._success[0].item())
        report["task_failure"] = bool(env._failure[0].item())
        report["outcome"] = "success" if report["success"] else "completed_without_success"
    except Exception as exc:
        report["outcome"] = "exception"
        report["exception"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        if env is not None and state is not None:
            try:
                report["last_apple_pose"] = apple_pose().tolist()
                report["last_tcp_position_m"] = tcp_position().tolist()
            except Exception:
                pass
    finally:
        for writer in writers.values():
            writer.release()
        if env is not None:
            report["steps"] = len(actions)
            report["camera_shapes"] = frame_shapes
            (output / "trajectory.json").write_text(
                json.dumps(
                    {
                        "format": "roboverse.icl.g1_apple_joint_demo.v1",
                        "simulator": simulator,
                        "robot": report["robot"],
                        "step_seconds": report.get("step_seconds"),
                        "control_joints": CONTROL_JOINTS,
                        "actions": actions,
                        "trace": trace,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            env.close()
        (output / "rollout_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", choices=("mujoco", "genesis", "newton"), default="mujoco")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="camera width for review videos (default: 640; model-input tasks may use 256)",
    )
    args = parser.parse_args()
    result = record_demo(simulator=args.sim, output=args.output, image_size=args.image_size)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result["outcome"] == "exception":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
