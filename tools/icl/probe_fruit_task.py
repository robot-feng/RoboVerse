"""Inspect a fruit-sorting task's reset, mounted camera and physical rollout."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from metasim.task.registry import get_task_class
from roboverse_pack.tasks.icl.control import G1CartesianController
from roboverse_pack.tasks.icl.fruit_sorting import CONTROL_JOINTS


def probe(*, simulator: str, output: Path, image_size: int, steps: int, delta_x: float) -> None:
    """Run a reviewable short rollout; it is not a successful manipulation demonstration."""
    if not 1 <= steps <= 600 or not np.isfinite(delta_x) or abs(delta_x) > 0.021:
        raise ValueError("Choose 1-600 steps and a delta_x within +/-0.021 m")
    output.mkdir(parents=True, exist_ok=True)
    view_names = ("overview", "head", "head_fisheye_left", "head_fisheye_right", "head_turned")
    paths = {name: output / f"{simulator}_{name}.png" for name in view_names}
    summary = output / f"{simulator}_result.json"
    if summary.exists() or any(path.exists() for path in paths.values()):
        raise FileExistsError("Choose an output directory without results for this simulator")
    env = get_task_class("icl.fruit_sorting")(simulator=simulator, image_size=image_size, headless=True)
    try:
        state, _ = env.reset()
        initial = state.objects["peach_target"].root_state[0, :3].detach().cpu().numpy().copy()
        control = env.control_state(state)[0].detach().cpu().numpy().copy()
        controller = G1CartesianController(env.scenario.robots[0], CONTROL_JOINTS)
        tcp_start, _ = controller.tcp_pose(control)
        original_head = state.cameras["head_rgb"].rgb[0].detach().cpu().numpy().copy()
        fisheye_frames = {
            name: state.cameras[name].rgb[0].detach().cpu().numpy().copy()
            for name in ("head_fisheye_left", "head_fisheye_right")
        }
        overview = state.cameras["overview"].rgb[0].detach().cpu().numpy().copy()
        moved_state = copy.deepcopy(env._initial_states)
        moved_state[0]["robots"]["agibot_g1"]["dof_pos"]["idx11_head_joint1"] = 0.3
        env.handler.set_states(moved_state)
        env.handler.refresh_render()
        turned = env.handler.get_states(mode="tensor").cameras["head_rgb"].rgb[0].detach().cpu().numpy().copy()
        changed_pixels = int(np.count_nonzero(np.any(original_head != turned, axis=-1)))
        if changed_pixels == 0:
            raise RuntimeError("Head camera did not follow the changed joint position")
        state, _ = env.reset()
        action = np.array([delta_x, 0, 0, 0, 0, 0, 0.0])
        target = controller.joint_targets(control, action) if delta_x else control
        target[CONTROL_JOINTS.index("idx81_gripper_r_outer_joint1")] = 0.3
        command = target.astype(np.float32)
        for _ in range(steps):
            state, _, success, timeout, info = env.step(command)
            if bool(timeout[0]):
                raise RuntimeError(f"Task ended during inspection: failure={bool(info['task_failure'][0])}")
        final_control = env.control_state(state)[0].detach().cpu().numpy()
        tcp_final, _ = controller.tcp_pose(final_control)
        names = [name.split("/")[-1] for name in env.handler.get_joint_names("agibot_g1", sort=True)]
        q = state.robots["agibot_g1"].joint_pos[0].detach().cpu().numpy()
        leader = float(q[names.index("idx81_gripper_r_outer_joint1")])
        follower = float(q[names.index("idx71_gripper_r_inner_joint1")])
        final = state.objects["peach_target"].root_state[0, :3].detach().cpu().numpy()
        if not np.isfinite(final_control).all() or not np.isfinite(final).all():
            raise RuntimeError("Non-finite robot or object state")
        frames = {"overview": overview, "head": original_head, **fisheye_frames, "head_turned": turned}
        for key, image in frames.items():
            Image.fromarray(image).save(paths[key])
        result = {
            "task": "icl.fruit_sorting",
            "simulator": simulator,
            "steps": steps,
            "camera": "mounted_head_rgb",
            "head_changed_pixels": changed_pixels,
            "camera_frames": {
                name: {"shape": list(image.shape), "pixel_std": float(image.std())}
                for name, image in {"head_rgb": original_head, **fisheye_frames}.items()
            },
            "fisheye_projection": "opencv_fisheye_borrowed_g2",
            "initial_peach_position": initial.tolist(),
            "final_peach_position": final.tolist(),
            "right_gripper_leader": leader,
            "right_gripper_follower": follower,
            "tcp_delta_m": (tcp_final - tcp_start).tolist(),
            "requested_delta_x_m": delta_x,
            "success": bool(success[0]),
            "failure": bool(info["task_failure"][0]),
            "note": "Short diagnostic rollout; not a physical task-completion demonstration.",
        }
        summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result), flush=True)
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", choices=("mujoco", "genesis", "newton"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="camera width for visual inspection (default: 640; model-input tasks may use 256)",
    )
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--delta-x", type=float, default=0.0)
    args = parser.parse_args()
    probe(simulator=args.sim, output=args.output, image_size=args.image_size, steps=args.steps, delta_x=args.delta_x)


if __name__ == "__main__":
    main()
