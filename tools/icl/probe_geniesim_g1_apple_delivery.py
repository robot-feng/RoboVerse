"""Record the G1 apple scene using G1 camera mounts and borrowed G2 fisheye optics."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from metasim.task.registry import get_task_class
from roboverse_pack.tasks.icl.fruit_sorting import (
    APPLE_DELIVERY_SCENE_ROOT,
    CONTROL_JOINTS,
    GenieSimG1AppleDeliveryTask,  # noqa: F401 — imports the registration module
)
from tools.icl.video_utils import RGBVideoWriter

_CAMERAS = ("head_rgb", "head_fisheye_left", "head_fisheye_right", "overview")


def record_scene(*, simulator: str, output: Path, image_size: int = 640, frames: int = 60) -> dict:
    """Save per-camera scene videos and a machine-readable smoke report.

    The probe holds the delivered robot pose, then pans the head briefly. It
    validates rendering, reset poses and scene stability; it does not claim
    apple pickup/place success or cross-engine physics parity.
    """
    if simulator not in {"mujoco", "genesis", "newton"}:
        raise ValueError("Choose mujoco, genesis or newton")
    if image_size < 64 or image_size % 2 or frames < 2:
        raise ValueError("image_size must be even and >=64; frames must be >=2")
    final_output = Path(output).resolve()
    if final_output.exists():
        raise FileExistsError(f"Choose an output directory without prior videos: {final_output}")
    final_output.parent.mkdir(parents=True, exist_ok=True)

    env = get_task_class("icl.geniesim_g1_apple_delivery")(
        simulator=simulator,
        image_size=image_size,
        headless=True,
    )
    temporary = tempfile.TemporaryDirectory(prefix="rvapplevid_", dir=final_output.parent)
    output = Path(temporary.name) / "record"
    writers: dict[str, RGBVideoWriter] = {}
    report_path = output / "scene_video_report.json"
    try:
        state, _ = env.reset()
        initial = {
            name: state.objects[name].root_state[0, :7].detach().cpu().tolist() for name in ("apple", "pink_plate")
        }
        initial_robot_root = state.robots["agibot_g1"].root_state[0, :7].detach().cpu().numpy().copy()
        initial_robot = state.robots["agibot_g1"].joint_pos[0].detach().cpu().numpy()
        controls = env.control_state(state)[0].detach().cpu().numpy().copy()
        head_yaw_name = "idx11_head_joint1"
        low, high = env.scenario.robots[0].joint_limits[head_yaw_name]
        pan_target = float(np.clip(controls[CONTROL_JOINTS.index(head_yaw_name)] + 0.18, low, high))
        held_command = controls.astype(np.float32)
        pan_command_values = controls.copy()
        pan_command_values[CONTROL_JOINTS.index(head_yaw_name)] = pan_target
        pan_command = pan_command_values.astype(np.float32)

        def write_state(current_state) -> None:
            nonlocal writers
            for camera_name in _CAMERAS:
                frame = current_state.cameras[camera_name].rgb[0].detach().cpu().numpy()
                if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
                    raise ValueError(f"Camera {camera_name} returned invalid RGB data: {frame.shape} {frame.dtype}")
                if not np.isfinite(frame).all() or float(frame.std()) < 1.0:
                    raise RuntimeError(f"Camera {camera_name} returned a blank frame")
                if camera_name not in writers:
                    writers[camera_name] = RGBVideoWriter(
                        output / f"{simulator}_{camera_name}.mp4", frame.shape[1], frame.shape[0]
                    )
                writers[camera_name].append(frame)

        output.mkdir(parents=True)
        write_state(state)
        for index in range(frames - 1):
            command = pan_command if frames // 4 <= index < 3 * frames // 4 else held_command
            state, _, _, timeout, info = env.step(command)
            if bool(timeout[0]) or bool(info.get("task_failure", [False])[0]):
                raise RuntimeError("The scene reached task failure/timeout during the render probe")
            write_state(state)
        final = {
            name: state.objects[name].root_state[0, :7].detach().cpu().tolist() for name in ("apple", "pink_plate")
        }
        final_robot_root = state.robots["agibot_g1"].root_state[0, :7].detach().cpu().numpy().copy()
        for writer in writers.values():
            writer.close()
        writers.clear()

        scene_json = env._metadata
        report = {
            "format": "roboverse.icl.scene_video_smoke.v1",
            "task": "icl.geniesim_g1_apple_delivery",
            "scene_id": scene_json["scene_id"],
            "archive_sha256": scene_json["source"]["archive_sha256"],
            "prepared_scene": str(APPLE_DELIVERY_SCENE_ROOT.resolve()),
            "scene_metadata_sha256": hashlib.sha256(
                (APPLE_DELIVERY_SCENE_ROOT / "scene.json").read_bytes()
            ).hexdigest(),
            "simulator": simulator,
            "output_directory": str(final_output),
            "render_settings": {
                "pink_plate_urdf": next(obj.urdf_path for obj in env.scenario.objects if obj.name == "pink_plate"),
                "newton_use_scenario_lights": bool(env.scenario.sim_params.newton_use_scenario_lights),
                "newton_camera_background_rgb": env.scenario.sim_params.newton_camera_background_rgb,
            },
            "capture": {
                "fps": 30,
                "frames": frames,
                "camera_image_size_argument": image_size,
                "camera_frames": {name: list(state.cameras[name].rgb.shape[1:]) for name in _CAMERAS},
                "video_encoded_dimensions": {
                    name: [
                        int(state.cameras[name].rgb.shape[2]) + int(state.cameras[name].rgb.shape[2]) % 2,
                        int(state.cameras[name].rgb.shape[1]) + int(state.cameras[name].rgb.shape[1]) % 2,
                    ]
                    for name in _CAMERAS
                },
                "video_files": [f"{simulator}_{name}.mp4" for name in _CAMERAS],
            },
            "camera_models": [
                {
                    "name": camera.name,
                    "projection": type(camera).__name__,
                    "calibration_source": getattr(camera, "calibration_source", None),
                    "calibration_source_sha256": getattr(camera, "calibration_source_sha256", None),
                    "max_fov_source": getattr(camera, "max_fov_source", None),
                }
                for camera in env.scenario.cameras
            ],
            "initial_objects_root_state": initial,
            "final_objects_root_state": final,
            "robot_root_state": {
                "initial_xyz_wxyz": initial_robot_root.tolist(),
                "final_xyz_wxyz": final_robot_root.tolist(),
                "max_translation_during_probe_m": float(np.linalg.norm(final_robot_root[:3] - initial_robot_root[:3])),
                "source_pose_preserved": bool(np.allclose(initial_robot_root[:7], final_robot_root[:7], atol=1e-6)),
            },
            "base_collision_ground_alignment": json.loads(
                (Path(env.scenario.robots[0].urdf_path).parent / "provenance.json").read_text(encoding="utf-8")
            ).get("collision_ground_alignment"),
            "initial_robot_joint_state_finite": bool(np.isfinite(initial_robot).all()),
            "task_success_claimed": False,
            "validation_scope": "rendered scene / configured start pose / short stable hold only; not task execution parity",
        }
        encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        report_path.write_text(encoded, encoding="utf-8")
        report["report_sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        if final_output.exists():
            raise FileExistsError(f"Output directory appeared during recording: {final_output}")
        shutil.move(str(output), str(final_output))
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return report
    finally:
        for writer in writers.values():
            writer.close()
        try:
            env.close()
        finally:
            temporary.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", choices=("mujoco", "genesis", "newton"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=640, help="source camera width; 640 produces 640x400 RGB")
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args()
    record_scene(simulator=args.sim, output=args.output, image_size=args.image_size, frames=args.frames)


if __name__ == "__main__":
    main()
