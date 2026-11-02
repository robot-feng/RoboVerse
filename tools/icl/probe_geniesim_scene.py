"""Render one imported Genie Sim G1 benchmark run through a RoboVerse backend."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from metasim.constants import PhysicStateType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.utils.setup_util import get_handler
from roboverse_pack.tasks.icl.fruit_sorting import build_geniesim_g1_robot
from roboverse_pack.tasks.icl.geniesim_scene import (
    build_run_scenario,
    geniesim_g1_cameras,
    resolve_geniesim_environment_hdr,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = ROOT / "roboverse_data/icl/geniesim_benchmark_specs/benchmark.json"
DEFAULT_ROBOT = ROOT / "roboverse_data/icl/robots/geniesim_g1_omnipicker_closed_v5"
DEFAULT_ASSETS = ROOT / "roboverse_data/icl/objects/geniesim"
DEFAULT_CACHE = ROOT / "roboverse_data/icl/cache/geniesim_scene_wrappers_v6"


def render_scene(
    *,
    simulator: str,
    run_id: str,
    output: Path,
    spec_path: Path = DEFAULT_SPEC,
    robot_root: Path = DEFAULT_ROBOT,
    assets_root: Path = DEFAULT_ASSETS,
    cache_root: Path = DEFAULT_CACHE,
    instance_index: int = 0,
    image_size: int = 640,
) -> dict:
    """Render source robot views and a clearly labeled external inspection view."""
    if simulator not in {"mujoco", "genesis", "newton"}:
        raise ValueError("Choose MuJoCo, Genesis or Newton")
    if image_size < 64:
        raise ValueError("image_size must be at least 64 pixels")
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing scene inspection: {output}")

    cameras = geniesim_g1_cameras(Path(robot_root), image_size=image_size)
    cameras.append(
        PinholeCameraCfg(
            name="inspection_overview",
            width=image_size,
            height=round(image_size * 9 / 16),
            data_types=["rgb"],
            pos=(2.4, -3.1, 2.3),
            look_at=(-0.25, 0.0, 0.95),
        )
    )
    scenario, run, instance, evaluation = build_run_scenario(
        spec_path=Path(spec_path),
        run_id=run_id,
        simulator=simulator,
        robot=build_geniesim_g1_robot(Path(robot_root)),
        assets_root=Path(assets_root),
        cache_root=Path(cache_root),
        instance_index=instance_index,
        cameras=cameras,
        headless=True,
    )

    handler = get_handler(scenario)
    try:
        robot_cfg = scenario.robots[0]
        initial_state = {
            "objects": {
                obj.name: {
                    "pos": torch.tensor(obj.default_position),
                    "rot": torch.tensor(obj.default_orientation),
                }
                for obj in scenario.objects
            },
            "robots": {
                robot_cfg.name: {
                    "pos": torch.tensor(robot_cfg.default_position),
                    "rot": torch.tensor(robot_cfg.default_orientation),
                    "dof_pos": dict(robot_cfg.default_joint_positions),
                }
            },
        }
        handler.set_states([initial_state])
        handler.refresh_render()
        state = handler.get_states(mode="tensor")
        names = ["head_rgb", "head_fisheye_left", "head_fisheye_right", "inspection_overview"]
        frames: dict[str, np.ndarray] = {}
        for name in names:
            frame = state.cameras[name].rgb[0].detach().cpu().numpy()
            if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
                raise ValueError(f"Camera {name} returned an invalid RGB frame: {frame.shape} {frame.dtype}")
            if not np.isfinite(frame).all() or float(frame.std()) < 1.0:
                raise RuntimeError(f"Camera {name} returned a blank or invalid RGB frame")
            frames[name] = frame

        report = {
            "source": "Genie Sim Benchmark",
            "run_id": run["id"],
            "source_subtask": run["subtask"],
            "instance": f"{instance['task']}/{instance['instance_id']}",
            "simulator": simulator,
            "robot": robot_cfg.name,
            "robot_position": list(robot_cfg.default_position),
            "robot_orientation_wxyz": list(robot_cfg.default_orientation),
            "robot_joint_positions": dict(robot_cfg.default_joint_positions),
            "objects": [
                {
                    "name": obj.name,
                    "position": list(obj.default_position),
                    "orientation_wxyz": list(obj.default_orientation),
                    "static": obj.physics == PhysicStateType.GEOM,
                }
                for obj in scenario.objects
            ],
            "evaluation_config": run["evaluation_config"],
            "source_scene_background": evaluation.get("scene", {}).get("scene_usd"),
            "rendered_scene_background": (
                "source dome HDR on a distant visual sphere"
                + (
                    " plus static room-plane meshes extracted from the source USD"
                    if any(obj.name == "geniesim_source_room_panels" for obj in scenario.objects)
                    else ""
                )
                + "; source NuRec/Gaussian-splat geometry is not loaded"
                if resolve_geniesim_environment_hdr(evaluation.get("scene", {}).get("scene_usd"))
                else "neutral gray studio enclosure; the source Gaussian-splat room is not loaded by these backends"
            ),
            "camera_frames": {
                name: {
                    "shape": list(frame.shape),
                    "pixel_std": float(frame.std()),
                    "position_world": state.cameras[name].pos[0].detach().cpu().tolist()
                    if state.cameras[name].pos is not None
                    else None,
                    "quaternion_world_wxyz": state.cameras[name].quat_world[0].detach().cpu().tolist()
                    if state.cameras[name].quat_world is not None
                    else None,
                }
                for name, frame in frames.items()
            },
            "fisheye_projection": "OpenCV fisheye; G1 mounts use borrowed G2 optical calibration",
            "inspection_overview": "external camera for scene inspection; not a source-calibrated camera",
        }
        with tempfile.TemporaryDirectory(prefix=f"{run_id}-", dir=output.parent) as temp_dir:
            staging = Path(temp_dir) / output.name
            staging.mkdir()
            for name, frame in frames.items():
                Image.fromarray(frame).save(staging / f"{simulator}_{name}.png")
            (staging / "scene_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            shutil.move(str(staging), output)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return report
    finally:
        handler.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", choices=("mujoco", "genesis", "newton"), required=True)
    parser.add_argument("--run-id", required=True, help="Imported G1 source run ID")
    parser.add_argument("--instance-index", type=int, default=0)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--robot-root", type=Path, default=DEFAULT_ROBOT)
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=640)
    args = parser.parse_args()
    render_scene(
        simulator=args.sim,
        run_id=args.run_id,
        instance_index=args.instance_index,
        spec_path=args.spec,
        robot_root=args.robot_root,
        assets_root=args.assets_root,
        cache_root=args.cache_root,
        output=args.output,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
