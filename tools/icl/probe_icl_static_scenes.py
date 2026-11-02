"""Reset and render RoboVerse ICL static object scenes on one backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from metasim.utils.hf_util import LOCAL_DIR
from roboverse_pack.tasks.icl.icl_static_scenes import (
    DEFAULT_ROBOT_PROFILE,
    ICL_STATIC_SCENE_IDS,
    SUPPORTED_SIMULATORS,
    build_icl_static_scenario,
    get_icl_static_scene_spec,
)

STATIC_VIDEO_FPS = 30
STATIC_VIDEO_FRAMES = 30


def _initial_state(scenario) -> dict:
    """Create one reset state from the public ScenarioCfg defaults."""
    import torch

    robot_states = {}
    for robot in scenario.robots:
        robot_states[robot.name] = {
            "pos": torch.tensor(robot.default_position),
            "rot": torch.tensor(robot.default_orientation),
            "dof_pos": dict(robot.default_joint_positions),
        }
    object_states = {
        obj.name: {
            "pos": torch.tensor(obj.default_position),
            "rot": torch.tensor(obj.default_orientation),
        }
        for obj in scenario.objects
    }
    return {"robots": robot_states, "objects": object_states}


def _asset_provenance(asset_root: Path, asset_ids: list[str]) -> list[dict]:
    records = []
    for asset_id in asset_ids:
        path = asset_root / asset_id / "provenance.json"
        info = json.loads(path.read_text(encoding="utf-8"))
        records.append(
            {
                "asset_id": asset_id,
                "source_repository": info.get("source_repository"),
                "source_revision": info.get("source_revision"),
                "license": info.get("license") or "not specified in local provenance",
                "source_files_sha256": info.get("source_files_sha256") or {},
                "source_manifest_sha256": info.get("source_manifest_sha256"),
                "hashes_verified_by_probe": False,
                "hashes_note": "Hashes are copied from provenance.json; this probe does not recompute them.",
            }
        )
    return records


def _roboverse_git_state() -> dict[str, str | bool | list[str] | None]:
    """Return commit and working-tree state, including reasons when Git is unavailable."""
    repo_root = Path(__file__).resolve().parents[2]
    state: dict[str, str | bool | list[str] | None] = {
        "head": None,
        "head_reason": None,
        "dirty": None,
        "status_porcelain": None,
        "status_porcelain_sha256": None,
        "status_reason": None,
    }
    try:
        head_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        reason = f"{type(exc).__name__}: {exc}"
        state["head_reason"] = reason
        state["status_reason"] = reason
        return state
    if head_result.returncode == 0 and head_result.stdout.strip():
        state["head"] = head_result.stdout.strip()
    else:
        state["head_reason"] = head_result.stderr.strip() or f"git rev-parse exited with {head_result.returncode}"

    try:
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        state["status_reason"] = f"{type(exc).__name__}: {exc}"
        return state
    if status_result.returncode != 0:
        stderr = status_result.stderr.decode("utf-8", errors="replace").strip()
        state["status_reason"] = stderr or f"git status exited with {status_result.returncode}"
        return state
    status_output = status_result.stdout
    status_lines = status_output.decode("utf-8", errors="replace").splitlines()
    state["dirty"] = bool(status_lines)
    state["status_porcelain"] = status_lines
    state["status_porcelain_sha256"] = hashlib.sha256(status_output).hexdigest()
    return state


def _write_static_video(video_path: Path, frame: np.ndarray) -> dict[str, int | str | float]:
    """Encode one second of the same reset RGB frame using RoboVerse's shared writer."""
    from tools.icl.video_utils import RGBVideoWriter

    height, width = frame.shape[:2]
    writer = RGBVideoWriter(video_path, width=width, height=height, fps=STATIC_VIDEO_FPS)
    try:
        for _ in range(STATIC_VIDEO_FRAMES):
            writer.append(frame)
    finally:
        writer.close()
    encoded_width = width + width % 2
    encoded_height = height + height % 2
    file_size_bytes = video_path.stat().st_size
    return {
        "path": video_path.name,
        "width": width,
        "height": height,
        "encoded_width": encoded_width,
        "encoded_height": encoded_height,
        "fps": STATIC_VIDEO_FPS,
        "frames": STATIC_VIDEO_FRAMES,
        "duration_seconds": STATIC_VIDEO_FRAMES / STATIC_VIDEO_FPS,
        "file_size_bytes": file_size_bytes,
        "sha256": _sha256_file(video_path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_scene(
    *,
    simulator: str,
    scene_id: str,
    output: Path,
    robot_profile: str = DEFAULT_ROBOT_PROFILE,
    asset_root: Path | None = None,
    robot_root: Path | None = None,
    image_size: int = 640,
    camera_names: tuple[str, ...] | None = None,
    headless: bool = True,
) -> Path:
    """Reset a scene, render its selected cameras, and save PNG/MP4 files plus a report.

    Each MP4 repeats the same reset frame for one second; it is a static scene
    inspection video, not a simulation rollout. No physics step, action,
    trajectory generation, or task-success check is performed. A failed camera
    or backend launch is recorded before the error is re-raised.
    """
    if simulator not in SUPPORTED_SIMULATORS:
        raise ValueError(f"Unsupported simulator {simulator!r}; choose from {SUPPORTED_SIMULATORS}")
    if robot_profile not in ("g1", "g2", "franka"):
        raise ValueError("robot_profile must be 'g1', 'g2', or 'franka'")
    get_icl_static_scene_spec(scene_id)

    selected_asset_root = Path(asset_root) if asset_root is not None else Path(LOCAL_DIR) / "icl" / "objects"
    scene_dir = Path(output) / simulator / robot_profile / scene_id
    scene_dir.mkdir(parents=True, exist_ok=False)
    report_path = scene_dir / "report.json"
    roboverse_git = _roboverse_git_state()
    report: dict = {
        "format": "roboverse.icl.static_scene_probe.v1",
        "status": "starting",
        "simulator": simulator,
        "scene_id": scene_id,
        "robot_profile": robot_profile,
        "asset_root": str(selected_asset_root.resolve()),
        "image_size_requested_width": image_size,
        "reset_method": "set_states_from_scenario_defaults",
        "physics_stepped": False,
        "task_success_checked": False,
        "roboverse_git_head": roboverse_git["head"],
        "roboverse_git_head_reason": roboverse_git["head_reason"],
        "roboverse_git_dirty": roboverse_git["dirty"],
        "roboverse_git_status_porcelain": roboverse_git["status_porcelain"],
        "roboverse_git_status_porcelain_sha256": roboverse_git["status_porcelain_sha256"],
        "roboverse_git_status_reason": roboverse_git["status_reason"],
        "video_semantics": "static reset frame repeated; no physics stepped",
        "video_fps": STATIC_VIDEO_FPS,
        "video_frames_per_camera": STATIC_VIDEO_FRAMES,
        "video_duration_seconds": STATIC_VIDEO_FRAMES / STATIC_VIDEO_FPS,
        "camera_frames": {},
    }
    handler = None
    error: Exception | None = None
    try:
        from metasim.utils.setup_util import get_handler

        scenario = build_icl_static_scenario(
            simulator=simulator,
            scene_id=scene_id,
            asset_root=selected_asset_root,
            robot_profile=robot_profile,
            robot_root=robot_root,
            image_size=image_size,
            camera_names=camera_names,
            headless=headless,
        )
        object_asset_ids = [
            Path(obj.urdf_path).parent.name for obj in scenario.objects if getattr(obj, "urdf_path", None)
        ]
        report["objects"] = [obj.name for obj in scenario.objects]
        report["robot_reset_state"] = {
            "position": [float(value) for value in scenario.robots[0].default_position],
            "orientation_wxyz": [float(value) for value in scenario.robots[0].default_orientation],
            "joint_positions": {
                name: float(value) for name, value in scenario.robots[0].default_joint_positions.items()
            },
        }
        report["object_reset_states"] = {
            obj.name: {
                "position": [float(value) for value in obj.default_position],
                "orientation_wxyz": [float(value) for value in obj.default_orientation],
            }
            for obj in scenario.objects
        }
        report["asset_provenance"] = _asset_provenance(selected_asset_root, object_asset_ids)
        report["camera_names_requested"] = [camera.name for camera in scenario.cameras]
        report["camera_dimensions"] = {
            camera.name: [camera.height, camera.width] for camera in scenario.cameras
        }

        handler = get_handler(scenario)
        handler.set_states([_initial_state(scenario)])
        handler.refresh_render()
        state = handler.get_states(mode="tensor")
        if not state.cameras:
            raise RuntimeError(f"{simulator} returned no camera frames")
        missing_cameras = set(report["camera_names_requested"]) - set(state.cameras)
        if missing_cameras:
            raise RuntimeError(f"{simulator} returned no frames for cameras: {sorted(missing_cameras)}")
        for camera_name in report["camera_names_requested"]:
            frame = state.cameras[camera_name].rgb[0].detach().cpu().numpy()
            if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
                raise ValueError(f"Invalid RGB frame for {camera_name}: shape={frame.shape}, dtype={frame.dtype}")
            pixel_std = float(frame.std())
            finite = bool(np.isfinite(frame).all())
            if not finite or not np.isfinite(pixel_std) or pixel_std < 1.0:
                raise ValueError(
                    f"Invalid RGB content for {camera_name}: finite={finite}, "
                    f"pixel_std={pixel_std:.3f} (expected >= 1.0)"
                )
            image_path = scene_dir / f"{camera_name}.png"
            Image.fromarray(frame).save(image_path)
            video_path = scene_dir / f"{camera_name}.mp4"
            video = _write_static_video(video_path, frame)
            report["camera_frames"][camera_name] = {
                "path": image_path.name,
                "shape": list(frame.shape),
                "dtype": str(frame.dtype),
                "pixel_std": pixel_std,
                "video": video,
            }
        report["status"] = "ok"
    except Exception as exc:
        error = exc
        report["status"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if handler is not None:
            try:
                handler.close()
            except Exception as exc:
                report["close_error"] = f"{type(exc).__name__}: {exc}"
                if error is None:
                    error = exc
                    report["status"] = "error"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if error is not None:
        raise RuntimeError(f"Static scene probe failed; see {report_path}: {error}") from error
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", choices=SUPPORTED_SIMULATORS, required=True)
    parser.add_argument("--scene", choices=(*ICL_STATIC_SCENE_IDS, "all"), default="all")
    parser.add_argument("--robot", choices=("g1", "g2", "franka", "all"), default=DEFAULT_ROBOT_PROFILE)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument(
        "--robot-root", type=Path, help="Override the selected robot profile's asset directory; omit with --robot all"
    )
    parser.add_argument("--camera-names", help="Optional comma-separated camera selection")
    parser.add_argument("--image-size", type=int, default=640, help="Requested camera width in pixels")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-headless", action="store_true", help="Show the simulator window")
    args = parser.parse_args()
    camera_names = None
    if args.camera_names:
        camera_names = tuple(name.strip() for name in args.camera_names.split(",") if name.strip())
    scenes = ICL_STATIC_SCENE_IDS if args.scene == "all" else (args.scene,)
    robot_profiles = ("g1", "g2") if args.robot == "all" else (args.robot,)
    if args.robot == "all" and args.robot_root is not None:
        parser.error("--robot-root selects one robot asset directory and cannot be combined with --robot all")
    failures = []
    reports = []
    for robot_profile in robot_profiles:
        for scene_id in scenes:
            try:
                report_path = probe_scene(
                    simulator=args.sim,
                    scene_id=scene_id,
                    output=args.output,
                    robot_profile=robot_profile,
                    asset_root=args.asset_root,
                    robot_root=args.robot_root,
                    image_size=args.image_size,
                    camera_names=camera_names,
                    headless=not args.no_headless,
                )
                reports.append(str(report_path))
            except Exception as exc:
                failures.append(f"{robot_profile}/{scene_id}: {exc}")
    print(json.dumps({"reports": reports, "failures": failures}, indent=2, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
