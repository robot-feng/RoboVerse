"""Safety tests for the preview-only Phase 1 gallery publisher."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

import tools.icl.publish_phase1_preview as preview_publisher
from tools.icl.publish_phase1_preview import PreviewPublishError, publish
from tools.icl.video_utils import RGBVideoWriter, write_preview_gif


def _png(width: int = 64, height: int = 48) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR" + struct.pack(">II", width, height)


def _record(path: Path, data: bytes, **metadata) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "path": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        **metadata,
    }


def _candidate(qa_root: Path, asset_root: Path, *, checks: dict | None = None) -> Path:
    run_root = qa_root / "unit_preview_case"
    robot_asset = asset_root / "icl" / "robot.urdf"
    object_asset = asset_root / "icl" / "apple.obj"
    robot_asset.parent.mkdir(parents=True, exist_ok=True)
    robot_asset.write_text("robot", encoding="utf-8")
    object_asset.write_text("apple", encoding="utf-8")
    combined = _record(run_root / "overview.png", _png(), width=64, height=48)
    run_root.mkdir(parents=True, exist_ok=True)
    video = _record(
        run_root / "head_rgb.mp4",
        b"\x00\x00\x00\x18ftypisom" + b"unit-test-video-payload",
        role="head_rgb",
        width=128,
        height=96,
        decoded_frames=3,
    )
    image = _record(run_root / "scene.png", _png(96, 64), role="scene", width=96, height=64)
    evidence = _record(run_root / "runtime_report.json", b'{"status":"verified"}')
    candidate = {
        "format": "roboverse.icl.phase1_environment_candidate.v1",
        "environment_id": "synthicl_pick_and_place_g1",
        "source": "SynthICL pseudo-demo",
        "upstream_revision": "1" * 40,
        "source_task_id": "pick_and_place_gn",
        "source_config": "configs/all_objects_skills.yaml",
        "task_variant": "default",
        "scene_id": "pick_and_place_gn",
        "robot_id": "G1",
        "migrated_task_id": "icl.synthicl.pick_and_place_gn",
        "roboverse_commit": "a" * 40,
        "required_backends": ["mujoco"],
        "combined_overview": {"path": "overview.png", "sha256": combined["sha256"], "width": 64, "height": 48},
        "runs": [
            {
                "simulator": "mujoco",
                "status": "VERIFIED",
                "seed": 3,
                "simulator_version": "3.15.0",
                "roboverse_commit": "a" * 40,
                "asset_hashes": {
                    "icl/robot.urdf": hashlib.sha256(robot_asset.read_bytes()).hexdigest(),
                    "icl/apple.obj": hashlib.sha256(object_asset.read_bytes()).hexdigest(),
                },
                "checks": checks
                or {
                    "asset_resolution": True,
                    "create": True,
                    "repeated_reset": True,
                    "step": True,
                    "finite_state": True,
                    "task_predicate_positive": True,
                    "task_predicate_negative": True,
                    "viewer_interactive": True,
                    "rendered_preview": True,
                },
                "evidence_files": [evidence],
                "previews": [video, image],
            }
        ],
    }
    path = run_root / "phase1_candidate.json"
    path.write_text(json.dumps(candidate), encoding="utf-8")
    return path


def _mock_gif_writer(monkeypatch):
    def write_gif(source: Path, target: Path, **kwargs):
        assert source.is_file()
        target.write_bytes(b"GIF89a")
        return 3, 96, 64, 15.0

    monkeypatch.setattr(preview_publisher, "write_preview_gif", write_gif)


def test_preview_gif_is_small_and_animated(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    np = pytest.importorskip("numpy")
    pytest.importorskip("av")
    source = tmp_path / "overview.mp4"
    writer = RGBVideoWriter(source, width=32, height=24, fps=30)
    try:
        for value in (0, 100, 220):
            frame = np.full((24, 32, 3), value, dtype=np.uint8)
            writer.append(frame)
    finally:
        writer.close()

    target = tmp_path / "overview.gif"
    frames, width, height, fps = write_preview_gif(source, target, max_size=20, target_fps=15)

    assert frames == 2
    assert (width, height) == (20, 15)
    assert fps == 15
    with image_module.open(target) as gif:
        assert gif.n_frames == 2
        assert gif.info["loop"] == 0


def test_phase1_publisher_copies_only_preview_media_and_readme(tmp_path, monkeypatch):
    _mock_gif_writer(monkeypatch)
    qa_root = tmp_path / "qa_runs"
    asset_root = tmp_path / "roboverse_data"
    candidate_path = _candidate(qa_root, asset_root)
    output_root = tmp_path / "ICL_Benchmark" / "environments"

    destination = publish(candidate_path, qa_root=qa_root, output_root=output_root, asset_root=asset_root)

    assert destination == output_root / "synthicl_pick_and_place_g1"
    files = {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()}
    assert files == {
        "README.md",
        "overview_first_frames.png",
        "mujoco/mujoco_head_rgb.gif",
        "mujoco/mujoco_head_rgb.mp4",
        "mujoco/mujoco_scene.png",
    }
    readme = (destination / "README.md").read_text(encoding="utf-8")
    assert "Phase 1 environment: VERIFIED" in readme
    assert "task predicate positive/negative cases" in readme
    assert "do not claim a successful manipulation rollout" in readme
    assert "![mujoco scene preview](mujoco/mujoco_head_rgb.gif)" in readme
    assert "[head_rgb MP4](mujoco/mujoco_head_rgb.mp4)" in readme
    assert "examples/output/qa_runs/unit_preview_case/phase1_candidate.json" in readme


def test_phase1_publisher_rejects_incomplete_scene_checks_without_output(tmp_path):
    failed_checks = {
        "asset_resolution": True,
        "create": True,
        "repeated_reset": True,
        "step": True,
        "finite_state": True,
        "task_predicate_positive": True,
        "task_predicate_negative": True,
        "viewer_interactive": True,
        "rendered_preview": False,
    }
    asset_root = tmp_path / "roboverse_data"
    candidate_path = _candidate(tmp_path / "qa_runs", asset_root, checks=failed_checks)
    output_root = tmp_path / "gallery"

    with pytest.raises(PreviewPublishError, match="rendered_preview"):
        publish(candidate_path, qa_root=tmp_path / "qa_runs", output_root=output_root, asset_root=asset_root)

    assert not output_root.exists()


@pytest.mark.parametrize("failed_case", ["task_predicate_positive", "task_predicate_negative"])
def test_phase1_publisher_requires_both_task_predicate_cases(tmp_path, failed_case):
    checks = {
        "asset_resolution": True,
        "create": True,
        "repeated_reset": True,
        "step": True,
        "finite_state": True,
        "task_predicate_positive": True,
        "task_predicate_negative": True,
        "viewer_interactive": True,
        "rendered_preview": True,
    }
    checks[failed_case] = False
    asset_root = tmp_path / "roboverse_data"
    candidate_path = _candidate(tmp_path / "qa_runs", asset_root, checks=checks)
    output_root = tmp_path / "gallery"

    with pytest.raises(PreviewPublishError, match=failed_case):
        publish(candidate_path, qa_root=tmp_path / "qa_runs", output_root=output_root, asset_root=asset_root)

    assert not output_root.exists()


def test_phase1_publisher_rejects_modified_preview_and_path_escape(tmp_path):
    qa_root = tmp_path / "qa_runs"
    asset_root = tmp_path / "roboverse_data"
    candidate_path = _candidate(qa_root, asset_root)
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["runs"][0]["previews"][0]["path"] = "../outside.mp4"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    with pytest.raises(PreviewPublishError, match="unsafe relative path"):
        publish(candidate_path, qa_root=qa_root, output_root=tmp_path / "gallery", asset_root=asset_root)

    candidate_path = _candidate(qa_root, asset_root)
    (candidate_path.parent / "head_rgb.mp4").write_bytes(b"mutated")
    with pytest.raises(PreviewPublishError, match="SHA-256"):
        publish(candidate_path, qa_root=qa_root, output_root=tmp_path / "gallery", asset_root=asset_root)


def test_phase1_publisher_never_overwrites_a_reviewed_environment(tmp_path, monkeypatch):
    _mock_gif_writer(monkeypatch)
    qa_root = tmp_path / "qa_runs"
    asset_root = tmp_path / "roboverse_data"
    candidate_path = _candidate(qa_root, asset_root)
    output_root = tmp_path / "gallery"
    destination = publish(candidate_path, qa_root=qa_root, output_root=output_root, asset_root=asset_root)
    marker = destination / "reviewer_note.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(PreviewPublishError, match="refusing to overwrite"):
        publish(candidate_path, qa_root=qa_root, output_root=output_root, asset_root=asset_root)

    assert marker.read_text(encoding="utf-8") == "keep"


def test_phase1_publisher_cleans_staging_directory_if_gif_conversion_fails(tmp_path, monkeypatch):
    def fail_gif(source: Path, target: Path, **kwargs):
        raise ValueError("invalid video stream")

    monkeypatch.setattr(preview_publisher, "write_preview_gif", fail_gif)
    qa_root = tmp_path / "qa_runs"
    asset_root = tmp_path / "roboverse_data"
    candidate_path = _candidate(qa_root, asset_root)
    output_root = tmp_path / "gallery"

    with pytest.raises(PreviewPublishError, match="could not make mujoco Markdown GIF"):
        publish(candidate_path, qa_root=qa_root, output_root=output_root, asset_root=asset_root)

    assert list(output_root.iterdir()) == []


def test_phase1_publisher_requires_every_declared_backend(tmp_path):
    qa_root = tmp_path / "qa_runs"
    asset_root = tmp_path / "roboverse_data"
    candidate_path = _candidate(qa_root, asset_root)
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["required_backends"] = ["mujoco", "genesis", "newton"]
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    with pytest.raises(PreviewPublishError, match="missing: genesis, newton"):
        publish(candidate_path, qa_root=qa_root, output_root=tmp_path / "gallery", asset_root=asset_root)


def test_phase1_publisher_requires_qa_evidence_file(tmp_path):
    qa_root = tmp_path / "qa_runs"
    asset_root = tmp_path / "roboverse_data"
    candidate_path = _candidate(qa_root, asset_root)
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["runs"][0].pop("evidence_files")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    with pytest.raises(PreviewPublishError, match="QA evidence file"):
        publish(candidate_path, qa_root=qa_root, output_root=tmp_path / "gallery", asset_root=asset_root)
