"""Tests for the one-command Phase 1 manifest-to-gallery workflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.icl.publish_phase1_environment import publish_manifest
from tools.icl.publish_phase1_preview import REQUIRED_CHECKS, PreviewPublishError
from tools.icl.publish_phase1_preview import publish as publish_candidate
from tools.icl.video_utils import RGBVideoWriter


def _write_png(path: Path, color: tuple[int, int, int]) -> None:
    image_module = pytest.importorskip("PIL.Image")
    path.parent.mkdir(parents=True, exist_ok=True)
    image_module.new("RGB", (96, 72), color=color).save(path)


def _phase1_manifest(qa_root: Path, asset_root: Path, *, failed_check: str | None = None) -> Path:
    run_root = qa_root / "manifest_case"
    backend_root = run_root / "mujoco"
    assets = asset_root / "icl" / "robot.urdf"
    assets.parent.mkdir(parents=True, exist_ok=True)
    assets.write_text("robot asset", encoding="utf-8")
    _write_png(backend_root / "overview.png", (40, 80, 120))
    (backend_root / "runtime_report.json").write_text('{"status":"ok"}\n', encoding="utf-8")
    checks = {name: True for name in REQUIRED_CHECKS}
    if failed_check is not None:
        checks[failed_check] = False
    manifest = {
        "format": "roboverse.icl.phase1_environment_manifest.v1",
        "environment_id": "unit_manifest_case",
        "source": "unit test",
        "upstream_revision": "test-revision",
        "source_task_id": "pick",
        "source_config": "config.yaml",
        "task_variant": "default",
        "scene_id": "scene_a",
        "robot_id": "G1",
        "migrated_task_id": "icl.unit_task",
        "roboverse_commit": "a" * 40,
        "required_backends": ["mujoco"],
        "runs": [
            {
                "simulator": "mujoco",
                "simulator_version": "3.15.0",
                "seed": 42,
                "checks": checks,
                "asset_paths": ["icl/robot.urdf"],
                "evidence_paths": ["mujoco/runtime_report.json"],
                "previews": [{"role": "overview", "path": "mujoco/overview.png"}],
            }
        ],
    }
    path = run_root / "phase1_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def test_one_command_manifest_publishes_gallery_and_keeps_evidence_in_qa(tmp_path):
    qa_root = tmp_path / "qa_runs"
    asset_root = tmp_path / "roboverse_data"
    output_root = tmp_path / "ICL_Benchmark" / "environments"
    manifest = _phase1_manifest(qa_root, asset_root)

    candidate_path, destination = publish_manifest(
        manifest,
        qa_root=qa_root,
        asset_root=asset_root,
        output_root=output_root,
    )

    assert candidate_path.is_file()
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    assert candidate["source_manifest"]["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert (
        candidate["runs"][0]["asset_hashes"]["icl/robot.urdf"]
        == hashlib.sha256((asset_root / "icl" / "robot.urdf").read_bytes()).hexdigest()
    )
    assert candidate["runs"][0]["evidence_files"][0]["path"] == "mujoco/runtime_report.json"
    assert destination == output_root / "unit_manifest_case"
    published_files = {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()}
    assert published_files == {"README.md", "overview_first_frames.png", "mujoco/mujoco_overview.png"}
    assert not (destination / "runtime_report.json").exists()
    readme = (destination / "README.md").read_text(encoding="utf-8")
    assert "mujoco/runtime_report.json" in readme
    assert "phase1_manifest.json" in readme


def test_manifest_gate_rejects_failed_check_before_publishing(tmp_path):
    qa_root = tmp_path / "qa_runs"
    asset_root = tmp_path / "roboverse_data"
    output_root = tmp_path / "gallery"
    manifest = _phase1_manifest(qa_root, asset_root, failed_check="task_predicate_negative")

    with pytest.raises(PreviewPublishError, match="task_predicate_negative"):
        publish_manifest(manifest, qa_root=qa_root, asset_root=asset_root, output_root=output_root)

    assert not output_root.exists()


def test_manifest_rejects_preview_path_escape(tmp_path):
    qa_root = tmp_path / "qa_runs"
    asset_root = tmp_path / "roboverse_data"
    manifest_path = _phase1_manifest(qa_root, asset_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runs"][0]["previews"][0]["path"] = "../../outside.png"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PreviewPublishError, match="unsafe relative path"):
        publish_manifest(
            manifest_path,
            qa_root=qa_root,
            asset_root=asset_root,
            output_root=tmp_path / "gallery",
        )


def test_candidate_is_bound_to_unchanged_source_manifest(tmp_path):
    qa_root = tmp_path / "qa_runs"
    asset_root = tmp_path / "roboverse_data"
    manifest = _phase1_manifest(qa_root, asset_root)
    candidate, _ = publish_manifest(
        manifest,
        qa_root=qa_root,
        asset_root=asset_root,
        output_root=tmp_path / "gallery",
    )
    manifest.write_text(manifest.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(PreviewPublishError, match=r"source_manifest.*SHA-256"):
        publish_candidate(
            candidate,
            qa_root=qa_root,
            output_root=tmp_path / "another_gallery",
            asset_root=asset_root,
        )


def test_manifest_decodes_mp4_and_publishes_markdown_gif(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("av")
    image_module = pytest.importorskip("PIL.Image")
    qa_root = tmp_path / "qa_runs"
    asset_root = tmp_path / "roboverse_data"
    manifest = _phase1_manifest(qa_root, asset_root)
    video_path = manifest.parent / "mujoco" / "overview.mp4"
    writer = RGBVideoWriter(video_path, width=32, height=24, fps=6)
    try:
        for value in (10, 130, 240):
            writer.append(np.full((24, 32, 3), value, dtype=np.uint8))
    finally:
        writer.close()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["runs"][0]["previews"][0]["path"] = "mujoco/overview.mp4"
    manifest.write_text(json.dumps(data), encoding="utf-8")

    _, destination = publish_manifest(
        manifest,
        qa_root=qa_root,
        asset_root=asset_root,
        output_root=tmp_path / "gallery",
    )

    gif_path = destination / "mujoco" / "mujoco_overview.gif"
    assert gif_path.is_file()
    with image_module.open(gif_path) as gif:
        assert gif.n_frames >= 2
    assert (destination / "mujoco" / "mujoco_overview.mp4").is_file()
