"""Release-gate tests for clean, evidence-backed ICL demo publication."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from tools.icl import publish_benchmark as publisher
from tools.icl.benchmark_queue import atomic_write_result, complete, connect_queue, enqueue, job_id_for, lease


def _spec(root: Path, *, backend: str, phase: str, dependencies: list[str] | None = None) -> dict:
    return {
        "source": "synthicl",
        "upstream_sha": "a" * 40,
        "task_id": "pick_and_place_gn",
        "variant": "default",
        "scene_id": "tabletop/scene_0",
        "embodiment": "franka",
        "backend": backend,
        "seed": 17,
        "roboverse_sha": "b" * 40,
        "phase": phase,
        "owner": "Release-QA",
        "dependencies": dependencies or [],
        "requirements": {"cpu_cores": 1, "ram_gb": 1},
        "artifact_dir": str(root / "runs" / backend / phase),
    }


def _result(spec: dict, status: str) -> dict:
    result = {
        **{field: spec[field] for field in publisher.KEY_FIELDS},
        "job_id": job_id_for(spec),
        "status": status,
        "exit_code": 0,
        "command": ["python", "-m", "pytest", "-q"],
        "artifact_dir": spec["artifact_dir"],
        "asset_hashes": {"scene": "c" * 64},
    }
    if spec["phase"] == "semantics":
        result["validation"] = {
            "kind": "task_success",
            "passed": True,
            "success": True,
            "criterion": "target object rests on destination surface",
            "observed": {"object_on_target": True},
        }
    elif spec["phase"] == "demo":
        result.update({
            "source_license": "Apache-2.0",
            "source_url": "https://github.com/sNiper-Qian/SynthICL",
            "asset_records": {
                "scene": {
                    "sha256": "c" * 64,
                    "license": "CC-BY-4.0",
                    "source_url": "https://huggingface.co/datasets/example/source-scene",
                }
            },
            "success_evidence": {
                "kind": "task_success",
                "passed": True,
                "success": True,
                "criterion": "target object rests on destination surface",
                "observed": {"object_on_target": True},
            },
            "artifacts": {"video": "demo.mp4", "thumbnail": "thumbnail.png", "trajectory": "trajectory.json"},
        })
        output = Path(spec["artifact_dir"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "demo.mp4").write_bytes(b"mp4 fixture")
        (output / "thumbnail.png").write_bytes(b"png fixture")
        (output / "trajectory.json").write_text(
            json.dumps({"actions": [[0.0], [1.0]], "trace": [{"state": [0.0]}, {"state": [1.0]}]}),
            encoding="utf-8",
        )
        result["artifact_sha256"] = {
            role: hashlib.sha256((output / filename).read_bytes()).hexdigest()
            for role, filename in {
                "video": "demo.mp4",
                "thumbnail": "thumbnail.png",
                "trajectory": "trajectory.json",
            }.items()
        }
    elif spec["phase"] == "parity":
        result["parity_evidence"] = {
            "passed": True,
            "compared_backends": ["mujoco", "genesis"],
            "metrics": {"final_object_position_m": {"value": 0.002, "tolerance": 0.01}},
        }
    return result


def _enqueue_complete(connection: sqlite3.Connection, spec: dict, status: str, extra_result: dict | None = None) -> str:
    job_id, _ = enqueue(connection, spec)
    claim = lease(connection, worker=f"test-{spec['phase']}", source=spec["source"], phase=spec["phase"])
    assert claim is not None and claim["job_id"] == job_id
    result_path = Path(spec["artifact_dir"]) / "result.json"
    Path(spec["artifact_dir"]).mkdir(parents=True, exist_ok=True)
    result = _result(spec, status)
    result.update(extra_result or {})
    atomic_write_result(result_path, result)
    assert complete(connection, job_id=job_id, token=claim["lease_token"], result_path=result_path) == status
    return job_id


def _complete_pair(connection: sqlite3.Connection, root: Path) -> None:
    specs = {}
    for backend in ("mujoco", "genesis"):
        smoke = _spec(root, backend=backend, phase="smoke")
        smoke_id, _ = enqueue(connection, smoke)
        semantics = _spec(root, backend=backend, phase="semantics", dependencies=[smoke_id])
        semantics_id, _ = enqueue(connection, semantics)
        demo = _spec(root, backend=backend, phase="demo", dependencies=[semantics_id])
        demo_id, _ = enqueue(connection, demo)
        specs[backend] = {"smoke": smoke, "semantics": semantics, "demo": demo, "demo_id": demo_id}
    for backend in ("mujoco", "genesis"):
        item = specs[backend]
        for phase in ("smoke", "semantics", "demo"):
            status = publisher._GATE_STATUS[phase]
            _enqueue_complete(connection, item[phase], status)
    for backend in ("mujoco", "genesis"):
        item = specs[backend]
        refs = [specs["mujoco"]["demo_id"], specs["genesis"]["demo_id"]]
        parity = _spec(root, backend=backend, phase="parity", dependencies=refs)
        compared_results = {}
        compared_trajectories = {}
        for job_id in refs:
            row = connection.execute("SELECT result_sha256, result_json FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            recorded = json.loads(row["result_json"])
            compared_results[job_id] = row["result_sha256"]
            compared_trajectories[job_id] = recorded["artifact_sha256"]["trajectory"]
        _enqueue_complete(
            connection,
            parity,
            "parity_passed",
            {
                "parity_evidence": {
                    "passed": True,
                    "compared_backends": ["mujoco", "genesis"],
                    "metrics": {"final_object_position_m": {"value": 0.002, "tolerance": 0.01}},
                    "compared_demo_result_sha256": compared_results,
                    "compared_trajectory_sha256": compared_trajectories,
                }
            },
        )


def test_release_gate_requires_smoke_semantics_demo_and_cross_backend_parity(tmp_path, monkeypatch):
    connection = connect_queue(tmp_path / "queue.sqlite")
    try:
        _complete_pair(connection, tmp_path)
        monkeypatch.setattr(
            publisher,
            "_inspect_video",
            lambda path: {"width": 256, "height": 256, "decoded_frames": 2, "duration_seconds": 1 / 30},
        )
        monkeypatch.setattr(publisher, "_inspect_thumbnail", lambda path: {"width": 64, "height": 64})

        audit = publisher.inspect_candidates(connection)

        assert audit["demo_candidate_count"] == 2
        assert len(audit["eligible"]) == 2
        assert audit["rejected"] == []
        output = tmp_path / "ICL_Benchmark"
        old = output / "unverified.txt"
        output.mkdir()
        old.write_text("preserve previous generation", encoding="utf-8")
        result = publisher.publish(audit["eligible"], output, queue_summary=audit["queue_summary"])
        assert result["published_count"] == 2
        assert Path(result["backup"]).is_dir()
        assert (Path(result["backup"]) / "unverified.txt").is_file()
        index = json.loads((output / "index.json").read_text(encoding="utf-8"))
        assert index["published_count"] == 2
        for item in index["items"]:
            location = output / item["path"]
            assert (location / "demo.mp4").is_file()
            assert (location / "thumbnail.png").is_file()
            assert (location / "trajectory.json").is_file()
            verification = json.loads((location / "verification.json").read_text(encoding="utf-8"))
            assert verification["status"] == "published"
            assert set(verification["phase_results"]) == {"smoke", "semantics", "demo", "parity"}
    finally:
        connection.close()


def test_demo_without_all_gates_is_not_eligible_and_output_is_untouched(tmp_path):
    connection = connect_queue(tmp_path / "queue.sqlite")
    try:
        demo = _spec(tmp_path, backend="mujoco", phase="demo")
        _enqueue_complete(connection, demo, "demo_verified")
        audit = publisher.inspect_candidates(connection)
        assert audit["eligible"] == []
        assert audit["rejected"][0]["reason"] == "required smoke gate is missing or not passed"
        output = tmp_path / "ICL_Benchmark"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("unchanged", encoding="utf-8")
        with pytest.raises(publisher.PublishError, match="no demo passed"):
            publisher.publish(audit["eligible"], output)
        assert marker.read_text(encoding="utf-8") == "unchanged"
    finally:
        connection.close()


def test_artifact_mutated_after_demo_verification_is_rejected(tmp_path, monkeypatch):
    connection = connect_queue(tmp_path / "queue.sqlite")
    try:
        _complete_pair(connection, tmp_path)
        (tmp_path / "runs" / "mujoco" / "demo" / "demo.mp4").write_bytes(b"replaced after verification")
        monkeypatch.setattr(
            publisher,
            "_inspect_video",
            lambda path: {"width": 256, "height": 256, "decoded_frames": 2, "duration_seconds": 1 / 30},
        )
        monkeypatch.setattr(publisher, "_inspect_thumbnail", lambda path: {"width": 64, "height": 64})

        audit = publisher.inspect_candidates(connection)

        assert audit["eligible"] == []
        assert any("changed after its verification result" in item["reason"] for item in audit["rejected"])
    finally:
        connection.close()


def test_artifact_path_traversal_is_rejected(tmp_path):
    job = {"artifact_dir": str(tmp_path / "job")}
    Path(job["artifact_dir"]).mkdir()
    with pytest.raises(publisher.PublishError, match="unsafe video artifact path"):
        publisher._artifact_path(job, "../outside.mp4", role="video")


def test_video_probe_decodes_a_real_mp4(tmp_path):
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    path = tmp_path / "sample.mp4"
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=5)
    stream.width, stream.height = 64, 48
    stream.pix_fmt = "yuv420p"
    for index in range(3):
        pixels = np.full((48, 64, 3), index * 40, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
        frame.pts = index
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    probe = publisher._inspect_video(path)

    assert probe["width"] == 64 and probe["height"] == 48
    assert probe["decoded_frames"] == 3
    assert probe["duration_seconds"] > 0


def test_thumbnail_probe_decodes_png(tmp_path):
    image = pytest.importorskip("PIL.Image")
    path = tmp_path / "thumbnail.png"
    image.new("RGB", (96, 64), (12, 34, 56)).save(path)

    assert publisher._inspect_thumbnail(path) == {"width": 96, "height": 64}


def test_json_trajectory_requires_aligned_finite_state_records(tmp_path):
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps({"actions": [[0.0], [float("nan")]], "trace": [{"q": [0.0]}, {"q": [1.0]}]}))

    with pytest.raises(publisher.PublishError, match="finite numeric values"):
        publisher._inspect_trajectory(path)

    path.write_text(json.dumps({"actions": [[0.0], [1.0]], "trace": [{"q": [0.0]}]}))
    with pytest.raises(publisher.PublishError, match="aligned state/observation"):
        publisher._inspect_trajectory(path)


def test_random_exploration_is_validated_without_being_called_task_success():
    validation = {
        "kind": "trajectory_validity",
        "passed": True,
        "valid": True,
        "success": False,
        "checks": [
            {"name": "finite_actions", "passed": True},
            {"name": "within_action_bounds", "passed": True},
            {"name": "timebase_aligned", "passed": True},
        ],
    }
    assert publisher._validate_semantics({"validation": validation}) is validation
    with pytest.raises(publisher.PublishError, match="cannot be labeled as task success"):
        publisher._validate_semantics({"validation": {**validation, "success": True}})
