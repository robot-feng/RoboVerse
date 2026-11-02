"""Contracts for the local, resumable ICL verification queue."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tools.icl.benchmark_queue import (
    QueueError,
    atomic_write_result,
    complete,
    connect_queue,
    enqueue,
    heartbeat,
    job_id_for,
    lease,
    list_attempts,
    report,
    retry,
)


def _spec(tmp_path: Path, *, phase: str = "smoke", **updates):
    spec = {
        "source": "geniesim",
        "upstream_sha": "a" * 40,
        "task_id": "g1op_s2r_pack_in_supermarket",
        "variant": "problem0",
        "scene_id": "table_task_g1_op/0",
        "embodiment": "g1_op",
        "backend": "mujoco",
        "seed": 0,
        "roboverse_sha": "b" * 40,
        "phase": phase,
        "owner": "Smoke-QA",
        "dependencies": [],
        "requirements": {"cpu_cores": 1, "ram_gb": 2},
        "artifact_dir": str(tmp_path / f"artifact-{phase}"),
    }
    spec.update(updates)
    return spec


def _successful_result(spec, status: str):
    return {
        **{
            key: spec[key]
            for key in (
                "source",
                "upstream_sha",
                "task_id",
                "variant",
                "scene_id",
                "embodiment",
                "backend",
                "seed",
                "roboverse_sha",
                "phase",
            )
        },
        "job_id": job_id_for(spec),
        "status": status,
        "exit_code": 0,
        "command": ["python", "-m", "pytest", "-q"],
        "artifact_dir": spec["artifact_dir"],
        "asset_hashes": {},
    }


def test_enqueue_is_idempotent_and_seed_is_part_of_stable_job_id(tmp_path):
    connection = connect_queue(tmp_path / "queue.sqlite")
    try:
        spec = _spec(tmp_path)
        first_id, inserted = enqueue(connection, spec)
        assert inserted
        assert enqueue(connection, spec) == (first_id, False)
        assert job_id_for({**spec, "seed": 1}) != first_id
        with pytest.raises(QueueError, match="different immutable metadata"):
            enqueue(connection, {**spec, "owner": "Physics-QA"})
        with pytest.raises(QueueError, match="already assigned"):
            enqueue(connection, {**spec, "seed": 1})
    finally:
        connection.close()


def test_dependency_gate_atomic_result_registration_and_report(tmp_path):
    connection = connect_queue(tmp_path / "queue.sqlite")
    try:
        migration = _spec(tmp_path, phase="migrate", task_id="sample_task")
        migration_id, _ = enqueue(connection, migration)
        smoke = _spec(tmp_path, phase="smoke", task_id="sample_task", dependencies=[migration_id])
        smoke_id, _ = enqueue(connection, smoke)

        assert lease(connection, worker="smoke-1", phase="smoke") is None
        claim = lease(connection, worker="migration-1", phase="migrate")
        assert claim["job_id"] == migration_id
        artifact = Path(migration["artifact_dir"])
        artifact.mkdir(parents=True)
        result_path = artifact / "result.json"
        atomic_write_result(result_path, _successful_result(migration, "imported"))
        assert (
            complete(connection, job_id=migration_id, token=claim["lease_token"], result_path=result_path) == "imported"
        )

        claim = lease(connection, worker="smoke-1", phase="smoke")
        assert claim["job_id"] == smoke_id
        artifact = Path(smoke["artifact_dir"])
        artifact.mkdir(parents=True)
        result_path = artifact / "result.json"
        atomic_write_result(result_path, _successful_result(smoke, "smoke_passed"))
        assert (
            complete(connection, job_id=smoke_id, token=claim["lease_token"], result_path=result_path) == "smoke_passed"
        )
        attempts = list_attempts(connection, job_id=smoke_id)
        assert len(attempts) == 1 and attempts[0]["status"] == "smoke_passed"
        assert attempts[0]["ended_at"] >= attempts[0]["started_at"]
        assert report(connection) == {
            "total_jobs": 2,
            "counts": [
                {"source": "geniesim", "phase": "migrate", "status": "imported", "count": 1},
                {"source": "geniesim", "phase": "smoke", "status": "smoke_passed", "count": 1},
            ],
        }
    finally:
        connection.close()


def test_expired_lease_is_reclaimed_and_old_token_cannot_heartbeat(tmp_path):
    connection = connect_queue(tmp_path / "queue.sqlite")
    try:
        spec = _spec(tmp_path)
        job_id, _ = enqueue(connection, spec)
        first = lease(connection, worker="worker-a", lease_seconds=10, now=100)
        assert first["job_id"] == job_id
        assert lease(connection, worker="worker-b", now=105) is None
        second = lease(connection, worker="worker-b", lease_seconds=10, now=111)
        assert second["job_id"] == job_id
        assert second["lease_token"] != first["lease_token"]
        assert second["attempts"] == 2
        history = list_attempts(connection, job_id=job_id)
        assert [(row["worker_pid"], row["started_at"], row["ended_at"], row["status"]) for row in history] == [
            (os.getpid(), 100.0, 111.0, "lease_expired"),
            (os.getpid(), 111.0, None, None),
        ]
        with pytest.raises(QueueError, match="No active lease"):
            heartbeat(connection, job_id=job_id, token=first["lease_token"])
    finally:
        connection.close()


def test_two_workers_cannot_lease_the_same_job(tmp_path):
    db_path = tmp_path / "queue.sqlite"
    connection = connect_queue(db_path)
    enqueue(connection, _spec(tmp_path))
    connection.close()

    def claim(worker):
        worker_connection = connect_queue(db_path)
        try:
            result = lease(worker_connection, worker=worker)
            return None if result is None else result["job_id"]
        finally:
            worker_connection.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(claim, ("worker-a", "worker-b")))
    assert sum(item is not None for item in claims) == 1


def test_gpu_resource_lease_limits_same_device_to_one_active_job(tmp_path):
    connection = connect_queue(tmp_path / "queue.sqlite")
    try:
        first = _spec(tmp_path, requirements={"gpu_id": 0, "vram_gb": 2})
        first_id, _ = enqueue(connection, first)
        second = _spec(
            tmp_path,
            task_id="another_task",
            requirements={"gpu_id": 0, "vram_gb": 2},
            artifact_dir=str(tmp_path / "artifact-other"),
        )
        enqueue(connection, second)
        claim = lease(connection, worker="gpu-worker-0")
        assert claim["job_id"] == first_id
        assert lease(connection, worker="gpu-worker-1") is None

        Path(first["artifact_dir"]).mkdir(parents=True)
        result_path = Path(first["artifact_dir"]) / "result.json"
        atomic_write_result(result_path, _successful_result(first, "smoke_passed"))
        complete(connection, job_id=first_id, token=claim["lease_token"], result_path=result_path)
        next_claim = lease(connection, worker="gpu-worker-1")
        assert next_claim["job_id"] == job_id_for(second)
    finally:
        connection.close()


def test_complete_rejects_mismatched_provenance_and_failed_exit_code(tmp_path):
    connection = connect_queue(tmp_path / "queue.sqlite")
    try:
        spec = _spec(tmp_path)
        job_id, _ = enqueue(connection, spec)
        claim = lease(connection, worker="smoke-1")
        Path(spec["artifact_dir"]).mkdir(parents=True)
        path = Path(spec["artifact_dir"]) / "result.json"
        bad = _successful_result(spec, "smoke_passed")
        bad["scene_id"] = "other-scene"
        atomic_write_result(path, bad)
        with pytest.raises(QueueError, match="provenance key"):
            complete(connection, job_id=job_id, token=claim["lease_token"], result_path=path)

        bad = _successful_result(spec, "failed")
        atomic_write_result(path, bad)
        with pytest.raises(QueueError, match="nonzero exit_code"):
            complete(connection, job_id=job_id, token=claim["lease_token"], result_path=path)

        bad = _successful_result(spec, "demo_verified")
        atomic_write_result(path, bad)
        with pytest.raises(QueueError, match="can only complete with smoke_passed"):
            complete(connection, job_id=job_id, token=claim["lease_token"], result_path=path)
    finally:
        connection.close()


def test_retry_preserves_attempt_counter(tmp_path):
    connection = connect_queue(tmp_path / "queue.sqlite")
    try:
        spec = _spec(tmp_path)
        job_id, _ = enqueue(connection, spec)
        claim = lease(connection, worker="worker")
        Path(spec["artifact_dir"]).mkdir(parents=True)
        result = _successful_result(spec, "failed")
        result["exit_code"] = 1
        result["error"] = "test failure"
        path = Path(spec["artifact_dir"]) / "failed.json"
        atomic_write_result(path, result)
        assert complete(connection, job_id=job_id, token=claim["lease_token"], result_path=path) == "failed"
        retry(connection, job_id=job_id)
        claim = lease(connection, worker="worker-2")
        assert claim["attempts"] == 2
        assert claim["status"] == "leased"
    finally:
        connection.close()
