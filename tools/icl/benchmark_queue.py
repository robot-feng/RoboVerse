"""Small SQLite lease queue for reproducible ICL benchmark verification jobs.

The queue tracks test work, not task definitions. Existing RoboVerse task and source catalogs
remain the source of truth. Result files must be atomically written by the worker before a job
can leave the leased state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

KEY_FIELDS = (
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
PHASES = {"migrate", "smoke", "semantics", "demo", "parity", "release"}
SUCCESS_STATUS_BY_PHASE = {
    "migrate": "imported",
    "smoke": "smoke_passed",
    "semantics": "semantics_passed",
    "demo": "demo_verified",
    "parity": "parity_passed",
    "release": "published",
}
TERMINAL_STATUSES = frozenset({
    "imported",
    "smoke_passed",
    "semantics_passed",
    "demo_verified",
    "parity_passed",
    "published",
    "failed",
    "blocked",
    "unsupported",
})
DEPENDENCY_STATUSES = frozenset({
    "imported",
    "smoke_passed",
    "semantics_passed",
    "demo_verified",
    "parity_passed",
    "published",
})
SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    upstream_sha TEXT NOT NULL,
    task_id TEXT NOT NULL,
    variant TEXT NOT NULL,
    scene_id TEXT NOT NULL,
    embodiment TEXT NOT NULL,
    backend TEXT NOT NULL,
    seed INTEGER NOT NULL,
    roboverse_sha TEXT NOT NULL,
    phase TEXT NOT NULL,
    owner TEXT NOT NULL,
    dependencies_json TEXT NOT NULL,
    requirements_json TEXT NOT NULL,
    artifact_dir TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'leased', 'imported', 'smoke_passed',
        'semantics_passed', 'demo_verified', 'parity_passed', 'published', 'failed', 'blocked', 'unsupported')),
    attempts INTEGER NOT NULL DEFAULT 0,
    lease_owner TEXT,
    lease_token TEXT,
    lease_until REAL,
    result_path TEXT,
    result_sha256 TEXT,
    result_json TEXT,
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE (source, upstream_sha, task_id, variant, scene_id, embodiment, backend, seed, roboverse_sha, phase)
);
CREATE INDEX IF NOT EXISTS jobs_claim_idx ON jobs(status, phase, source, created_at);
CREATE TABLE IF NOT EXISTS resource_leases (
    resource_key TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(job_id),
    lease_token TEXT NOT NULL,
    lease_until REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS job_attempts (
    job_id TEXT NOT NULL REFERENCES jobs(job_id),
    attempt INTEGER NOT NULL,
    worker TEXT NOT NULL,
    worker_pid INTEGER NOT NULL,
    lease_token TEXT NOT NULL UNIQUE,
    started_at REAL NOT NULL,
    lease_until REAL NOT NULL,
    heartbeat_at REAL,
    ended_at REAL,
    status TEXT,
    PRIMARY KEY (job_id, attempt)
);
"""


class QueueError(ValueError):
    """Raised when a queue operation violates the job or result contract."""


def job_id_for(spec: dict[str, Any]) -> str:
    """Return a stable digest for one phase of a benchmark test unit."""
    key = {field: spec[field] for field in KEY_FIELDS}
    canonical = json.dumps(key, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalize_spec(raw: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in KEY_FIELDS if field not in raw]
    if missing:
        raise QueueError(f"Job is missing required key fields: {', '.join(missing)}")
    spec = {field: raw[field] for field in KEY_FIELDS}
    for field, value in spec.items():
        if field == "seed":
            if type(value) is not int:
                raise QueueError("seed must be an integer")
        elif not isinstance(value, str) or not value.strip():
            raise QueueError(f"{field} must be a non-empty string; use an explicit value such as 'none'")
    if spec["phase"] not in PHASES:
        raise QueueError(f"Unsupported phase: {spec['phase']}")
    owner = raw.get("owner", "unassigned")
    artifact_dir = raw.get("artifact_dir")
    dependencies = raw.get("dependencies", [])
    requirements = raw.get("requirements", {})
    if not isinstance(owner, str) or not owner.strip():
        raise QueueError("owner must be a non-empty role or worker identifier")
    if not isinstance(artifact_dir, str) or not artifact_dir.strip():
        raise QueueError("artifact_dir is required and must be isolated for this job")
    if not isinstance(dependencies, list) or any(not isinstance(item, str) or not item for item in dependencies):
        raise QueueError("dependencies must be a list of job IDs")
    if not isinstance(requirements, dict):
        raise QueueError("requirements must be a JSON object")
    return {
        **spec,
        "owner": owner.strip(),
        "artifact_dir": artifact_dir,
        "dependencies": dependencies,
        "requirements": requirements,
        "job_id": job_id_for(spec),
    }


def connect_queue(path: Path) -> sqlite3.Connection:
    """Open a local SQLite queue and create its schema in WAL mode."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30.0, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(SCHEMA)
    return connection


def enqueue(connection: sqlite3.Connection, raw: dict[str, Any]) -> tuple[str, bool]:
    """Insert a job idempotently; return ``(job_id, inserted)``."""
    spec = _normalize_spec(raw)
    now = time.time()
    with connection:
        existing = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (spec["job_id"],)).fetchone()
        if existing is not None:
            fixed = {
                "owner": spec["owner"],
                "dependencies_json": json.dumps(spec["dependencies"], sort_keys=True),
                "requirements_json": json.dumps(spec["requirements"], sort_keys=True),
                "artifact_dir": spec["artifact_dir"],
            }
            if any(existing[key] != value for key, value in fixed.items()):
                raise QueueError(f"Job {spec['job_id']} already exists with different immutable metadata")
            return spec["job_id"], False
        artifact_owner = connection.execute(
            "SELECT job_id FROM jobs WHERE artifact_dir = ?", (spec["artifact_dir"],)
        ).fetchone()
        if artifact_owner is not None:
            raise QueueError(
                f"Artifact directory is already assigned to job {artifact_owner['job_id']}; use one directory per job"
            )
        connection.execute(
            """INSERT INTO jobs (
                job_id, source, upstream_sha, task_id, variant, scene_id, embodiment, backend, seed,
                roboverse_sha, phase, owner, dependencies_json, requirements_json, artifact_dir,
                status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)""",
            (
                spec["job_id"],
                *(spec[field] for field in KEY_FIELDS),
                spec["owner"],
                json.dumps(spec["dependencies"], sort_keys=True),
                json.dumps(spec["requirements"], sort_keys=True),
                spec["artifact_dir"],
                now,
                now,
            ),
        )
    return spec["job_id"], True


def enqueue_manifest(
    connection: sqlite3.Connection, manifest: dict[str, Any] | list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Idempotently add jobs from a JSON object with ``jobs`` or a raw job list."""
    jobs = manifest if isinstance(manifest, list) else manifest.get("jobs") if isinstance(manifest, dict) else None
    if not isinstance(jobs, list):
        raise QueueError("Manifest must be a job list or an object containing a 'jobs' list")
    results = []
    for raw in jobs:
        job_id, inserted = enqueue(connection, raw)
        results.append({"job_id": job_id, "inserted": inserted})
    return results


def _dependencies_satisfied(connection: sqlite3.Connection, row: sqlite3.Row) -> bool:
    dependencies = json.loads(row["dependencies_json"])
    for dependency in dependencies:
        status = connection.execute("SELECT status FROM jobs WHERE job_id = ?", (dependency,)).fetchone()
        if status is None or status["status"] not in DEPENDENCY_STATUSES:
            return False
    return True


def lease(
    connection: sqlite3.Connection,
    *,
    worker: str,
    lease_seconds: float = 600.0,
    source: str | None = None,
    phase: str | None = None,
    worker_pid: int | None = None,
    now: float | None = None,
) -> dict[str, Any] | None:
    """Atomically claim the oldest eligible queued or expired job."""
    if not worker.strip() or lease_seconds <= 0:
        raise QueueError("worker must be non-empty and lease_seconds must be positive")
    worker_pid = os.getpid() if worker_pid is None else worker_pid
    if type(worker_pid) is not int or worker_pid <= 0:
        raise QueueError("worker_pid must be a positive process ID")
    now = time.time() if now is None else now
    connection.execute("BEGIN IMMEDIATE")
    try:
        clauses = ["(status = 'queued' OR (status = 'leased' AND lease_until <= ?))"]
        params: list[Any] = [now]
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        if phase is not None:
            clauses.append("phase = ?")
            params.append(phase)
        candidates = connection.execute(
            f"SELECT * FROM jobs WHERE {' AND '.join(clauses)} ORDER BY created_at, job_id", params
        ).fetchall()
        row = None
        for candidate in candidates:
            if not _dependencies_satisfied(connection, candidate):
                continue
            gpu_id = json.loads(candidate["requirements_json"]).get("gpu_id")
            if gpu_id is not None:
                resource = connection.execute(
                    "SELECT job_id, lease_until FROM resource_leases WHERE resource_key=?", (f"gpu:{gpu_id}",)
                ).fetchone()
                if resource is not None and resource["lease_until"] > now and resource["job_id"] != candidate["job_id"]:
                    continue
            row = candidate
            break
        if row is None:
            connection.rollback()
            return None
        if row["status"] == "leased":
            connection.execute(
                "UPDATE job_attempts SET ended_at=?, status='lease_expired' WHERE job_id=? AND lease_token=?",
                (now, row["job_id"], row["lease_token"]),
            )
        token = uuid.uuid4().hex
        deadline = now + lease_seconds
        attempt_number = int(row["attempts"]) + 1
        connection.execute(
            """UPDATE jobs SET status='leased', lease_owner=?, lease_token=?, lease_until=?,
               attempts=attempts+1, updated_at=? WHERE job_id=?""",
            (worker, token, deadline, now, row["job_id"]),
        )
        connection.execute(
            """INSERT INTO job_attempts (
                job_id, attempt, worker, worker_pid, lease_token, started_at, lease_until
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (row["job_id"], attempt_number, worker, worker_pid, token, now, deadline),
        )
        gpu_id = json.loads(row["requirements_json"]).get("gpu_id")
        if gpu_id is not None:
            connection.execute(
                """INSERT INTO resource_leases (resource_key, job_id, lease_token, lease_until)
                   VALUES (?, ?, ?, ?) ON CONFLICT(resource_key) DO UPDATE SET
                   job_id=excluded.job_id, lease_token=excluded.lease_token, lease_until=excluded.lease_until""",
                (f"gpu:{gpu_id}", row["job_id"], token, deadline),
            )
        leased = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (row["job_id"],)).fetchone()
        connection.commit()
        result = dict(leased)
        result["dependencies"] = json.loads(result.pop("dependencies_json"))
        result["requirements"] = json.loads(result.pop("requirements_json"))
        result["lease_token"] = token
        return result
    except Exception:
        connection.rollback()
        raise


def heartbeat(connection: sqlite3.Connection, *, job_id: str, token: str, lease_seconds: float = 600.0) -> float:
    """Extend a lease only when the caller still owns its current token."""
    if lease_seconds <= 0:
        raise QueueError("lease_seconds must be positive")
    now = time.time()
    deadline = now + lease_seconds
    with connection:
        cursor = connection.execute(
            "UPDATE jobs SET lease_until=?, updated_at=? WHERE job_id=? AND status='leased' AND lease_token=?",
            (deadline, now, job_id, token),
        )
        if cursor.rowcount != 1:
            raise QueueError(f"No active lease for job {job_id} and supplied token")
        connection.execute(
            "UPDATE resource_leases SET lease_until=? WHERE job_id=? AND lease_token=?",
            (deadline, job_id, token),
        )
        connection.execute(
            "UPDATE job_attempts SET lease_until=?, heartbeat_at=? WHERE job_id=? AND lease_token=?",
            (deadline, now, job_id, token),
        )
    return deadline


def atomic_write_result(path: Path, result: dict[str, Any]) -> None:
    """Write a JSON result through a same-directory temporary file and atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def complete(connection: sqlite3.Connection, *, job_id: str, token: str, result_path: Path) -> str:
    """Register a previously written result, validating provenance and outcome."""
    result_path = Path(result_path)
    if not result_path.is_file() or result_path.stat().st_size == 0:
        raise QueueError(f"Result file is missing or empty: {result_path}")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QueueError(f"Could not read result JSON: {result_path}") from exc
    if not isinstance(result, dict):
        raise QueueError("Result must be a JSON object")
    row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    if row is None:
        raise QueueError(f"Unknown job: {job_id}")
    if row["status"] != "leased" or row["lease_token"] != token:
        raise QueueError(f"Job {job_id} is not held by the supplied lease token")
    if not Path(row["artifact_dir"]).is_dir():
        raise QueueError(f"Job artifact directory does not exist: {row['artifact_dir']}")
    if result.get("job_id") != job_id:
        raise QueueError("Result job_id does not match the leased job")
    if any(result.get(field) != row[field] for field in KEY_FIELDS):
        raise QueueError("Result provenance key does not match the leased job")
    if result.get("artifact_dir") != row["artifact_dir"]:
        raise QueueError("Result artifact_dir does not match the leased job")
    if not isinstance(result.get("command"), (str, list)) or not result["command"]:
        raise QueueError("Result must record the executed command")
    if not isinstance(result.get("asset_hashes"), dict):
        raise QueueError("Result must record asset hashes, or an empty object when no assets apply")
    exit_code = result.get("exit_code")
    if type(exit_code) is not int:
        raise QueueError("Result exit_code must be an integer")
    status = result.get("status")
    if status not in TERMINAL_STATUSES:
        raise QueueError(f"Result has an unsupported terminal status: {status!r}")
    if status in SUCCESS_STATUS_BY_PHASE.values() and status != SUCCESS_STATUS_BY_PHASE[row["phase"]]:
        raise QueueError(f"Phase {row['phase']} can only complete with {SUCCESS_STATUS_BY_PHASE[row['phase']]}")
    if status == SUCCESS_STATUS_BY_PHASE[row["phase"]] and exit_code != 0:
        raise QueueError("A passing result must have exit_code 0")
    if status == "failed" and exit_code == 0:
        raise QueueError("A failed result must have a nonzero exit_code")
    if status in {"blocked", "unsupported"} and not str(result.get("reason", "")).strip():
        raise QueueError(f"A {status} result must include a concrete reason")
    result_sha = hashlib.sha256(result_path.read_bytes()).hexdigest()
    with connection:
        cursor = connection.execute(
            """UPDATE jobs SET status=?, result_path=?, result_sha256=?, result_json=?, error=?,
               lease_owner=NULL, lease_token=NULL, lease_until=NULL, updated_at=?
               WHERE job_id=? AND status='leased' AND lease_token=?""",
            (
                status,
                str(result_path.resolve()),
                result_sha,
                json.dumps(result, ensure_ascii=False, sort_keys=True),
                result.get("reason") or result.get("error"),
                time.time(),
                job_id,
                token,
            ),
        )
        if cursor.rowcount != 1:
            raise QueueError("Lease was reassigned before result registration")
        connection.execute("DELETE FROM resource_leases WHERE job_id=? AND lease_token=?", (job_id, token))
        connection.execute(
            "UPDATE job_attempts SET ended_at=?, status=? WHERE job_id=? AND lease_token=?",
            (time.time(), status, job_id, token),
        )
    return status


def retry(connection: sqlite3.Connection, *, job_id: str) -> None:
    """Requeue a failed, blocked, or unsupported job without erasing its attempt history."""
    with connection:
        cursor = connection.execute(
            """UPDATE jobs SET status='queued', result_path=NULL, result_sha256=NULL, result_json=NULL,
               error=NULL, lease_owner=NULL, lease_token=NULL, lease_until=NULL, updated_at=?
               WHERE job_id=? AND status IN ('failed', 'blocked', 'unsupported')""",
            (time.time(), job_id),
        )
        if cursor.rowcount != 1:
            raise QueueError(f"Job {job_id} is not in a retryable terminal state")
        connection.execute("DELETE FROM resource_leases WHERE job_id=?", (job_id,))


def list_jobs(connection: sqlite3.Connection, *, status: str | None = None) -> list[dict[str, Any]]:
    """List jobs without exposing secret lease tokens."""
    if status is None:
        rows = connection.execute("SELECT * FROM jobs ORDER BY created_at, job_id").fetchall()
    else:
        rows = connection.execute("SELECT * FROM jobs WHERE status=? ORDER BY created_at, job_id", (status,)).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item.pop("lease_token", None)
        item["dependencies"] = json.loads(item.pop("dependencies_json"))
        item["requirements"] = json.loads(item.pop("requirements_json"))
        result.append(item)
    return result


def list_attempts(connection: sqlite3.Connection, *, job_id: str) -> list[dict[str, Any]]:
    """Return process IDs and execution intervals for every lease attempt."""
    rows = connection.execute(
        "SELECT job_id, attempt, worker, worker_pid, started_at, lease_until, heartbeat_at, ended_at, status "
        "FROM job_attempts WHERE job_id=? ORDER BY attempt",
        (job_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def report(connection: sqlite3.Connection) -> dict[str, Any]:
    """Aggregate durable queue state by source, phase, and status."""
    rows = connection.execute(
        "SELECT source, phase, status, COUNT(*) AS count FROM jobs GROUP BY source, phase, status "
        "ORDER BY source, phase, status"
    ).fetchall()
    counts = [dict(row) for row in rows]
    return {"total_jobs": sum(row["count"] for row in counts), "counts": counts}


def _read_manifest(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, (dict, list)):
        raise QueueError("Manifest root must be an object or array")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("examples/output/qa_runs/icl_benchmark_queue.sqlite"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="create the local WAL database")
    add_manifest = commands.add_parser("enqueue", help="idempotently enqueue a JSON job manifest")
    add_manifest.add_argument("manifest", type=Path)
    claim = commands.add_parser("lease", help="atomically lease one eligible job")
    claim.add_argument("--worker", required=True)
    claim.add_argument("--worker-pid", type=int, default=os.getpid())
    claim.add_argument("--source")
    claim.add_argument("--phase", choices=sorted(PHASES))
    claim.add_argument("--lease-seconds", type=float, default=600.0)
    beat = commands.add_parser("heartbeat", help="extend a current lease")
    beat.add_argument("job_id")
    beat.add_argument("token")
    beat.add_argument("--lease-seconds", type=float, default=600.0)
    finish = commands.add_parser("complete", help="register an atomically written result JSON")
    finish.add_argument("job_id")
    finish.add_argument("token")
    finish.add_argument("result", type=Path)
    requeue = commands.add_parser("retry", help="requeue a failed or blocked job")
    requeue.add_argument("job_id")
    show = commands.add_parser("list", help="list queue jobs")
    show.add_argument("--status")
    commands.add_parser("report", help="summarize queue state")
    attempts = commands.add_parser("attempts", help="show worker PIDs and intervals for one job")
    attempts.add_argument("job_id")
    args = parser.parse_args()
    connection = connect_queue(args.db)
    try:
        if args.command == "init":
            result: Any = {"database": str(args.db.resolve()), "journal_mode": "WAL"}
        elif args.command == "enqueue":
            result = enqueue_manifest(connection, _read_manifest(args.manifest))
        elif args.command == "lease":
            result = lease(
                connection,
                worker=args.worker,
                worker_pid=args.worker_pid,
                lease_seconds=args.lease_seconds,
                source=args.source,
                phase=args.phase,
            )
        elif args.command == "heartbeat":
            result = {
                "lease_until": heartbeat(
                    connection, job_id=args.job_id, token=args.token, lease_seconds=args.lease_seconds
                )
            }
        elif args.command == "complete":
            result = {"status": complete(connection, job_id=args.job_id, token=args.token, result_path=args.result)}
        elif args.command == "retry":
            retry(connection, job_id=args.job_id)
            result = {"job_id": args.job_id, "status": "queued"}
        elif args.command == "list":
            result = list_jobs(connection, status=args.status)
        elif args.command == "attempts":
            result = list_attempts(connection, job_id=args.job_id)
        else:
            result = report(connection)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    except QueueError as exc:
        parser.error(str(exc))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
