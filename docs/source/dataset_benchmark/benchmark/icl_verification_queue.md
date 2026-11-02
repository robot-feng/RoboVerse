# ICL verification queue

`tools/icl/benchmark_queue.py` provides a small local SQLite queue for ICL
verification work. It tracks test attempts; RoboVerse task and source catalogs
remain the source of truth. The database uses WAL and should stay on local disk,
not a network share. Put per-job logs, videos, and result JSON under an isolated
QA run directory. Keep `examples/output/ICL_Benchmark/` reserved for artifacts
that have passed release checks and human review.

Run from the RoboVerse repository root. Each job binds source revision, task,
variant, scene, embodiment, backend, seed, RoboVerse commit, and verification
phase. A field that does not apply must still have an explicit value such as
`"none"`. Dependencies refer to earlier job IDs and only become leaseable when
their jobs have a successful status.

```powershell
$db = "examples/output/qa_runs/icl_benchmark_queue.sqlite"
python tools/icl/benchmark_queue.py --db $db init
python tools/icl/benchmark_queue.py --db $db enqueue path/to/verified_jobs.json
python tools/icl/benchmark_queue.py --db $db lease --worker smoke-cpu-01 --phase smoke
python tools/icl/benchmark_queue.py --db $db report
python tools/icl/benchmark_queue.py --db $db list --status blocked
python tools/icl/benchmark_queue.py --db $db attempts <job-id>
```

The JSON manifest is either an array of jobs or an object with a `jobs` array.
Every job has these key fields:

```json
{
  "source": "geniesim",
  "upstream_sha": "<verified source revision>",
  "task_id": "<upstream task ID>",
  "variant": "<level or explicit none>",
  "scene_id": "<upstream scene ID>",
  "embodiment": "<robot ID>",
  "backend": "mujoco",
  "seed": 0,
  "roboverse_sha": "<RoboVerse commit>",
  "phase": "smoke",
  "owner": "Smoke-QA",
  "dependencies": [],
  "requirements": {"cpu_cores": 1, "ram_gb": 2},
  "artifact_dir": "examples/output/qa_runs/<unique-job-id>"
}
```

Valid phases are `migrate`, `smoke`, `semantics`, `demo`, `parity`, and
`release`. Result statuses identify the achieved gate (`imported`,
`smoke_passed`, `semantics_passed`, `demo_verified`, `parity_passed`, or
`published`), or explicitly record `failed`, `blocked`, or `unsupported`.
`complete` accepts only a nonempty, atomically written result JSON whose full
job key matches the lease. The result must also record `job_id`, `status`,
`exit_code`, the executed `command`, `artifact_dir`, and `asset_hashes`; use an
empty hash object only when no assets apply. The queue hashes the registered
result file and does not mark a physical task successful based on a screenshot
or a smoke test.

Leases have opaque tokens and heartbeats. An expired lease can be reclaimed;
the prior attempt is retained with its worker ID, PID, and start/end interval,
and a stale token cannot write the reassigned job. Supplying `gpu_id` in
`requirements` reserves that GPU until the lease is completed or expires. For
accurate process IDs, claim and run the simulator from the same worker process.
The queue does not launch simulators, implement a task registry, or publish
human-approved demos; those remain separate verification and release steps.

The queue contract is covered by `tests/test_icl_benchmark_queue.py`, including
idempotent enqueue, dependency gating, concurrent claims, expired-lease
recovery, result provenance, GPU mutual exclusion, and retry history.
