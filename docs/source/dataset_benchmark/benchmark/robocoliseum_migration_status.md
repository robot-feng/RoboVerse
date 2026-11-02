# RoboColiseum task migration status

RoboVerse is migrating the AGIBOT Genie Sim Benchmark that powers RoboColiseum.
The local catalog contains 86 runnable configurations, 36 semantic tasks, 759
scene instances, and 854 task-instance assignments across instruction,
manipulation, spatial, robustness, and sim-to-real families. Eight configs use
the user's G1; 78 use G2. The imported source is Genie Sim 3.2.0, but the
checkout has no upstream Git revision. The catalog's source content hash is
`f0cd8166a1e0fba4739f8c15500262718e55ac3cdadc83b4ec0e47e41cddb8a4`.

The benchmark config files define the reference task, prompt, robot, assets,
scoring predicates, and robustness changes. The migration should preserve those
contracts while using RoboVerse task/scenario APIs and the three requested
engines: MuJoCo, Genesis, and Newton. A successful import or one simulation
step is only a smoke check; it is not task parity.

## Current verified state

The checked-in catalog is the authoritative status table in
`roboverse_pack/tasks/icl/benchmark_catalog.json`. All 36 semantic tasks are
currently `implemented_unverified`, all their MuJoCo/Genesis/Newton entries are
`not_run`, and overall behavioral and timing parity are `not_verified`. It
records zero current smoke-tested runs. Historical smoke runs used a previous
import path without task-specific reset poses or USD scene snapshots and do not
count as current validation.

The latest videos and reports in `examples/output/qa_runs/` cover loading,
nonblank rendering, camera projection, and held start states. Their benchmark
checker status is still `running`; none demonstrates task completion. The G1
`pick_block_color` attempts have not established a successful grasp. The
G2-only tasks must remain separately labeled until a G1 adaptation is explicitly
validated against the user's robot.

The user's two downloaded videos were decoded locally and indexed at
`examples/output/qa_runs/20261031_robocoliseum_reference_audit/`. Both are
640x400 VP9 at 30 FPS (4.70 s and 4.966 s). Clip 1 visually shows a hand moving
a green sphere among colored spheres; clip 2 shows bimanual interaction with
colored objects/blocks. Their generic filenames and streams carry no task ID or
instruction. They may resemble `pick_billiards_color` and `pick_block_color`,
respectively, but that mapping is unconfirmed. The RoboColiseum usage page could
not be fetched in this environment, so do not label these as a specific task or
success example without its task prompt/metadata.

## Phase 1: environment migration

The immediate milestone is to make the source scenes usable as RoboVerse
environments. It does not require grasping, motion planning, task completion,
or policy evaluation. For each selected environment, record:

1. pinned source revision, asset hashes, license/provenance, and missing inputs;
2. robot variant, scene instance, object set, authored start poses, and ground
   alignment;
3. successful scene construction and usable camera output on MuJoCo, Genesis,
   and Newton, with unsupported backends marked explicitly;
4. decoded scene-review video or image for each backend, including the chosen
   camera profile and resolution;
5. warnings or visual/model differences that need correction.

The G1 apple environment has a scene-smoke recording on all three backends in
`examples/output/icl_g1_apple_scene_qa/wall_removed_front_640/`. The GenieSim
catalog's 86 configurations and local payload wrappers establish static
inventory only; its current run-level backend matrix remains `not_run`. The
Imitator and SynthICL environment inventories must likewise keep file presence
separate from actual scene loading. Do not label a catalog entry as
environment-verified until that specific scene has a backend record.

Phase 2 can add task execution after the environment coverage is in place:
source action traces or a controller, source and RoboVerse predicate results,
trajectory/timing checks, and task-behavior videos. These are not Phase 1 gates.

## Later task-level evidence

For each runnable configuration and backend, record these separately:

1. source snapshot/hash, robot variant, scene/instance, instruction, seed, and
   initial object/robot states;
2. asset loading and camera output, with decoded video and image geometry;
3. a known action trace or controller run with joint/TCP trajectory and object
   state measurements;
4. the source benchmark predicates and RoboVerse checker result on the same
   terminal state;
5. reset reproducibility, timing/control rate, and backend-specific dynamics
   differences;
6. video/event comparison against a task-identified reference demonstration.

Track statuses as `PASS_SMOKE`, `PASS_TASK`, `FAIL`, `BLOCKED_ENV`,
`BLOCKED_ASSET`, `UNSUPPORTED`, or `NOT_RUN`. Only `PASS_TASK` with a decoded
video and source-predicate agreement can contribute to task parity. An unlabelled
video can be used to inspect appearance but not to establish the task identity,
target, or success criterion.

## Deferred task-behavior milestone

When Phase 1 environment coverage is ready, `g1op_s2r_select_color` is a
candidate for a later behavior milestone. It uses `pick_block_color` and its
source instructions specify the selected block and arm. That later phase would
need a scripted or replayed pick, source/RoboVerse checker agreement,
per-backend task videos, and trajectory/terminal-state comparison. It would
then expand to all eight G1 sim-to-real tasks and their 145 instances. The 78
G2 configs remain a separate robot variant and must not be described as G1
transfer without validation.

The two reference videos currently provide visual evidence only. They do not
include controller actions, state logs, calibration, or per-episode task text;
video similarity alone cannot establish sim-to-real physics alignment. Official
Genie Sim docs describe the same benchmark runtime behind RoboColiseum and the
sim/real data-by-evaluation design, but they do not certify this RoboVerse port:
[benchmark usage](https://github.com/AgibotTech/genie_sim/blob/main/source/geniesim_benchmark/USAGE.md),
[official benchmark overview](https://github.com/AgibotTech/genie_sim/blob/main/README.md).
