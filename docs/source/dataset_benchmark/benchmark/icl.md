# Reference-conditioned ICL prototype

`icl.push_to_region` and `icl.pick_place_region` are RoboVerse tasks derived from IG-10K assets, not an
official The Imitator Game benchmark. Both colored target regions remain visible
in every query; the checker chooses the target privately. The intended evaluation
compares the same query seed with left and right demonstrations. The present
implementation includes two single-object tasks and an object-choice prototype;
it is not evidence of generalization to unseen task families.

The GenieSim/RoboColiseum migration scope and three-engine verification status
are tracked separately in [RoboColiseum task migration status](robocoliseum_migration_status.md).

## Project phase 1: build and verify environments

The current phase is environment construction in RoboVerse. It covers source
asset provenance and availability, robot and object model loading, scene layout
and floor alignment, materials and lighting, configurable camera views, and
runtime/viewer checks on MuJoCo, Genesis, and Newton. Every task also needs
success and failure predicate tests on constructed states. Use the selected
three-quarter overview for scene review; task observations can independently
select the robot's head RGB and fisheye cameras.

Phase 1 does **not** require grasping, motion planning, a successful task
trajectory, or training-data collection. A Phase 1 pass means the registered
task loads, resets, advances physics, produces finite states, correctly
classifies constructed positive and negative predicate cases, renders usable
camera images, and can be inspected through the interactive Viewer. It is not
evidence of successful manipulation or cross-simulator behavioral parity.

The current environment inventory is:

| Source | Available in RoboVerse | Phase 1 status |
| --- | --- | --- |
| Supplied G1 apple scene | Prepared G1 robot, table, apple and plate; preferred overview and three head views | Old scene previews exist for three engines; current asset-bound runtime, task-predicate and Viewer checks remain incomplete |
| GenieSim benchmark | 86 configurations, 36 task pairs, 759 scene instances and local wrappers for 146 payload IDs | Static coverage only; run-level backend checks remain `not_run` |
| The-Imitator-Game | H7/H19 adapters and YCB wrappers; pinned RoboTwin basket/tissue source files are local and hash-verified, and a URDF/MJCF converter is integrated; converted wrappers are not generated yet. Adapters use a Franka pair | Source/adapter inventory only; no simulator scene loads recorded |
| SynthICL pseudo-demo | Four prepared object wrappers and four static scene profiles; G1/G2 with optional Franka, configurable head cameras | Static reset/render probes are implemented; the full scene × robot × backend matrix is still pending |

These rows describe distinct kinds of evidence. A file or wrapper being present
does not mean a task loads. Expand environment coverage by recording each
selected task, scene, robot variant, backend, and camera separately. Test task
predicates independently from manipulation trajectories; planning and grasp
checks remain outside this phase.

The detailed upstream source inventory and per-task/config/scene/robot/backend
matrix can be regenerated locally with `python tools/icl/build_phase1_inventory.py`.
Its JSON and CSV reports are written under
`examples/output/qa_runs/icl_phase1_inventory_20261102/`; they are local QA
artifacts, not repository source. The reports keep static source coverage
separate from task-loading, reset/step, predicate, rendering, and viewer results.
Only a concrete task/scene/robot/backend combination with
reproducible, asset-bound Phase 1 evidence
should be added to the human-review gallery in
`examples/output/ICL_Benchmark/environments/`; that gallery contains rendered
previews and a short README, while QA evidence remains under
`examples/output/qa_runs/`.

Asset availability is separate from redistribution permission. The local
SynthICL pseudo-demo source and object package do not provide a verified
license record, and the user-supplied apple-scene bundle does not specify one.
The Oxford VGG DTD page points to terms granting researchers a non-transferable
licence for strictly non-commercial research for 12 months; do not redistribute
its images, and verify the applicable term for the local download. Keep
unlicensed assets local and out of published benchmark artifacts until their
terms are confirmed. The current inventory also does not authorize copying the
upstream pseudo-demo implementation into RoboVerse ([VGG dataset terms of
access](https://www.robots.ox.ac.uk/~vgg/terms/dataset-group-2-access.html)).

## Deferred Phase 2: first object-choice experiment

`icl.pick_object` is a small controlled task inspired by LIBERO-Object, not an
official LIBERO task. Both a red and a blue cube appear in the same scene with
a single destination region. The private checker selects one candidate and
requires that candidate to be lifted with the gripper commanded closed, released
and settled in the destination, while the other candidate remains outside it.
The lift check is a state-based proxy; it does not measure finger contacts.
The same scene has the same observation for both goals. `layout=1` swaps their
initial positions, providing a simple control against a fixed-position shortcut.
The region is a visual marker, not a physical basket; these colored cubes do not
test generalization to new object shapes or realistic scenes.

Run the scripted physical probe from the repository root, using a new output
directory for each run:

```powershell
python tools/icl/probe_object_choice.py --sim mujoco --target red --layout 0 --output examples/output/icl_object_probe
python tools/icl/probe_object_choice.py --sim mujoco --target blue --layout 0 --output examples/output/icl_object_probe
python tools/icl/probe_object_choice.py --sim genesis --target red --layout 1 --output examples/output/icl_object_probe
python tools/icl/probe_object_choice.py --sim newton --target red --layout 1 --output examples/output/icl_object_probe
```

The probe writes a 30 FPS video and initial/final object states and success to
JSON, including failures. It is an oracle using privileged object positions; it
is not a policy evaluation or a SynthICL training-data exporter. The recorded
initial RGB hash is calculated before MP4 encoding; the lossy preview video is
for inspection and must not be used as evidence that two raw observations match.
For the intended
context-swap experiment, use exactly the same query state for both held-out red
and blue references and report the two outcomes together. Test swapped layouts
and independently held-out object assets and scenes as separate conditions;
switching positions alone does not demonstrate generalization to unseen scenes.
Include wrong-context and no-context controls. The current collector and model
evaluator do not yet support this new task, so these model experiments remain
unimplemented and must not be inferred from the scripted probe.

The task uses MuJoCo, Genesis, or Newton, a 30 Hz control period, world-frame
Cartesian TCP deltas and a closed/open gripper command. MuJoCo is also used for
Franka forward and inverse kinematics when another physics backend is selected.
For compatibility with the upstream SynthICL image keys, `images_wrist` contains
an **external overhead** camera, not a wrist-mounted camera; the metadata records
the actual camera roles. The generated object wrappers use a convex collision mesh
and approximate inertia and do not reproduce original asset dynamics exactly.
The goal checker requires the object's center to be inside the selected square
and both its linear and angular speeds to be below the configured thresholds.
The pick-and-place checker additionally requires the object to have risen above
10 cm while the gripper was commanded closed and to be released at the goal. This is a
state-based proxy for grasping; it does not verify finger contact forces.

## Assets and collection

Use the existing RoboVerse separation between assets, datasets, and run outputs:

```text
roboverse_data/icl/objects/       prepared simulation assets and provenance
datasets/icl/sources/             downloaded third-party datasets, kept in source format
datasets/icl/generated/<name>/    generated raw episodes and relative-link pair files
examples/output/qa_runs/          development-only smoke tests and logs
```

These paths are Git-ignored. `ROBOVERSE_DATA_DIR`, when set, replaces the
`roboverse_data` asset root, matching MetaSim's convention. The source and
generated dataset directories can be relocated independently; keep each
dataset's `raw/` and pair directories together so their relative HDF5 links
continue to work. Do not publish third-party assets or demonstrations without
checking the original terms.

On Windows, run the following from the repository root before downloading if
the cache and temporary files should also stay on the repository's drive:

```powershell
$Cache = Join-Path (Get-Location) "datasets/icl/cache"
New-Item -ItemType Directory -Force "$Cache/huggingface/hub", "$Cache/huggingface/xet", "$Cache/torch", "$Cache/tmp" | Out-Null
$env:HF_HOME = "$Cache/huggingface"
$env:HF_HUB_CACHE = "$Cache/huggingface/hub"
$env:HF_XET_CACHE = "$Cache/huggingface/xet"
$env:TORCH_HOME = "$Cache/torch"
$env:GS_CACHE_FILE_PATH = "$Cache/genesis"
$env:TEMP = "$Cache/tmp"
$env:TMP = "$Cache/tmp"
```

### Genie Sim assets and AgiBot G1 inspection

[Genie Sim assets](https://huggingface.co/datasets/agibot-world/GenieSimAssets)
include an AgiBot G1 model and simulation objects. This G1 is distinct from
the Unitree G1 configurations already in RoboVerse. The pinned source library
is about 29.4 GB, excluding caches, and is subject to CC BY-NC-SA 4.0 and
the upstream asset terms. Its separate demonstration datasets are not needed
for the following inspection. Set the D-drive caches above, then download the
asset snapshot and generate the prototype from the local Genie Sim source:

```powershell
$env:HF_HUB_DISABLE_IMPLICIT_TOKEN = "1"
$GenieAssets = "datasets/icl/sources/geniesim-assets"
uv pip install -e ".[icl,icl-geniesim,mujoco,genesis,newton]"
python tools/icl/download_geniesim_assets.py --output $GenieAssets --workers 2
$G1Urdf = "../icl/genie_sim-main/genie_sim-main/source/geniesim_benchmark/src/geniesim_benchmark/robot/isaac_sim/robot_urdf/G1_120s.urdf"
$G1Asset = "roboverse_data/icl/robots/geniesim_g1_120s_v7"
python tools/icl/prepare_geniesim_g1.py --source $GenieAssets --urdf $G1Urdf --output $G1Asset
python tools/icl/probe_geniesim_g1.py --sim mujoco --asset $G1Asset --output examples/output/g1_mujoco.png
$Apple = "roboverse_data/icl/objects/geniesim_apple_aligned"
python tools/icl/prepare_geniesim_apple.py --source $GenieAssets --output $Apple
python tools/icl/probe_geniesim_g1.py --sim mujoco --asset $G1Asset --apple $Apple --view head_rgb --head-pitch 0.3 --output examples/output/g1_apple_head_rgb.png
```

The downloader resumes an interrupted snapshot, alternates the official and
public mirror on connection failures, and checks all source hashes after the
download. Run it again to resume; `--status` checks presence and file sizes,
and `--verify` also reads and hashes every source file. Neither sends local Hub
credentials to the mirror. A complete snapshot can take considerable time. If a
prepared output already exists, skip its preparation command or select a new
output path; preparation deliberately refuses to overwrite it.
Use `--prefer-mirror` to try the public mirror first when the official endpoint
is slow; final checksum verification still uses the official pinned manifest.

The preparation command requires an official pinned-revision manifest at
`$GenieAssets/roboverse_source_manifest.json`, verifies the G1 USD inputs against
its hashes, and refuses to overwrite an existing output. It uses the local Genie
Sim benchmark's MPL-2.0 URDF and records its checksum and asset provenance. The
prototype extracts mesh geometry, material subsets, diffuse colors and authored
normals from the geometry layer, then applies each link's visual attachment transform
from the composed `robot.usda` layer. This distinction matters for G1 Omnipicker:
`base_link/visuals` is authored 0.2111347 m above the collision link so the rendered
mobile base sits on the ground; omitting that overlay makes the chassis appear buried
in the floor. The converter records this transform in the prepared v4 asset and keeps
it visual-only, leaving the source collision geometry and robot joint frames unchanged.
It also adds both wrist-camera housings from the pinned `Cam.usd` payload. It must not
reuse the URDF's original FBX mesh origins on the already-aligned USD geometry. The prototype does not reproduce the
original PBR shader, and collision geoms are hidden
from the MuJoCo inspection view while remaining active in simulation. It retains
the URDF inertias, substitutes source visual geometry for mesh collisions, and
uses generic placeholder MJCF actuator gains; gripper
coupling and physical behavior have not been validated. The probe is a static
external view, not an ICL task or a real-robot model.

The source camera metadata records the head RGB and two head fisheye transforms
and authored lens attributes. Mounted pinhole cameras now follow their link in
MuJoCo, Genesis and Newton; the three selected backends do not reproduce the
source fisheye projection. The external inspection view
is not one of those cameras. The optional MuJoCo `head_rgb` preview uses the
source head-link transform and focal/aperture ratio; `--head-pitch` sets a
kinematic preview joint state. This does not establish real-camera calibration.
The apple converter preserves the source mesh's UVs and downsamples its color
texture; its convex collision hull and mass are approximations and its source
normal map is not applied. A G1 benchmark needs real-camera calibration, control
and task integration before collecting training data; do not label pinhole
stand-ins as calibrated fisheye images. Existing apple
assets prepared before the world-frame conversion was corrected should be rebuilt
at a new path; do not reuse the old `geniesim_benchmark_apple_000` wrapper.

### Supplied G1 apple delivery scene

The user-supplied `g1_apple_scene_delivery_20261031_v4.zip` is imported as a
separate RoboVerse task, `icl.geniesim_g1_apple_delivery`. The verified source
archive is recorded by SHA-256 in
`roboverse_pack/tasks/icl/benchmark_extensions.json`; this keeps the extension
separate from Genie Sim's pinned 86-run catalog. The preparation script checks
the archive manifest and every listed file hash before extracting or converting
the apple, plate, table and supplied G1 start pose. Prepared runtime assets live
under `roboverse_data/icl/scenes/`, which is local generated data rather than a
large source-control payload.

The current portable conversion is revision `r4`; object link names match the
task's stable `apple` and `pink_plate` state keys, including in Newton. It omits
the source floor cube where it would overlap MetaSim's ground, and uses a
Newton-specific visual mesh to remove overlapping coplanar plate faces. The
plate collision mesh is unchanged. The source rear wall and floor cube are
omitted; the checker ground is provided by MetaSim, keeping the view focused
on the G1, table, apple, and plate. The refreshed three-backend probe records
all four views at 640 pixels for 60 frames under
`examples/output/icl_g1_apple_scene_qa/wall_removed_front_640/`; its overview
comparison is `overview_first_frames.png`. These short hold-and-head-pan
recordings do not demonstrate a pick or verify behavioral parity.

The reset layout puts the tabletop, apple base, and plate base at z=0.78 m.
The G1 root is at z=-0.001215 m and its ground-aligned base geometry reaches
z=0 m, so the chassis rests on the floor in the configured start pose.
Genesis also reports geometry-based inertial diagnostics for this asset.
Of the 42 MJCF bodies, 39 have direct authored inertials. The other three are
geometry-free welded helper bodies; Genesis assigns an epsilon mass floor to
such links when they belong to a moving chain. Non-watertight-mesh warnings
come from geometry-based inertial consistency estimates, but the log does not
identify the individual mesh paths. The `body_link2` center-of-mass warning
remains unresolved against the source geometry and is not suppressed.

From the RoboVerse root in PowerShell:

```powershell
python tools/icl/prepare_geniesim_g1_apple_delivery.py `
  --archive ../g1_apple_scene_delivery_20261031_v4.zip
```

The task shares the prepared G1 Omnipicker URDF/MJCF and exposes head RGB, both
head fisheye poses and an external overview. Run the renderer probe separately
for each simulator; it writes one MP4 for each of the four views plus a JSON
report to a new output directory:

The default external overview uses the preferred rear-side three-quarter view
at `pos=(1.7, -1.8, 1.9)`, looking at `(0.0, 0.15, 0.75)` with a 20 mm focal
length. It places the robot to the left of the table and keeps the apple and
plate visible without the source rear wall. The three head-mounted cameras
remain the robot observation views; the overview is for scene inspection.

```powershell
python tools/icl/probe_geniesim_g1_apple_delivery.py --sim mujoco `
  --output examples/output/qa_runs/icl_g1_apple_delivery_r4/mujoco
python tools/icl/probe_geniesim_g1_apple_delivery.py --sim genesis `
  --output examples/output/qa_runs/icl_g1_apple_delivery_r4/genesis
python tools/icl/probe_geniesim_g1_apple_delivery.py --sim newton `
  --output examples/output/qa_runs/icl_g1_apple_delivery_r4/newton
```

The scene-render probe defaults to 640 pixels wide so the apple, plate, and
robot details remain inspectable. The older 256-pixel QA clips used a
policy-sized capture, which made the small objects look as if they had no
surface detail. The source USD defines the table and pedestal with flat
`displayColor` values; those two parts are intentionally untextured upstream.
The apple and plate use their source color maps. Source normal and full PBR
maps are not part of this portable scene conversion.

For an interactive inspection of the registered task, launch the existing
Viser task inspector with `icl.geniesim_g1_apple_delivery` and the MuJoCo
backend. The inspector exposes the task's bounded 20-joint target action,
reset, step, pause/run controls, and the three head cameras plus overview. Its
task predicate is tested separately with constructed states; this is not a
claim that the G1 can yet complete the placement through a grasp trajectory.

The prepared source robot has a small collision/visual mismatch at the floor:
one conservative `base_link` box extends 2.58 mm below the visible chassis.
Derive the ground-aligned G1 asset from the source-driven model before running
the apple scene. This shifts only the base collision proxies to the composed
visual floor and records the correction in provenance; the robot visuals and
the imported scene pose stay unchanged.

```powershell
python tools/icl/prepare_geniesim_g1.py `
  --align-base-ground-collision-from-asset roboverse_data/icl/robots/geniesim_g1_omnipicker_closed_v10 `
  --output roboverse_data/icl/robots/geniesim_g1_omnipicker_closed_v11
```

G1 fruit-sorting and GenieSim tasks use v11 by default. A ground-alignment
regression checks that both URDF and MJCF collision bounds meet the authored
base visual at the scene ground.

These earlier v5 G1 renders are 640×403. H.264's portable `yuv420p` encoding
requires even dimensions, so the review MP4 pads one edge row to 640×404; the
simulation image itself stays 640×403 and the JSON report records both sizes.

These short clips hold the delivered start pose and pan the head. They validate
scene construction, nonblank rendering and a short stable hold; they are not
successful pick-and-place demonstrations. The state checker requires the
apple center to remain inside the plate's goal bounds for two observations; it
does not assert that the robot grasped the apple. Table size, apple mass, plate
collision geometry and inertias are portable approximations, and the supplied
camera/joint calibration has not been checked on a physical G1. Do not treat
this extension as task-complete until a contact-aware scripted or learned
trajectory succeeds and is reviewed on each backend.

The original archive does not specify an asset license. Keep it local until its
redistribution terms are confirmed; the extension catalog records that status.

### G2 dual-arm robot model

Some upstream benchmark configurations specify the dual-arm G2. The generated
model therefore keeps the CRS arms and Omnipicker grippers together as one
robot. The G1 and G2 are different robots: a successful G2 run is not a G1
or real-robot validation, and using a G2-only task with G1 requires a separate
adaptation and reach/contact checks. The generated G2 model comes from the
pinned [AgiBot Genie G2 robot model](https://github.com/AgibotTech/genie_sim_robot_model)
revision `6d5b2159806ae0e64b432b638e9ee288d8bddc8f`; the AgiBot G2 and Omnipicker
assets are Apache-2.0. The Wuji hand is a separate MIT component and is excluded.

On Windows, download only the required robot description and mesh tree to the
repository drive, then prepare portable URDF/MJCF files under `roboverse_data`:

```powershell
$ModelRoot = "datasets/icl/sources/genie_sim_robot_model"
git clone --filter=blob:none --sparse https://github.com/AgibotTech/genie_sim_robot_model.git $ModelRoot
git -C $ModelRoot sparse-checkout set xacro robots/genie/g2
git -C $ModelRoot checkout 6d5b2159806ae0e64b432b638e9ee288d8bddc8f
uv pip install -e ".[icl-geniesim]"
python tools/icl/prepare_geniesim_g2.py `
  --source $ModelRoot `
  --output roboverse_data/icl/robots/geniesim_g2_omnipicker_v11 `
  --benchmark-control-limits
```

The v11 converter turns upstream Collada visuals into STL submeshes and keeps
their diffuse colors in URDF materials. It reads the actual benchmark
`robot_fix.usda` and verified referenced layers from the pinned GenieSimAssets
snapshot. Joint drive values, disabled collision shapes, zero-target drives on
other joints, and gripper loop constraints follow those source layers. The
snapshot, including the asset manifest and the adjacent `robot.usda` and
`robot_fix.usda` files, must be available before conversion.
MuJoCo's `balanceinertia` correction and the `1.1e-6 kg·m²` lower inertia bound
are recorded in `provenance.json`. The model remains a converted approximation:
its source inertia triangle violations are corrected, finger collision meshes
come from the Xacro model, and it does not reproduce the source simulator's complete
actuator/contact stack. Earlier three-backend smoke tests used the
legacy scene and reset state; the current source-scene and named-reset snapshot
requires fresh tests.
Neither a smoke test nor a successful scene launch establishes source visual or
behavioral parity.

The drive settings come from the pinned GenieSimAssets revision
`a0813aad7c16165daffbe6c2737754e0344809ee`, not the independently imported
benchmark code. For a previously prepared v7 model with the old incorrect drive
revision, correct its metadata without rebuilding the model:

```powershell
python tools/icl/prepare_geniesim_g2.py `
  --correct-drive-provenance `
  --output roboverse_data/icl/robots/geniesim_g2_omnipicker_v7
```

This verifies the source manifest, USD, extracted drives and prepared model
files before updating provenance; it retains the previous revision in the
correction record. It fails if the required inputs cannot be verified.

Prepare the G2 rigid scene assets and the source laptop/door articulations from
the downloaded snapshot with:

```powershell
$GenieAssets = "datasets/icl/sources/geniesim-assets"
$GenieSpec = "roboverse_data/icl/geniesim_benchmark_specs_source_scene_v2/benchmark.json"
python tools/icl/prepare_geniesim_objects.py `
  --source $GenieAssets `
  --spec $GenieSpec `
  --robot g2_op `
  --flatten-articulated-static `
  --output roboverse_data/icl/objects/geniesim
python tools/icl/prepare_geniesim_articulations.py `
  --source $GenieAssets `
  --output roboverse_data/icl/objects/geniesim_articulated `
  --asset-id benchmark_laptop_000 `
  --asset-id benchmark_door_000
```

The rigid converter resumes verified assets and lists unavailable source USDs
in `roboverse_data/icl/objects/geniesim/prepare_failures.json`. The v2 scene
snapshot resolves each object's asset from that instance's authored USD payload.

Some checkers address a child prim rather than the object's root. In
`stock_and_straighten_shelf`, both referenced beverage `entity` frames have an
authored rotation; the checker uses the verified source-relative frame instead
of silently treating it as the root pose. Newly converted objects record these
frames in provenance. To add them to existing prepared beverage assets, verify
their pinned source files and refresh their provenance with:

```powershell
python tools/icl/prepare_geniesim_objects.py `
  --source $GenieAssets `
  --output roboverse_data/icl/objects/geniesim `
  --asset-id genie_beverage_bottle_008 `
  --asset-id genie_beverage_bottle_009 `
  --refresh-subframes
```

The sorting-packages checkers reference two fixed objects in the warehouse
background. Their world bounds are read from the original environment USD for
evaluation. Warehouse background collision meshes are also generated from the
source USD, with nonconvex parts approximated by convex decomposition. A block
drop probe has passed in MuJoCo, Genesis (CPU backend), and Newton after the
warehouse scene enabled multiple mesh contacts. This single contact probe does
not establish full task success or equality with source PhysX friction.
For relative-position checks, converted local geometry bounds are transformed
to a world AABB. This is an approximation for non-box meshes and moving
articulated parts; it is not a live source-USD world-bounds query.

### Full benchmark migration inventory

The local source snapshot contains **86 run configurations**, covering **36
robot/subtask combinations** and **759 task-scene instances**. Eight run
configurations use the source G1 model; 78 use the dual-arm G2. Fifty configs
are five robustness variants of ten instruction tasks. The two supplied
RoboColiseum clips show dual-arm table scenes and therefore serve as visual
references for the G2 task family; they do not verify the single-arm G1 fruit
prototype. Keep source G2 runs on G2 when measuring parity with those clips.
The source task graphs use 14 distinct checker actions (including Follow,
PickUpOnGripper, PushPull, Stack and VLM). The shared RoboVerse checker
evaluator compiles all 14 and preserves multi-object selection and the source
`placeholder_str1` hand-off between checker stages. An older import was
smoke-tested on instance 0/problem 0 for each run and backend. Those checks
covered reset, camera output and one control step using the old scene and robot
reset data. The USD-scene and task-specific reset changes require a new
three-backend run; the catalog marks the revised configurations as
`implemented_unverified` until that evidence exists.

Record a start-pose scene video for one canonical run of each of the 36 semantic
tasks on all three backends. The default `hardware` camera profile provides
head RGB and both head fisheye poses to match the intended AgiBot G1 observation
layout; it is not the G2 benchmark's head-plus-wrist inference stream. This
writes one video per configured camera and a machine-readable report under each
task/backend directory; it resumes completed entries. Clips hold source joint
targets and are for checking scene/camera appearance, not task success:

```powershell
python tools/icl/record_geniesim_benchmark_video.py `
  --all-semantic-tasks `
  --simulators mujoco,genesis,newton `
  --camera-profile hardware `
  --image-size 640 `
  --frames 45 `
  --resume `
  --matrix-output examples/output/icl_geniesim_video_matrix
```

The semantic matrix covers one source run per semantic task, not all robustness
variants or all 759 scene instances. Use a specific `--run-id`, `--instance-index`
and `--problem-index` for focused inspection of a source configuration.

| Configuration family | Count | Current migration status |
| --- | ---: | --- |
| G1 skill-to-reason tasks (`s2r`) | 8 | shared runner implemented; revised scenes and resets require backend validation |
| G2 instruction following (`if`) | 10 | shared runner implemented; revised scenes and resets require backend validation |
| G2 manipulation (`manip`) | 10 | shared runner implemented; revised scenes and resets require backend validation |
| G2 robustness variants | 50 | shared runner implemented; each perturbation requires backend validation |
| G2 spatial tasks | 8 | shared runner implemented; revised scenes and resets require backend validation |

In eight `pick_object_type` scenes, `scene_info.json` names
`waic_beverage_bottle_009` while the same object's `scene.usda` explicitly
references `genie_beverage_bottle_009`. The v2 import follows each scene's USD
payload and records its source path and hash. This does not establish that the
two asset IDs are interchangeable in other scenes. The older metadata-only
import used a dimension-matched, identity-unverified compatibility alias.

The run-by-run inventory, SHA-256 of each upstream YAML, scene
family and test status is stored in
`roboverse_pack/tasks/icl/benchmark_catalog.json`. Its current status fields
explicitly distinguish smoke tests from behavioral parity. Regenerate a review
copy from a local source checkout with:

```powershell
python tools/icl/build_task_catalog.py `
  --source-root ../icl/genie_sim-main/genie_sim-main/source/geniesim_benchmark/src/geniesim_benchmark `
  --imported-spec roboverse_data/icl/geniesim_benchmark_specs_source_scene_v2/benchmark.json `
  --output examples/output/qa_runs/benchmark_catalog_review.json
```

To import the full task payload for local RoboVerse development, run:

```powershell
$GenieCode = "../icl/genie_sim-main/genie_sim-main/source/geniesim_benchmark/src/geniesim_benchmark"
python tools/icl/import_geniesim_specs.py `
  --source-root $GenieCode `
  --output roboverse_data/icl/geniesim_benchmark_specs_source_scene_v2
```

This writes the 86 source run YAMLs, 36 robot/subtask groupings, 759 scene
instances, 23 evaluation configurations, 854 semantic-task/scene assignments,
checker graphs, instructions, per-scene USD payload snapshots, task-specific
robot reset poses and 146 active USD payload asset IDs into
`roboverse_data/icl/geniesim_benchmark_specs_source_scene_v2/benchmark.json`.
The source files stay in the local Genie Sim directory and the object assets stay
in the pinned asset source directory. The importer does not copy demonstrations,
videos or training trajectories. The generated directory is Git-ignored and the
importer refuses to overwrite an existing output.

The supplied source is an extracted directory without its own Git metadata, so
its benchmark repository revision is `unknown`. The importer records SHA-256
hashes for 3,153 task and reset source files, a source-content fingerprint, and a
separate imported-payload fingerprint. Reset tables are parsed as static
literals without executing upstream Python. Changes to the reviewed reset
semantics cause an explicit import error. Nineteen source scene instances contain
non-unit object scaling and are currently rejected by the runtime pending a
backend-consistent implementation.

### Shared checker graph runtime

`roboverse_pack.tasks.icl.geniesim_checker` compiles the imported action trees
(`ActionList`, wait-any/all/some, time limits and all 14 source checker types)
into a persistent evaluator. It selects numbered problems in the same order as
the source task's instruction list and can also compile per-instruction
`subtask_steps`. Geometric checkers use world poses and prepared local object
bounds; gripper-sensitive checkers receive named left/right gripper poses;
articulated `PushPull` checkers receive the source-ordered joint positions.
VLM checkers require caller-supplied scores in each `CheckerFrame`; this module
does not start a vision model. `checker_frame_from_states` adapts RoboVerse's
shared MetaSim tensor state, including named gripper link poses. Genesis now
reports the same sorted per-link pose/velocity layout as MuJoCo and Newton.
`icl.geniesim_benchmark` is a registered G1/G2 task that instantiates a selected
source run, exposes 20 G1 or 24 G2 joint-position targets, and returns the
source checker status and instruction in `info`. Its default observations use
head RGB and both head fisheye camera poses. The fisheye projections use G2's
source optics temporarily for G1; these are not measured physical G1 parameters.

The old imported scene data was smoke-tested with instance 0/problem 0 of all
eight G1 source runs on MuJoCo, Genesis and Newton. Those checks are historical:
the revised USD scene snapshots and task-specific reset poses have not yet been
validated for every run and backend. Source door and laptop objects retain joints
through converted URDF/MJCF, while their inertial, collision and material
properties are approximations. NuRec/Gaussian-splat rooms, calibrated fisheye
distortion and VLM scoring still need separate adapters.

Compile every imported task graph and subtask checker from the RoboVerse root:

```powershell
python -m tools.icl.validate_geniesim_specs `
  --spec roboverse_data/icl/geniesim_benchmark_specs_source_scene_v2/benchmark.json
```

The v2 local source snapshot reports revision `unknown` and its checker audit
compiles 759 scene instances, 1,367 problem graphs and 460 subtask steps. This
checks graph structure; it does not prove success predicates or physical
behavior match the source. Each task needs closed-loop execution, recorded
outcomes and cross-backend comparison. The source's VLM checks additionally
need an explicitly selected image scorer.

Once the asset snapshot is complete, prepare the G1 object assets used by the
benchmark. The drawer is explicitly flattened at its source pose as static
geometry; other articulated motion is not represented:

```powershell
$GenieAssets = "datasets/icl/sources/geniesim-assets"
$GenieSpec = "roboverse_data/icl/geniesim_benchmark_specs_source_scene_v2/benchmark.json"
python tools/icl/prepare_geniesim_objects.py `
  --source $GenieAssets `
  --spec $GenieSpec `
  --robot g1_op `
  --flatten-articulated-static `
  --output roboverse_data/icl/objects/geniesim
```

For the first G1 `sort_fruit/0` scene only, convert just its five unique assets
to get a smaller initial preview:

```powershell
python tools/icl/prepare_geniesim_objects.py `
  --source $GenieAssets `
  --output roboverse_data/icl/objects/geniesim `
  --asset-id benchmark_lemon_027 `
  --asset-id benchmark_peach_000 `
  --asset-id benchmark_storage_box_008 `
  --asset-id benchmark_storage_box_010 `
  --asset-id table_000
```

The command verifies each USD, description and texture against the pinned asset
manifest, then resumes already valid outputs on later runs. Unsupported assets
are listed in `roboverse_data/icl/objects/geniesim/prepare_failures.json`.
The converter preserves rigid USD meshes and face-bound diffuse material
subsets. For explicitly selected specs, articulated assets can be flattened
at their authored pose into separate static mesh parts when no articulated
converter is available. The G1 drawer cabinet uses this static fallback; it
does not preserve drawer motion. G2 doors and laptops use the separate
articulation converter shown above. Other PBR maps are omitted. Genesis preserves imported image
maps when its MJCF loader creates an implicit default material. Rigid-object
collision remains a coarse axis-aligned bounding box, so hollow-object contacts
and task success still need dedicated validation. These wrappers support
appearance and scene-load prototypes; they do not establish source-accurate
object dynamics or behavioral parity.

The scene adapter in `roboverse_pack.tasks.icl.geniesim_scene` turns an imported
source run and one of its scene instances into a RoboVerse `ScenarioCfg`. It
keeps the source object positions and orientations, reads the robot spawn pose
and default joint configuration from the run and evaluation configs, marks evaluation-declared fixed objects plus
objects categorized as tables/desks/counters/floor as static, and
selects MJCF for MuJoCo/Genesis and URDF for Newton. It writes small per-scene
XML wrappers under an ignored cache directory and reuses the converted mesh
files rather than duplicating them. It adds one shared neutral-gray static
floor and disables each backend's checkerboard ground; MuJoCo also uses the
configured scene-light rig. The floor is a portable visual/physics fallback,
not a reconstruction of the source room. Source PBR materials, the GenieSim
NuRec/Gaussian-splat room and identical light response are not reproduced, so
the current cross-backend views remain visibly different from the source
renderer. A Python caller can build a source
run as follows:

```python
from pathlib import Path

from roboverse_pack.tasks.icl.fruit_sorting import build_geniesim_g1_robot
from roboverse_pack.tasks.icl.geniesim_scene import build_run_scenario, geniesim_g1_cameras

robot_root = Path("roboverse_data/icl/robots/geniesim_g1_omnipicker_closed_v6")
scenario, source_run, source_instance, source_eval = build_run_scenario(
    spec_path=Path("roboverse_data/icl/geniesim_benchmark_specs_source_scene_v2/benchmark.json"),
    run_id="g1op_s2r_sort_fruit",
    simulator="mujoco",
    robot=build_geniesim_g1_robot(robot_root),
    assets_root=Path("roboverse_data/icl/objects/geniesim"),
    cache_root=Path("roboverse_data/icl/cache/geniesim_scene_wrappers_v6"),
    cameras=geniesim_g1_cameras(robot_root, image_size=320),
)
```

Run one imported G2 problem through the RoboVerse task API after preparing the
objects in the chosen scene. Replace `run_id` with a `g1op_...` run ID to use
the G1 model; the runner selects the robot and action dimension from the source
configuration:

```python
import torch

from metasim.task.registry import get_task_class
task = get_task_class("icl.geniesim_benchmark")(
    run_id="g2op_if_pick_billiards_color",
    simulator="mujoco",  # genesis or newton use the same task and action layout
    instance_index=0,
    problem_index=0,
)
try:
    observation, info = task.reset()
    targets = torch.tensor([
        task.scenario.robots[0].default_joint_positions[name] for name in task.control_joint_names
    ])
    observation, reward, success, timeout, info = task.step(targets)
    print(info["geniesim_instruction"], info["geniesim_checker_status"])
finally:
    task.close()
```

`task.reset()` also advances the scene through a source-like settling interval
before sampling the initial checker state. The default `reset_settle_seconds=2`
mirrors GenieSim's scene startup and robot-reset convergence waits; these
simulator steps are excluded from the episode action count. This matters for
objects authored above a support surface: in the billiards scene, the source
positions are at `z=0.894 m`, while the table contact centers are near `z=0.759 m`.
For deliberately exact, externally supplied states, pass `states=` to `reset()`
or set `reset_settle_seconds=0` when constructing the task.

The task registration can select any of the 86 source run IDs. Pass
`robot_variant="G1"` or `"G2"` to load the same scene with either robot;
omitting it uses the source configuration's robot. For a substituted robot,
the task uses that robot's source reset pose for the same task when present,
otherwise the source `pick_block_color` reference pose. The source run's base
pose is reused. `info["geniesim_robot_adaptation"]` records this choice; cross-
robot spawn safety, reachability, control and success are not yet validated.
Both variants default to the three head cameras.

Scene execution requires the referenced assets to be prepared. Eight `pick_object_type` scenes
with a `waic_beverage_bottle_009` metadata ID explicitly reference a
`genie_beverage_bottle_009` payload in their source USD; the imported scene
records and uses that payload. This does not establish a global equivalence of
the two asset IDs. Supported G2
door and laptop objects retain their joints through converted URDF/MJCF assets.
`problem_index` selects the corresponding source instruction/checker pair. The
default action is a 20-joint G1 or 24-joint G2 position command; select
`tcp_delta` when the policy uses Cartesian actions.

For SynthICL-compatible control, pass `action_mode="tcp_delta"`. This exposes
the same seven-value world-frame translation/rotation-vector/gripper-open
action used by `roboverse_learn.il.icl.data.ACTION_SPEC`; the task maps it to
G1's 20 or G2's 24 position targets through the corresponding kinematic
controller, and reports the selected action specification in
`info["geniesim_action_spec"]`. The controlled arm is selected from source
metadata, checker references or instruction text, with the right arm as the
fallback when no side is specified; `tcp_arm="left"` or `"right"` overrides
selection. Ambiguous or conflicting source references require an explicit
choice. For two-arm control, `action_mode="bimanual_tcp_delta"` accepts two
seven-value actions in left-then-right order. These controls use a configured
1/30-second control period, but cross-backend timing and control parity remain
under validation.

For a short, measured three-backend control diagnostic, run:

```powershell
python tools/parity/parity_geniesim_control.py --all `
  --output examples/output/qa_runs/geniesim_control_probe
```

This sends identical named G1 joint targets at each simulated timestamp and
records the head camera, observed gripper poses, joint states and object poses.
The output reports maximum differences without assigning a pass threshold or
claiming task success. Each backend runs in a separate process; use a fresh
output directory. The benchmark scene opts into MuJoCo's `implicitfast`
integrator because the same commands produced a large wrist transient with the
default integrator. Genesis floating MJCF objects receive an explicit free
joint when the source has none. On Genesis versions exposing the option, MJCF
imports retain source armature values rather than injecting Genesis's default
0.1 rotor inertia; older versions without the option retain their prior behavior.
These settings reduce observed differences for
the tested motion but do not establish task or contact parity.

For Cartesian actions, untouched joints hold their measured positions. If a
measured joint transiently exceeds its configured position limit, the internally
generated hold target is clipped to that limit; `info["geniesim_internal_target_clipping"]`
reports the measured position, unclipped target and applied target for each
affected joint. A nonempty report can indicate a dynamics or constraint issue
and is not evidence that the measured state respected the limit. Explicit
joint-position actions outside their limits are still rejected.

The source PiEnv/VecEnv checker evaluates at reset and before every 30th
action, rather than after every action. This adapter follows that cadence and
keeps the source action count for step-based checks. Time-based predicates use
completed simulation steps times the environment step duration; the source
uses wall-clock time, so equivalence under slow inference or rendering remains
unverified. The `geniesim_checker_action_step`,
`geniesim_checker_completed_sim_steps`, and `geniesim_checker_time_source`
fields expose the sampling state.

For a reproducible rendered inspection of an imported G1 run, prepare the
referenced rigid assets first, then run the same source instance on each
backend:

```powershell
python tools/icl/probe_geniesim_scene.py `
  --sim mujoco `
  --run-id g1op_s2r_sort_fruit `
  --output examples/output/qa_runs/geniesim_g1_mujoco
```

Use `--sim genesis` or `--sim newton` and a new output directory to compare
backends. The probe restores the source robot base pose, default joint state,
and object transforms; it saves head RGB, both head fisheye approximations,
and a separately labeled external inspection view. It refuses to overwrite an
existing output directory. This checks scene loading and rendering, not task
completion or visual parity.

To record an action replay rather than a held-pose probe, provide an action
sequence JSON file. It must specify the exact task identity, action mode,
physical step interval and source of the actions. For example:

```json
{
  "format": "roboverse.icl.benchmark_action_sequence.v1",
  "run_id": "g1op_s2r_select_color",
  "instance_index": 0,
  "problem_index": 0,
  "action_mode": "tcp_delta",
  "benchmark_source_content_sha256": "<source.content_sha256 from the imported benchmark snapshot>",
  "step_seconds": 0.03333333333333333,
  "source": {"type": "manual_example"},
  "actions": [[0, 0, 0, 0, 0, 0, 1]]
}
```

```powershell
python tools/icl/record_geniesim_benchmark_video.py `
  --sim mujoco --run-id g1op_s2r_select_color `
  --actions-json path/to/actions.json --fps 30 `
  --output examples/output/my_action_replay
```

Replace the fingerprint placeholder with `source.content_sha256` from the
imported benchmark snapshot. The recorder verifies that the fingerprint and
action period agree with the active snapshot and environment step
and video frame rate. It stores the input actions, source hash, videos, per-step
object and control states, checker status and clipping diagnostics. An action
exception is recorded as an exception outcome rather than task success. A
checker result alone does not establish source-checker or cross-backend parity.
Use `--robot-root` on a single run to select a prepared robot asset; the report
records its directory and hashes of the provenance, URDF and MJCF files. An
action sequence may include `robot_asset_sha256` with those three hashes to
reject replay with a different robot model. These hashes do not cover every
referenced mesh or texture.

`--robot-variant G1` or `--robot-variant G2` selects one robot for a separate
scene load. Reports record the source robot, runtime robot, reset adaptation,
camera mounts and solver settings. A sequence may declare its `robot_variant`;
legacy sequences without one are attributed to the source robot. Replaying a
sequence on a different runtime robot is rejected, including Cartesian actions.
The default hardware camera profile provides the head RGB and two head views;
the side views use the source G2 OpenCV fisheye parameters by default.

For auditing source physics, `prepare_geniesim_objects.py --physics-source usd`
can produce a **separate** v4 object directory. It records hashes for the USD
layers, the authored rigid-body mass, per-collider static and dynamic friction,
and emits the USD mass in both robot-neutral asset formats. MJCF uses the USD
dynamic coefficient as a single-coefficient proxy, or static if dynamic is not
authored. MuJoCo/Genesis and PhysX have different friction and contact-pair
combination rules; this is not exact material equivalence. The converter records
Newton friction but does not yet apply it. A v4 object alone also does not
include per-scene overrides, so it must not be substituted for the v3 benchmark
assets until scene override and backend handling are integrated and tested.

`geniesim_g1_cameras` reads pinned G1 camera poses and head RGB calibration.
The G1 side lenses temporarily borrow the corresponding G2 OpenCV fisheye
intrinsics and distortion coefficients, while keeping G1 mounting transforms.
The fruit-sorting task exposes these
three G1 views plus an external overview camera; it does not add wrist cameras.

The adapter loads the selected run's object layout, exact table asset and
source spawn poses. Genie Sim stores those poses relative to `workspace_00`,
so the adapter adds the source workspace's XY origin before placing the robot
and task objects alongside the source USD background. If the source USD
references a local HDR dome map, that map is tone-mapped and projected onto an
inward-facing visual sphere. The background layer's source rotation is applied,
and the sphere is kept within the supported renderers' practical far range.
Static source USD scenery meshes now retain world transforms, diffuse material
colors, UVs, and albedo textures across the portable URDF/MJCF paths. To keep
static backgrounds practical to load, meshes above 40,000 faces are simplified
and albedo textures are capped at 1024 pixels to keep Genesis offscreen texture
allocation within practical GPU limits; this does not alter robot,
manipulation-object, or collision geometry. Background scenery is visual-only.
The source NuRec/Gaussian-splat room is still not converted, and exact lighting,
parallax, occlusion, and non-diffuse PBR maps are not
reproduced. A neutral enclosure remains the fallback when no source HDR or
usable USD mesh is available.
Rigid objects retain diffuse color textures in MJCF and URDF, including the
Newton path; non-diffuse PBR maps are omitted and object collisions are coarse
bounds. A successful scene launch is therefore not a visual or physical parity
result. Source splat rooms, some articulated assets and remaining action
checkers need dedicated conversion and cross-backend validation before
claiming scene-level parity. A reset-time head-camera preview is not directly
comparable to a recorded policy video if that video includes policy-driven
head or arm motion.

The full asset snapshot is about 29.4 GB. The resumable downloader stores it
under `datasets/icl/sources/geniesim-assets`, verifies the official pinned
manifest and can prefer the public mirror; only simulation assets are needed.
The source robot USD (`robot/G2_omnipicker/robot_fix.usda`), scene backgrounds
and all referenced object USDs remain required for source-view rendering. G1
camera poses and head RGB pinhole intrinsics are available to RoboVerse; the
source Gaussian-splat room is still not reproduced. The
source HDR dome and textured static USD meshes are used as portable visual
approximations. The static G2 reset preview above is not a visual match of the
recorded policy video.

For both G1 and G2, the default observation is head RGB plus left and right
head-mounted fisheye views. This unified profile differs from the source
benchmark's head-plus-wrist recording protocol. `camera_profile="benchmark"`
explicitly requests the source wrist-camera profile for G2. The G2 source
OpenCV model is used for its side cameras and temporarily borrowed for G1.
Neither establishes calibration for the user's physical robot.

In the inspected G2 `pick_block_color` reset, both the original G2 task and a
cross-robot scene render a head RGB image largely occluded by the robot's own
grippers. The reset joint values and tested joint-local frames agree with the
source definitions; the source rendering behavior and other reset poses still
need investigation. Do not treat this view as validated training data merely
because a nonblank image was rendered.

For video inspection and RoboColiseum-compatible G2 observations, preserve the
protocol's head image at 640×400 and hand images at 1280×1056; there is no
single resolution shared by all manipulation benchmarks. The G1 source head
RGB metadata gives 640×400 at 640 pixels wide; with borrowed G2 optics its two
fisheye views are 640×512. Keep that
calibrated render as the source and use a documented crop/resize only at the
model-input boundary. A 224, 256 or 112 square image is a policy/encoder input
choice, not the simulator camera's native capture resolution. The two supplied
reference videos are 640×400. See the [official RoboColiseum inference
protocol](https://github.com/AgibotTech/genie_sim/blob/main/source/geniesim_benchmark/skills/robocoliseum/challenge-inference-protocol/SKILL.md)
and [Genie Sim benchmark usage](https://github.com/AgibotTech/genie_sim/blob/main/source/geniesim_benchmark/USAGE.md).

The migration contract is one RoboVerse task family per source behavior, with
configuration-level entries retaining their source robot, scene, instruction
task and robustness transform. A valid parity report must compare the same
source scene and robot, task completion predicates, camera placement and
resolution, initial robot/object state and time base. Physics traces and
success outcomes must be measured independently on MuJoCo, Genesis and Newton;
a registry entry or successful scene launch is not evidence of alignment.
The shared runner registers all imported G1 and G2 source configurations.
Historical reset/render/one-step smoke results cover the earlier metadata-only
snapshot; they do not validate the current source-USD scene and named-reset
snapshot. The current implementation still requires independent three-backend
testing, and none of these smoke checks establishes task success, visual parity
or matching physics traces. The G2 static preview still lacks the source room
reconstruction and material fidelity. The fruit grasp probe currently fails by
pushing the target off the table, so it is not eligible for successful
demonstration collection.

### G1 fruit-sorting migration

`icl.fruit_sorting` is the first RoboVerse task adapted from Genie Sim's
`sort_fruit/1` scene and its Inside/Onfloor conditions. It is a development
prototype, not the complete Genie Sim benchmark or an official benchmark score.
It loads the G1 Omnipicker, two boxes, two peaches, a lemon and a table. The
registered task uses MetaSim scenarios and can be run on MuJoCo, Genesis and
Newton. It checks the selected peach against the source box's world-axis-aligned
bounding box for two consecutive control steps and reports an object on the
floor separately as a failure. The control step is 1/30 s; equivalence of the
source check's timing has not been established.

After the asset snapshot above is available, prepare the source meshes and scene
from a repository-root PowerShell session. Each command refuses to overwrite an
existing output; remove neither original data nor a previous preparation run.
All inputs must match the pinned source manifest.

```powershell
$GenieAssets = "datasets/icl/sources/geniesim-assets"
$GenieCode = "../icl/genie_sim-main/genie_sim-main/source/geniesim_benchmark/src/geniesim_benchmark"
$ObjectRoot = "roboverse_data/icl/objects"
foreach ($Kind in @("peach", "lemon", "box_008", "box_010", "table")) {
    python tools/icl/prepare_geniesim_apple.py --source $GenieAssets --kind $Kind --output "$ObjectRoot/geniesim_${Kind}_aligned"
}
$G1Closed = "roboverse_data/icl/robots/geniesim_g1_omnipicker_closed_v6"
python tools/icl/prepare_geniesim_g1.py --source $GenieAssets --variant G1_omnipicker --urdf "$GenieCode/robot/isaac_sim/robot_urdf/G1_omnipicker.urdf" --output $G1Closed
python tools/icl/prepare_fruit_scene.py --scene-info "$GenieCode/benchmark/config/llm_task/sort_fruit/1/scene_info.json" --assets $ObjectRoot --output roboverse_data/icl/scenes/fruit_sorting_1
python tools/icl/probe_fruit_task.py --sim mujoco --output examples/output/qa_runs/icl_fruit_probe/mujoco --image-size 640 --steps 60 --delta-x 0.01
python tools/icl/probe_fruit_task.py --sim genesis --output examples/output/qa_runs/icl_fruit_probe/genesis --image-size 640 --steps 60 --delta-x 0.01
python tools/icl/probe_fruit_task.py --sim newton --output examples/output/qa_runs/icl_fruit_probe/newton --image-size 640 --steps 60 --delta-x 0.01
```

The visual probe defaults to a 640-pixel camera width; the `FruitSortingTask`
training/API default stays at 256 pixels for model observations. The probe
saves an overview, head RGB before and after turning the head, and a JSON
report. It moves the right TCP and commands the coupled gripper; it is not
a demonstration of picking or sorting fruit. The task exposes 20 named body,
head, arm and gripper-leader position targets consistently across backends. A
7D world-frame Cartesian delta controller uses MuJoCo kinematics to create those
targets; physical tracking takes multiple control steps and is not identical
between simulators. Gripper gains are provisional and are not a real G1
calibration. The source robot's loop and mimic constraints are required; an
asset without them is rejected.

The scene follows the source `scene_info.json` poses, including the table height;
the upstream `scene.usda` gives the table a different height and contains an
extra lemon absent from that metadata. The laboratory background is not yet
included. Mesh textures use source color maps; normal and PBR materials are not
reproduced. The table, open boxes and fruit use approximate collision geometry,
and fruit mass/inertia are approximations. In observed resets, the source fruit
starts above the table, falls and rolls different distances across backends.
Therefore these scenes are not dynamically equivalent to the Isaac source and
are not yet suitable for claiming reproducible cross-engine benchmark scores.
The mounted head RGB and two fisheye views use source robot camera transforms;
G1 lens coefficients are borrowed from G2 rather than physically calibrated.
A physical expert that reliably sorts fruit,
demonstration export and the rest of the Genie Sim tasks remain to be validated
or implemented.
One MuJoCo exploratory approach-and-close rollout pushed the peach off the
table; it is a failed physical attempt, not a successful demonstration.

### SynthICL object assets and DTD textures

The [SynthICL object bundle](https://huggingface.co/datasets/sniper-qian/SynthICL-PseudoDemo-Objects)
provides meshes, materials, catalogs and GraspNet annotations; it does not
contain robot trajectories. Store it separately from generated episodes and
retain the source layout. On Windows, after setting the D-drive caches above:

```powershell
$Source = "datasets/icl/sources/synthicl-pseudo-objects"
hf download sniper-qian/SynthICL-PseudoDemo-Objects --repo-type dataset --revision 5f3124eff350ea0e1014253c700842aa94345607 --local-dir $Source
python tools/icl/prepare_synthicl_assets.py --source $Source --write-manifest
python tools/icl/prepare_synthicl_assets.py --source $Source --verify-only
python tools/icl/download_dtd.py --output datasets/icl/sources/dtd --workers 6
python tools/icl/prepare_synthicl_assets.py --source $Source --output roboverse_data/icl/objects --objects tea_box_base2
```

The manifest is retrieved from the official Hugging Face revision and contains
the Git or LFS checksum for every source file. The preparation command verifies
the inputs before making an independent `synthicl_tea_box_base2` asset. For a
public mirror, set `HF_ENDPOINT` only for the download, retain the same revision,
avoid sending credentials to the mirror, and run `--verify-only` against the
official manifest. Existing manifests and generated assets are never overwritten.

The DTD downloader checks the official archive's size and published MD5 before
extracting all 5,640 images. To prepare a DTD-textured variant:

```powershell
$Dtd = "datasets/icl/sources/dtd"
python tools/icl/prepare_synthicl_assets.py --source $Source --output roboverse_data/icl/objects --objects tea_box_base2 --dtd-root $Dtd --texture "$Dtd/extracted/dtd/images/banded/banded_0002.jpg"
```

The variant's name includes the texture hash; its metadata records the source
image and the original mesh and grasp-file hashes. It keeps the original object
scale. Its collision mesh is an **approximate convex hull**, so a cup or bowl
interior is filled for collision purposes; do not use it as a physical container.
The source bundle card does not specify a single license covering every original
asset; check its source terms before redistribution. DTD is available for
research use under the terms on its [official page](https://www.robots.ox.ac.uk/~vgg/data/dtd/).

`roboverse_learn.il.icl.waypoints` converts a GraspNet annotation into
source-local pregrasp/grasp transforms and resolves a shared source/target-relative
pick-and-place plan in multiple scene poses. The asset wrapper leaves the mesh
and its grasp annotations in their original common frame; if a caller rotates a
mesh it must rotate its grasp poses consistently. These poses are **kinematic
targets**: reachability, collisions and physical grasp success have not been
established by the conversion. The current physical expert does not yet use
these GraspNet targets.

### Static SynthICL environment profiles

RoboVerse includes static scene configurations for the four upstream skill names:
`pick_and_place_gn`, `push_gn`, `pour_gn`, and `random_gn`. They reuse the four
locally prepared SynthICL object wrappers, an explicit tabletop, and the
existing MetaSim `ScenarioCfg`. G1 is the default robot and provides head RGB,
two G2-optics-borrowed fisheye views, and a separate inspection overview;
`--robot g2` selects the prepared G2 model with its matching three hardware head
cameras, and `--robot franka` selects the original data-collection robot
baseline. `--robot all` renders both G1 and G2. These profiles describe initial
environments only. In particular, `pour_gn` shows an empty cup and bowl; it
does not simulate liquid or implement pouring.

The SynthICL cup's `source_base_orientation.z_negative` provenance flag
reproduces the source loader's +1.57 rad Z yaw. Selecting only
`--camera-names overview` skips loading the robot-mounted camera calibration;
the default view set includes head RGB and both fisheye views.

From the RoboVerse repository root, render all four scenes on each selected
backend:

```powershell
python tools/icl/probe_icl_static_scenes.py `
  --sim mujoco --scene all --robot all --image-size 640 `
  --output examples/output/qa_runs/icl_phase1_static_scenes
python tools/icl/probe_icl_static_scenes.py `
  --sim genesis --scene all --robot all --image-size 640 `
  --output examples/output/qa_runs/icl_phase1_static_scenes
python tools/icl/probe_icl_static_scenes.py `
  --sim newton --scene all --robot all --image-size 640 `
  --output examples/output/qa_runs/icl_phase1_static_scenes
```

The probe applies the configured reset state and renders without stepping
physics. It writes a PNG and a one-second static MP4 per selected camera, plus
a report for each robot/scene combination with the backend, scene, camera dimensions, source provenance hashes,
Git HEAD and dirty-worktree state. It does not test skill execution or
manipulation success. Source object and texture redistribution terms remain
unconfirmed, so keep these outputs and assets local.

New smoke captures belong under `examples/output/qa_runs/` until their asset
revision, RoboVerse revision, backend versions, output hashes, and visual review
have been checked. The older G1 apple-scene captures at
`examples/output/icl_g1_apple_scene_qa/wall_removed_front_640/` use the v10 robot
asset; the current default is v11, so those videos are historical diagnostics,
not evidence for the current model. The clean `ICL_Benchmark` release area lists
only artifacts that have passed the release checks. The four SynthICL object
wrappers and static scene builder are present, but their complete G1/G2 by
backend render matrix has not been run yet.

### Kinematic pseudo-demonstration prototype

The first migration units cover the documented `pick_and_place_gn` and
`push_gn` waypoint contracts. Pick-and-place grasps source object 1 with a
GraspNet pose, then places it on top of destination object 2. Push uses one
object and the source geometric contact offset, GraspNet approach, and postpush
displacement. Both preserve the original object scale and reuse one sampled
grasp or waypoint plan across two fixed initial poses.

The source task generator is pinned to
[`sNiper-Qian/synthicl-pseudo-demo` at `131d22f9a4a8f066aa2fd7bee8d88118dcc0766a`](https://github.com/sNiper-Qian/synthicl-pseudo-demo/tree/131d22f9a4a8f066aa2fd7bee8d88118dcc0766a).
That repository has no `LICENSE` file at the pinned revision. RoboVerse does
not copy its sampler code or object assets; this task contract is an independent
implementation based on the public README and configuration behavior. The
asset bundle has its own source and license terms, recorded separately from the
generator SHA. Static environment profiles exist for `pour_gn` and `random_gn`,
but their source sampling, waypoint generation, physical execution and
trajectory-pair collection are not implemented. The broader task, scene,
appearance and object distributions also remain outside the current prototype.

```powershell
python tools/icl/collect_pseudo.py --sim mujoco --seed 0 --image-size 256 --output datasets/icl/generated/pseudo_demo/episode_0.h5
python tools/icl/collect_pseudo.py --sim mujoco --task push_gn --seed 0 --image-size 256 --output datasets/icl/generated/pseudo_demo/push_episode_0.h5
```

Use `genesis` or `newton` to select the other supported backends. Pick-and-place
records two RGB trajectories, TCP states and actions, source and target object
poses, six waypoint indices, camera configuration and source hashes. `push_gn`
records one object pose per frame and six indices for start plus the five source
waypoint entries. Omitting `--grasp-index` makes the seeded sampler select a
GraspNet row; specifying it fixes that choice. Push preserves the source command
order: approach while opening, close without attachment, approach the grasp
pose, attach the object to the hand, then move to the postpush pose. This mirrors
the upstream kinematic attachment and transport behavior; it does not claim
contact-driven sliding. The prototype uses RoboVerse's existing ground plane
rather than reconstructing the upstream scene geometry.

Push preparation requires `bbox_size`, `grasp_angle`, and
`base_orientation.z_negative` in the source object entry. The asset preparer
stores those as `bbox_half_extents_m`, `grasp_angle_rad`, and
`source_base_orientation` in the generated provenance manifest. If an older
prepared asset lacks these fields, prepare it again into a new directory and
pass that directory with `--asset-root`.

Both prototypes set kinematic states directly: they do not simulate object
contacts, check collisions, or establish physical grasp or push success. Scope
is one selected source object, one shared GraspNet row or waypoint plan and two
fixed initial poses. RGB is unsegmented, and `images_wrist` denotes an external
overhead camera, not a mounted camera. The current physical-training wrapper
requires physical provenance and does not accept these prototype files. The
upstream loader can read paired `trajectory_0`/`trajectory_1` data with
`num_traj_per_task=2` and `num_prompt_traj=1`; this format check does not
establish reproduction or training readiness.

Obtain `assets.tar.zst` from the pinned revision
[`2d7a339b27aff14a0780db4bd4d5e3a68c6358bc`](https://huggingface.co/datasets/imitator-game/IG-10K-Assets/tree/2d7a339b27aff14a0780db4bd4d5e3a68c6358bc).
Consult the source asset owners' licenses before redistribution. From the
RoboVerse repository root, install the ICL extra together with the intended
simulator extras in one dependency resolution (for example,
`uv pip install -e ".[icl,mujoco,genesis,newton]"`), then run:

```powershell
python tools/icl/prepare_assets.py --archive roboverse_data/icl/ig10k-source/assets.tar.zst --output roboverse_data/icl/objects --objects 008_pudding_box 061_foam_brick
$Dataset = "datasets/icl/generated/my_run"
python tools/icl/collect.py collect --sim mujoco --goal left --split train --seed 0 --output $Dataset
python tools/icl/collect.py collect --sim mujoco --goal left --split train --seed 1 --output $Dataset
python tools/icl/collect.py pair --reference "$Dataset/raw/mujoco_008_pudding_box_left_0.hdf5" --query "$Dataset/raw/mujoco_008_pudding_box_left_1.hdf5" --output "$Dataset/train/pair_0.h5"
python tools/icl/collect.py check-splits --library $Dataset
```

Add `--task pick_place --object-id 061_foam_brick` to collect grasp-and-place
episodes. The default task is `push`. The current scripted gripper approach is
not suitable for every shape; failures are retained and not exported as successful
demonstrations. The original SynthICL generator renders pseudo-demonstrations
without physics, whereas these recordings are physical closed-loop rollouts.

The collector defaults to 256 x 256 RGB images. `--image-size 512` renders at
512 x 512 for inspection or later downsampling; the upstream training script
resizes its inputs to 256 x 256. Increasing the image dimensions does not fix
an object that is too small in the camera's field of view. The front camera is
framed to include the robot and both targets; side and overhead cameras retain
closer task views. Rendering at 512 increases storage and memory requirements.

Repeat collection with other seeds, goals and simulators. Reserve different asset
hashes for `train`, `validation` and `test` before collection; do not put held-out
episodes under the training pair directory. Each pair needs two independent
successful trajectories of the same task and goal. Pair files use relative HDF5
links to the raw recordings; moving the dataset requires preserving their relative
paths. Failed episodes produce a physical `.npz` and a result `.json`, but no
training trajectory. The scripted expert is not guaranteed to succeed for every
asset or random seed; examine the recorded failure rather than relabeling it.

For a reproducible batch, create a JSON array such as:

```json
[
  {"sim": "mujoco", "goal": "left", "split": "train", "object_id": "008_pudding_box", "seed": 0},
  {"sim": "genesis", "goal": "left", "split": "train", "object_id": "008_pudding_box", "seed": 1},
  {"sim": "newton", "goal": "right", "split": "test", "object_id": "013_apple", "seed": 2}
]
```

After preparing both objects, run
`python tools/icl/collect.py batch --plan PLAN.json --output $Dataset`. The plan
is checked for duplicate origins and asset overlap with existing split data before
starting simulators. Jobs use separate processes and preserve per-job logs; a
second invocation reports existing successes or failures without overwriting
them. Specify `"task": "pick_place"` on a job to select the second task;
omitting it selects pushing. A failed job makes the batch command return a
nonzero exit status. `--position-jitter` sets the independent per-axis initial
object-position sampling range in meters (default `0.003`, maximum `0.03`);
its value and the sampled position are retained in each episode's metadata.
Larger ranges are not guaranteed to be solvable by the scripted expert.
The batch report distinguishes task failures, incomplete recordings, and
execution errors such as backend startup failures. Rerunning preserves existing
records and retries an execution error that produced no recording; each retry
receives a new log file. Each completed batch also saves its exact plan, image
size, jitter and outcome under the dataset's `reports/` directory.

## Relationship to the papers and upstream generator

[ICRT](https://arxiv.org/abs/2408.15980) emphasizes scenes in which multiple
tasks are possible, so the context is needed to select an action. Its prompts
include robot images, proprioception and actions from teleoperation. Human-only
video is a different input domain, not a robot action trajectory.
[Instant Policy](https://arxiv.org/abs/2411.12633) uses point clouds and
procedurally generated, semantically consistent pseudo-demonstrations.
[SynthICL](https://arxiv.org/abs/2606.08154) uses segmented multi-view RGB,
object-relative paired trajectories and an auxiliary subgoal image objective.
Its generator samples relative waypoints once for a pair, then changes object
poses while retaining the task relationship; grasped objects are attached
kinematically. The default generator also varies object and gripper poses,
appearance, lighting and task type and can use precomputed GraspNet grasps.

The physical RoboVerse collector is a **prototype**, not a reproduction of that
generator. It records two independent successful task executions with
matching task, goal and phase schema, rather than sharing sampled object-relative
waypoints. It has push and pick-and-place only, uses unsegmented RGB and an
external overhead view in place of the wrist camera, and does not yet implement
the upstream lighting, texture or grasp-pose sampling. It does not currently
generate an apple-in-basket task. The configured world-frame translation and
left-multiplied rotation deltas use the same convention as the upstream
generator; the physical controller and upstream kinematic gripper have
different tracking behavior. Compatible HDF5 keys and successful loading alone
do not establish reproduction of the training distribution or reported results.
Newly collected trajectories explicitly record `demonstration_kind`,
`image_processing` and each camera's pose, field of view and resolution in their
metadata so that physical, pseudo and differently rendered data can be audited.

The IG-10K task mapping associates human `human_H7` and robot `robot_H7_L0`
with `L0_TwoRobotPickAppleBasket-v1`. This is a task-level correspondence, not
evidence that particular recordings form synchronized pairs. Human H7 contains
video and tracked hand observations but no native robot action feature. Its
camera names alone do not establish an egocentric viewpoint. Keep these source
episodes and their metadata separate from generated robot-action pairs until a
validated cross-domain observation adapter and equivalent RoboVerse task exist.
Task identity, source recording, asset, appearance and view splits should be
recorded before collection; evaluate same-scene context swaps and held-out
combinations so scene appearance alone cannot identify the goal.

To reproduce the upstream data-generation approach on RoboVerse, the remaining
work is to:

1. Extend the fixed two-scene kinematic prototype to sample task specifications,
   object-relative waypoints and grasp/release events with controlled randomness.
   Check relative transforms and attached-object motion throughout the pair.
2. Implement and verify the actual front, side and mounted-wrist views, plus
   foreground segmentation, on all three selected backends. Record camera
   calibration, segmentation semantics and randomization seeds.
3. Add the missing task and appearance distributions, valid grasp poses,
   collision/reachability checks where physical feasibility is claimed, and
   held-out task/object/scene splits. Keep kinematic pseudo-demonstrations
   explicitly distinct from physical rollouts.
4. Verify every recorded waypoint index, next-subgoal assignment and
   observation/action time alignment against the SynthICL loader. Compare
   generated pairs and model inputs with upstream examples, then test a trained
   policy in closed loop; the current loader smoke test is only an interface
   check. The upstream default samples pick-and-place, push, pour and random
   waypoint tasks; the current two physical tasks do not cover that mixture.

## Inspecting existing demonstrations

The public [IG-10K dataset](https://huggingface.co/datasets/imitator-game/IG-10K-Dataset)
can be downloaded per task. For example, a pinned task subset can be retrieved
without fetching the full dataset:

```powershell
hf download imitator-game/IG-10K-Dataset --repo-type dataset --revision 9a57b27edcb54823d8da7d2b77efa61c092ffa95 --include "imitator_sim_v1_zed2i/L0_TwoRobotPickAppleBasket-v1/*" --local-dir datasets/icl/sources/ig10k-dataset
python tools/icl/inspect_dataset.py --task-dir datasets/icl/sources/ig10k-dataset/imitator_sim_v1_zed2i/L0_TwoRobotPickAppleBasket-v1 --episode 0 --source-repo https://huggingface.co/datasets/imitator-game/IG-10K-Dataset --source-domain simulation --source-revision 9a57b27edcb54823d8da7d2b77efa61c092ffa95 --output INSPECTION
```

The inspector checks episode boundaries, native actions, timestamps and RGB video
frames, then saves a provenance report and first/last previews. This task has
16-dimensional two-arm joint actions and one RGB view; it is not directly
compatible with this prototype's 7-dimensional single-arm TCP actions and three
views. Keep it as source data until an explicit, validated task and control
adapter exists. Its human and robot examples must not be assumed to be paired
without checking the source task mapping.

For a human source episode that has no robot-action feature, use
`--source-domain human --video-key observation.images.cam1` and point
`--task-dir` to `imitator_human_v1/human_H7`. The inspector checks the source
timestamps and available hand observations, and records the missing action as
`null`; it does not infer a robot command. Download the selected RGB video,
episode metadata and parquet rows first. Other camera and depth files are not
needed for this inspection.

## Optional local visual encoder

The `timm/vit_small_patch16_dinov3.lvd1689m` encoder may be stored under
`checkpoints/icl/backbones/vit_small_patch16_dinov3.lvd1689m/`. With the cache
paths above configured, download the pinned files:

```powershell
hf download timm/vit_small_patch16_dinov3.lvd1689m --revision 3bf4720a82ec2066db88137180ff1f83a675cef0 --include config.json model.safetensors README.md LICENSE.md --local-dir checkpoints/icl/backbones/vit_small_patch16_dinov3.lvd1689m
```

Pass the local file explicitly to avoid an implicit Hub download:

```python
import timm

weights = "checkpoints/icl/backbones/vit_small_patch16_dinov3.lvd1689m/model.safetensors"
encoder = timm.create_model(
    "vit_small_patch16_dinov3.lvd1689m",
    pretrained=True,
    features_only=True,
    pretrained_cfg_overlay={"file": weights},
).eval()
```

This is ViT-S, whereas the current upstream SynthICL trainer constructs Meta's
ViT-S+ through its local `torch.hub` backbone. Matching 384-channel feature shapes
does not make their checkpoints interchangeable. The optional encoder can be
used for local feature extraction; training integration and compatibility must
be implemented and tested before it replaces the upstream backbone.

## Training and evaluation

### Native camera images and model input size

Raw `.hdf5` trajectories store each RGB stream at its own native height and
width; all streams still have the same number of time steps. The saved
`camera_native_hw` metadata records the actual array dimensions. For the G1/G2
head-camera profile, `HEAD_CAMERA_ROLES` maps SynthICL's historical dataset
keys as follows: `images_front` → `head_rgb`, `images_wrist` →
`head_fisheye_left`, and `images_side` → `head_fisheye_right`. The key named
`images_wrist` is a compatibility name in this profile; it does not indicate
a wrist-mounted camera.

The original SynthICL `PseudoDemoDataset` independently resizes each camera
to `shared_config.image_size` using PyTorch bilinear interpolation with
`align_corners=False` and `antialias=True`. The local training configuration
sets 256×256, so directly loading a native 640×400 image **stretches** it to a
square. Raw images of different sizes therefore do not need to be changed to
be read by this loader, but its default stretch changes the aspect ratio.

If aspect-preserving inputs are preferred, create new derived trajectories;
the native files are retained unchanged. Both trajectories in a pair, and all
pairs in a training set, must use the same image policy:

```powershell
python tools/icl/collect.py preprocess --source DATASET/native/a.hdf5 `
  --output DATASET/prepared/a.hdf5 --mode letterbox --height 256 --width 256
python tools/icl/collect.py preprocess --source DATASET/native/b.hdf5 `
  --output DATASET/prepared/b.hdf5 --mode letterbox --height 256 --width 256
python tools/icl/collect.py pair --reference DATASET/prepared/a.hdf5 `
  --query DATASET/prepared/b.hdf5 --output DATASET/train/episode_0.h5
```

`letterbox` preserves the full image with black padding; `center_crop` fills
the target and removes equal portions of the long dimension; `stretch` resizes
each axis independently. Dimensions are rounded to integer pixels and the
per-camera metadata records the actual resize, padding, crop and 3×3 affine
mapping of source pixel centers to output pixel centers. The derived RGB is
rounded to `uint8`; the upstream loader's own float stretch has no intermediate
`uint8` rounding, so those paths are not bit-identical. The derived metadata
stores a relative path and SHA-256 of the native trajectory. Keep both files
inside `--library` for training validation: the original physical `.npz`
record remains beside the native trajectory. A changed native file is rejected.

Optional per-camera `camera_calibrations` metadata is keyed by the image keys.
Preserve `projection_model`, distortion coefficients and the intrinsic matrix
when known. If an `intrinsic` matrix is supplied, `intrinsic_provenance` must
be `renderer_reported`, `source_nominal`, or `physical_calibration`; the
output matrix is the pixel affine multiplied by the supplied matrix and keeps
that provenance. A nominal source matrix does **not** establish that a backend
actually rendered using it. When no matrix is supplied, no rendered intrinsic
is invented. Resize, pad and crop do not undistort or rectify a fisheye image;
the original projection type and distortion parameters remain recorded.
For G1 cameras using the temporarily borrowed G2 calibration, set
`calibration_source: borrowed_g2` and use the appropriate nominal provenance;
this is not a physical G1 calibration. The recorded output intrinsic represents
the supplied input projection composed with the pixel affine, not an independent
second affine to apply on top of the recorded projection.

The writer optionally accepts one static Boolean `valid_masks` array per
image key, shaped to the native image, together with `camera_valid_masks`
metadata stating its source. They are stored separately as `valid_mask_front`,
`valid_mask_wrist` and `valid_mask_side`; `true` means the source pixel is
valid. The existing Franka expert collector captures `CameraState.valid_mask`
when available and checks that it stays constant throughout the episode; a
camera without a mask remains supported. This does not imply that a G1/G2
physical-demonstration collector already exists. The preprocessing step creates corresponding target masks: pixels are
valid only when their bilinear interpolation support is valid, and padding is
invalid. If a source mask was not supplied, its source is explicitly marked
`all_source_pixels_assumed_valid`; this assumption does not establish physical
sensor validity. The original SynthICL loader ignores these additional datasets:
it still receives RGB including black invalid regions and does not use the mask
as an attention or loss mask. Mask-aware training requires a separate model and
loader change; storing the masks alone does not provide it.

These storage and input transforms are independent of demonstration validity.
The existing SynthICL HDF5 action schema is a seven-component TCP action; it
does not by itself export or train on the G1/G2 benchmark's full joint or
bimanual action spaces. A matching collector, action adapter and successful
physical demonstrations are still required.

`tools/icl/train.py` checks split and physical-record provenance, disallows reuse
of an original episode across pairs (the upstream loader randomly splits pairs),
and calls the original [SynthICL trainer](https://github.com/sNiper-Qian/synthicl).
At least two disjoint pairs are necessary, but far more diverse examples are
needed for a meaningful experiment. DINOv3's official ViT-S/16+ weights require
access through [Meta's DINOv3 distribution](https://github.com/facebookresearch/dinov3).

```powershell
python tools/icl/train.py --synthicl-dir ../icl/synthicl --dataset-dir DATASET/train --library DATASET --dino-repo DINO_REPO --dino-weights DINO_WEIGHTS --checkpoint-dir CHECKPOINTS --epochs 10
python tools/icl/evaluate.py --synthicl-dir ../icl/synthicl --checkpoint CHECKPOINTS/RUN/model_final.pth --dino-repo DINO_REPO --dino-weights DINO_WEIGHTS --library DATASET --reference DATASET/raw/HELD_OUT_DEMO.hdf5 --sim mujoco --seed 200 --output RESULT.json
```

The evaluation entry point runs a closed-loop episode with a frozen model and a
held-out reference. For a context-swap experiment, run the same checkpoint,
backend, object and query seed with distinct left and right test references and
report both successes and failures. A successful scripted rollout or a data-loader
smoke test is not a trained-policy result. The current evaluator and end-to-end
training still require verification with the gated weights and a sufficiently
large, split-safe dataset.

## Tests

The lightweight contracts and physical collection smoke tests follow the
repository's existing pytest conventions. The latter require the prepared
objects and a working renderer; Genesis and Newton are explicitly selected:

```powershell
python -m pytest tests/test_icl.py -q
$env:ROBOVERSE_ICL_TEST_BACKENDS = "genesis,newton"
python -m pytest tests/test_icl_integration.py -q -ra
```

To run the Genie Sim conversion and G1 tabletop preview checks, also set:

```powershell
$env:ROBOVERSE_GENIESIM_SOURCE = (Resolve-Path $GenieAssets).Path
$env:ROBOVERSE_GENIESIM_G1_ASSET = (Resolve-Path $G1Asset).Path
$env:ROBOVERSE_GENIESIM_APPLE_ASSET = (Resolve-Path $Apple).Path
python -m pytest tests/test_icl_geniesim.py -q -ra
$env:ROBOVERSE_ICL_FRUIT_SCENE = (Resolve-Path roboverse_data/icl/scenes/fruit_sorting_1).Path
$env:ROBOVERSE_GENIESIM_G1_CLOSED_ASSET = (Resolve-Path $G1Closed).Path
python -m pytest tests/test_icl_fruit_sorting.py -q -ra
python -m pytest tests/test_icl_integration.py -k geniesim_fruit_scene -q -ra
```

The checker runtime's pure graph and predicate contracts are tested without
starting a simulator:

```powershell
python -m pytest tests/test_icl_geniesim_checker.py -q
```

To smoke-test instance 0/problem 0 from all eight G1 source runs on all three
backends, enable Genesis and Newton and run:

```powershell
$env:ROBOVERSE_ICL_TEST_BACKENDS = "genesis,newton"
python -m pytest tests/test_icl_geniesim.py::test_geniesim_registered_task_runs_checker_on_shared_backend_states -q -ra
```

To run the full G2 catalog smoke matrix (78 run configurations, three
backends), use the prepared G2 v11 model and imported assets:

```powershell
$env:ROBOVERSE_ICL_TEST_BACKENDS = "genesis,newton"
python -m pytest tests/test_icl_geniesim.py -k g2_catalog_run_builds -q -ra
```

The earlier G2 matrix reported 216 passed and 18 skips for the legacy import;
the six skipped configurations were later smoke-checked with an unverified
compatibility alias. Those historical results do not cover the v2 USD scene
snapshots or the task-specific reset changes. Genesis and Newton must be
installed, and Newton requires a CUDA GPU.

Newton requires a CUDA GPU.

Newton requires a CUDA GPU. A missing optional backend, asset, or GPU is reported
as a skip; a simulator that is present but fails the task remains a test failure.

### Unique-scene robot loading and recording matrix

This probe selects one deterministic source run and instance for each unique
scene and independently loads G1 and G2 on each selected backend. By default it
records a short head-pan motion and all three head cameras. It checks loading, motion,
video decoding, and frozen source, robot and code hashes; it does **not**
establish task success, validate all source configuration variants, or prove
that robot motion is equivalent across engines. The currently configured head
side cameras use G2 source fisheye optics, including when attached to G1; this
does not constitute physical G1 camera calibration.

Start with one scene and inspect its three-camera videos and per-job report:

```powershell
python tools/icl/record_geniesim_benchmark_video.py --all-scenes --scene-count 1 --image-size 128 --frames 12 --matrix-output examples/output/qa_runs/geniesim_scene_matrix_smoke
```

Use `--camera-names head_rgb` to record only the head RGB camera, or a
comma-separated selection such as `--camera-names head_rgb,head_fisheye_left`.
The default hardware profile offers `head_rgb`, `head_fisheye_left`, and
`head_fisheye_right`; duplicate or unavailable names are errors. Single G2
recordings can select `--camera-profile benchmark` and its `head_rgb`,
`hand_left`, and `hand_right` cameras. Matrix plans and reports retain the
selection, and a resumed run must use the same selection.

The Python task also accepts `camera_names=("head_rgb",)` and
`fisheye_projection="pinhole"` for an explicit pinhole fallback. Until G1 has
measured lens calibration, all three G1 views use G2's corresponding lens
intrinsics and distortion parameters while retaining G1's camera mounting
transforms. This is a temporary cross-robot approximation; fisheye camera
reports label the borrowed calibration `borrowed_g2`. The source does not specify a usable
maximum angle for the OpenCV polynomial, so the implementation conservatively
limits it to its first monotonic domain; pixels outside that domain are black
and exposed through `CameraState.valid_mask`. Head RGB OpenCV pinhole distortion
is applied to returned RGB observations, and its out-of-image pixels are marked
invalid. The capture width defaults to 640 while preserving each source lens's
aspect ratio: head RGB is 640x400 and the two fisheyes are 640x512. Resize or
letterbox only in the model input transform; squaring images at capture time
would change their calibrated geometry. Newton traces native fisheye rays;
MuJoCo and Genesis sample six perspective renders. MuJoCo and Genesis currently
support fisheye RGB only, and Genesis supports one environment at a time.
Camera masks survive RoboVerse ICL preprocessing, but the original SynthICL
loader currently ignores them. The G1/G2 demonstration collection path has not
yet been validated end to end.

The three backends use the same camera poses and calibrated lens mapping, but
their renderers still differ in exposure, shadows, and material response. A
G1 three-camera cross-backend capture is indexed at
`examples/output/qa_runs/20261031_camera_return_validation/README.md`; it is a
camera-output smoke check, not a real-camera calibration or task-success claim.

After the static visual-scene import fix, the three backends were recorded
again at 640-pixel capture width. The diagnosis, scoped Genesis warning
handling, three-backend results and remaining calibration limits are documented
in [GenieSim Genesis and camera QA](icl_geniesim_camera_qa.md). These short
captures verify loading and camera rendering only; the task checker remained
running, and G1 still uses borrowed G2 lens calibration.

Optional `--camera-config path/to/calibration.json` overrides the lens model for
selected views only. The following numbers illustrate the schema; replace them
with measured calibration before using the images for real-robot evaluation:

```json
{
  "format": "roboverse.icl.geniesim_camera_calibration.v1",
  "cameras": {
    "head_fisheye_left": {
      "projection_model": "opencv_fisheye",
      "calibration_size": [1280, 720],
      "optical_center": [640, 360],
      "focal_length_px": [500, 500],
      "coefficients": [0, 0, 0, 0],
      "max_fov_rad": 3.0,
      "calibration_source": "illustrative example only"
    }
  }
}
```

For `ftheta`, omit `focal_length_px` and supply five coefficients in ascending
powers for `radius_pixels = k0 + k1*theta + ... + k4*theta^4`. Coordinates and
focal lengths use pixels at `calibration_size`; `max_fov_rad` is the total
angular field, not its half-angle. Each override can include a nonempty
`calibration_source` label. Configuring an unselected camera, an invalid field,
or an ambiguous lens domain fails rather than silently using another model.
The report preserves the configuration file hash, actual camera parameters,
calibration source, mount pose, and geometric valid-pixel fraction. Resuming a
matrix requires the same calibration file and content.

Run a larger segment with a separate output directory. For example:

```powershell
python tools/icl/record_geniesim_benchmark_video.py --all-scenes --scene-start 0 --scene-count 10 --image-size 256 --frames 30 --matrix-output examples/output/qa_runs/geniesim_scene_matrix_segment_0000
```

Repeat the identical command with `--resume` after interruption; successful jobs
are reused only after provenance and complete video decoding are rechecked.
Failed jobs retain their logs and get a new attempt directory. Select subsequent
segments using `--scene-start` and distinct output directories. Omitting
`--scene-count` selects every remaining scene: the current snapshot has 759
scenes, or 4,554 jobs for two robots and three backends, so plan runtime and
storage accordingly. `matrix_plan.json` freezes inputs and settings;
`matrix_status.json`, reports, logs and videos record the outcome of each job.

## Verified demo publication

`tools/icl/publish_benchmark.py` is the release gate for
`examples/output/ICL_Benchmark/`. It reads the local verification queue in
read-only mode by default and reports eligible/rejected demos without changing
the output directory:

```powershell
python tools/icl/publish_benchmark.py
```

Publication requires the same source revision, task, variant, scene, robot,
backend, seed, and RoboVerse revision to have passing `smoke`, `semantics`,
`demo`, and `parity` queue records. The dependency chain must connect those
records, and parity must depend on matching successful demos from other target
backends. The release gate verifies queue-result hashes, matching semantics
and demo checker records, parity evidence bound to compared demo-result and
trajectory hashes, per-asset SHA-256/HTTPS source/supported SPDX license
records, the pinned source repository and its supported SPDX license
identifier, a decoded multi-frame MP4, a valid PNG thumbnail, and a replay
trajectory with finite actions and aligned state or observation records. Random exploration must pass finite-action,
action-bound, and time-base checks; it can be published as trajectory-valid,
but it is never labeled task-successful.

The publisher checks recorded attestations and artifact identity; it does not
rerun a physics engine or recompute task-specific checkers and parity metrics.
Smoke, semantics, demo, and parity workers must therefore record the actual
commands and evidence. Human review stays separate from automated attestations.

Only after reviewing the dry-run report should Release-QA request publication:

```powershell
python tools/icl/publish_benchmark.py --publish
```

The publisher stages a clean gallery, then switches it with two same-volume
renames; ordinary errors restore the previous gallery, which is retained beside
the new one as a backup. A process interruption between the renames can leave
the target path temporarily absent; restore the hidden `.previous.*` directory
in that case. Failed runs and raw logs stay in `examples/output/qa_runs/`. The generated README reports
the queue's enqueued work-unit counts by source, phase, and status; those counts
are explicitly not upstream task-coverage denominators. Automated publication
does not set a human approval flag; record human review separately in the ICL
Inspector.
