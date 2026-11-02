# The Imitator Game: RoboVerse task adapters

This RoboVerse pack contains native MetaSim migration slices for human tasks **H7** and **H19**.
The first adapter is task **H7**,
`TwoRobotPickAppleBasket`. The scenario declares two independently named RoboVerse Franka/Panda
robots. The left robot performs the source task; the right robot remains available as the second
agent, matching the source demonstration and planner contract. The source `panda_wristcam` model is
not part of RoboVerse's robot catalog, so this adapter uses the canonical Franka model and does not
claim wrist-camera calibration parity.

The procedural workcell reproduces the source 2.0×1.2 m tabletop footprint, four legs, back wall,
and ground plane. Its solids use RoboVerse primitive configs; texture randomization and source
background materials are not yet reproduced.

This slice uses the source's nominal robot and object poses. ManiSkill's per-reset 0.02 joint noise,
small object-position offsets, and L1/L2 scene mutations are not yet ported.

The source low-resolution observation is one 224×224 ZED2i pinhole view. RoboVerse keeps that view
as the default camera and accepts camera configs through `build_scenario`; the original task's
optional wrist sensors can be supplied as mounted MetaSim cameras when the sensor layout is needed.

## H7 source mapping

| Level | Source simulator ID | Robot-data aliases | Simulator-data aliases | Objects | RoboVerse task |
| --- | --- | --- | --- | --- | --- |
| L0 | `TwoRobotPickAppleBasket-v1` | `robot_H7_L0` | `L0_TwoRobotPickAppleBasket-v1` | YCB `013_apple`, `011_banana`, `014_lemon`; RoboTwin `076_breadbasket` model 3 | `imitator_game.h7.pick_apple_basket.l0` |
| L1 | `TwoRobotPickAppleBasket-v1` | `robot_H7_L1` | `L1_TwoRobotPickAppleBasket-v1` | Same base objects with a spatial offset in the source environment | Same L0 RoboVerse task; level-specific reset randomization is not included in this slice |
| L2 | `TwoRobotPickAppleBasket-v1` | `robot_H7_L2` | `L2_TwoRobotPickAppleBasket-v1` | Source changes the apple to YCB `016_pear` | Not migrated in this slice |
| L3 | `TwoRobotPickAppleBasketL3-v1` | `robot_H7_L3`, `robot_H7_L3_1` | `L3_TwoRobotPickAppleBasket-v1` | YCB `013_apple`; RoboTwin `062_plasticbox` model 2 | `imitator_game.h7.pick_apple_basket.l3` |

The level is demonstration-to-query metadata; it is not an input to the policy. L0/L1/L2 share the
upstream base simulator task, while L3 has its own scene, goal geometry, and planner alias. The two
L3 trajectory aliases resolve to the same simulator task ID.

Record runs with the key `(source, upstream_sha, task_id, level_or_variant, scene_id, embodiment,
backend, seed)`, using `The-Imitator-Game`, the pinned reference SHA above, `human_H7`, the requested
level, its `scene_id`, `roboverse_franka_pair`, the selected backend, and the episode seed. Keep the
source `panda_wristcam_pair` embodiment in source provenance to distinguish it from RoboVerse's
current model realization.

## Asset contract

Assets are read from RoboVerse's standard user data root at
`roboverse_data/icl/objects/imitator_game/`. Set `ROBOVERSE_DATA_DIR` to another writable/shared
data root when the checkout-local data directory is not suitable. This migration does not vendor
asset files.

Each YCB directory must contain:

```text
roboverse_data/icl/objects/imitator_game/ycb/013_apple/{object.urdf,object.xml,provenance.json}
roboverse_data/icl/objects/imitator_game/ycb/011_banana/{object.urdf,object.xml,provenance.json}
roboverse_data/icl/objects/imitator_game/ycb/014_lemon/{object.urdf,object.xml,provenance.json}
roboverse_data/icl/objects/imitator_game/ycb/016_pear/{object.urdf,object.xml,provenance.json}  # later L2 adapter
```

The Robotwin variants must be prepared under their exact model IDs:

```text
roboverse_data/icl/objects/imitator_game/robotwin_prepared/076_breadbasket/model_3/{object.urdf,object.xml,provenance.json}
roboverse_data/icl/objects/imitator_game/robotwin_prepared/062_plasticbox/model_2/{object.urdf,object.xml,provenance.json}
```

`provenance.json` must identify `source_object`, include the exact `model_id` for Robotwin, and
provide finite local collision bounds as `bounds_m: [[min_x,min_y,min_z],[max_x,max_y,max_z]]` in
meters. The apple's provenance and both URDF/MJCF wrappers must declare mass `0.5 kg`, matching H7's
task-specific override. The shared prepared `013_apple` currently declares `0.1 kg`, so its mesh is
available but that wrapper is not physically valid for H7 as-is. The URDF/MJCF and every referenced
mesh/material/texture must live beside the wrapper or at its referenced relative path.
`validate_assets(level, asset_root)` checks these contracts before a simulator is started and fails
with all missing wrapper paths; it never replaces a source mesh with a primitive or an unrelated
cached object.

The source YCB archive is pinned by the RoboVerse ICL asset preparer. Download that simulator-asset
archive (not the IG-10K demonstration dataset) and prepare only H7's three required YCB objects:

```powershell
hf download imitator-game/IG-10K-Assets `
  --repo-type dataset `
  --revision 2d7a339b27aff14a0780db4bd4d5e3a68c6358bc `
  --local-dir D:\RoboVerse-asset-staging\imitator-game

python tools/icl/prepare_imitator_game_h7_assets.py `
  --archive D:\RoboVerse-asset-staging\imitator-game\assets.tar.zst
```

The pinned source archive is 27,154,679 bytes (about 25.9 MiB); the three prepared YCB objects occupy
about 3.8 MB in the local cache. The preparation command verifies the archive SHA-256, extracts
only the three YCB model folders, creates RoboVerse URDF/MJCF wrappers and provenance, and applies
H7's 0.5 kg apple mass in a task-local cache. It refuses to overwrite existing data. Set
`$env:ROBOVERSE_DATA_DIR` before the command when the assets should be placed outside the checkout.

### Acquisition, provenance, and licenses

- The Imitator Game code snapshot is MIT licensed. The local source snapshot was fingerprinted as
  `b67adb166eb70411fc341985b4fa6fe48d5bf78428282d099587ae997e905ce5`; the recorded upstream Git
  reference is `3c3404af98727f6d4783537732d0db8e973dfba5`. The local tree is not proven byte-identical
  to that Git commit. Preserve both values in any dataset provenance record.
- The H7 YCB items are available in the Imitator Game asset bundle. ManiSkill identifies its YCB
  assets as CC BY-NC 4.0; follow the non-commercial restriction and retain source/asset checksums in
  each prepared asset's provenance. The shared RoboVerse cache has an IG-10K `013_apple` mesh, but
  its generic task wrapper mass is 0.1 kg instead of H7's 0.5 kg. The task-specific preparer writes
  the corrected instance to the Imitator Game namespace without replacing that shared cache entry.
- The Robotwin object variants are distributed through the official RoboTwin 2.0 assets (`objects.zip`)
  and source object documentation. The pinned RoboTwin 2.0 dataset card and `objects/README.md`
  declare the asset license as MIT. The two converted variants in this migration are pinned to
  revision `39b4a57384f28b6ed0ea1118871ae302e67eae38`; the converter verifies each source-file
  SHA-256, model ID, metadata scale and license provenance before writing wrappers.
- Only the required GLB and metadata files are needed; the full multi-gigabyte archive is not.
  The H7 breadbasket model and H19 tissue box use distinct `robotwin_prepared` paths so converted
  wrappers never overwrite downloaded GLBs. H7's H7-L3 plastic box model 2 is outside this conversion
  batch and still needs its own pinned provenance review. No substitute geometry is used.

Download only the H7 basket and H19 tissue-box visual/collision meshes and metadata from the pinned
dataset revision. The source files occupy about 8.6 MiB for the basket and about 1.7 MiB for the
tissue box (about 10.3 MiB total). Run from the RoboVerse repository root. Resolve the output root
the same way as MetaSim's `LOCAL_DIR`: use the absolute `ROBOVERSE_DATA_DIR` when it is set, or
`<repository root>\roboverse_data` otherwise. This keeps the prepared paths aligned with the task
adapters when using an external data cache:

```powershell
$assetStage = "D:\RoboVerse-asset-staging\robotwin"
if ([string]::IsNullOrWhiteSpace($env:ROBOVERSE_DATA_DIR)) {
  $assetDataRoot = Join-Path (Get-Location).Path "roboverse_data"
} else {
  $assetDataRoot = [Environment]::ExpandEnvironmentVariables($env:ROBOVERSE_DATA_DIR)
  if ($assetDataRoot -eq "~") {
    $assetDataRoot = [Environment]::GetFolderPath("UserProfile")
  } elseif ($assetDataRoot.StartsWith("~/") -or $assetDataRoot.StartsWith("~\")) {
    $userProfile = [Environment]::GetFolderPath("UserProfile")
    $assetDataRoot = Join-Path $userProfile $assetDataRoot.Substring(2)
  }
  $assetDataRoot = [System.IO.Path]::GetFullPath($assetDataRoot)
}
$imitatorAssetRoot = Join-Path $assetDataRoot "icl\objects\imitator_game"
hf download TianxingChen/RoboTwin2.0 `
  --repo-type dataset `
  --revision 39b4a57384f28b6ed0ea1118871ae302e67eae38 `
  objects/076_breadbasket/visual/base3.glb `
  objects/076_breadbasket/collision/base3.glb `
  objects/076_breadbasket/model_data3.json `
  objects/023_tissue-box/visual/base3.glb `
  objects/023_tissue-box/collision/base3.glb `
  objects/023_tissue-box/model_data3.json `
  --local-dir $assetStage

python tools/icl/prepare_robotwin_imitator_assets.py `
  --profile h7_breadbasket `
  --source-directory "$assetStage\objects\076_breadbasket" `
  --output-directory (Join-Path $imitatorAssetRoot "robotwin_prepared\076_breadbasket\model_3")

python tools/icl/prepare_robotwin_imitator_assets.py `
  --profile h19_tissue_box `
  --source-directory "$assetStage\objects\023_tissue-box" `
  --output-directory (Join-Path $imitatorAssetRoot "h19\robotwin_prepared\023_tissue-box\model_3")
```

The converter writes a normalized source provenance record even if the download folder has no
sidecar manifest. It keeps the source GLBs and `model_data3.json` alongside generated wrappers,
preserves scene-node transforms and local coordinates, exports visual base-color textures, and keeps
each source collision decomposition component as a separate mesh. H7 bakes the metadata scale
`(0.1, 0.1, 0.1)` while keeping the task's additional `1.8` scale in the scene config. H19 bakes its
`replace_scale=True` task scale `(0.2, 0.13, 0.06)` and does not also apply metadata scale `(0.06,
0.06, 0.06)`. It refuses to replace an existing output directory.

Generated inertia uses a diagonal box approximation from collision bounds. OBJ/URDF/MJCF conversion
preserves base-color textures and UVs, but does not preserve all glTF metallic/roughness channels.
These wrappers have CPU provenance/structure tests; simulator rendering and dynamics parity still
need backend review.

Official source references: [The Imitator Game](https://github.com/imitator-game/The-Imitator-Game),
[ManiSkill asset documentation](https://maniskill.readthedocs.io/), [RoboTwin 2.0 objects](https://robotwin-platform.github.io/doc/objects/index.html),
and [RoboTwin 2.0 asset package](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/blob/main/objects.zip).

## H19 source mapping and implementation scope

The pinned task registry maps `human_H19` to four source levels. L0-L2 share the base simulator
environment; L3 is a separate tissue-box variant.

| Level | Source simulator ID | Robot-data alias | Simulator-data alias | Source task objects | RoboVerse task |
| --- | --- | --- | --- | --- | --- |
| L0 | `TwoRobotCleanDesk-v1` | `robot_H19_L0` | `L0_TwoRobotCleanDesk-v1` | YCB `026_sponge` at 0.8 scale and 0.5 kg; YCB `061_foam_brick` at 0.4 scale | `imitator_game.h19.clean_desk.l0` |
| L1 | `TwoRobotCleanDesk-v1` | `robot_H19_L1` | `L1_TwoRobotCleanDesk-v1` | L0 objects; the sponge receives the source -0.1 m X offset | `imitator_game.h19.clean_desk.l1` |
| L2 | `TwoRobotCleanDesk-v1` | `robot_H19_L2` | `L2_TwoRobotCleanDesk-v1` | YCB `036_wood_block` replaces the sponge at 0.5 scale and 0.5 kg; YCB `061_foam_brick` target | `imitator_game.h19.clean_desk.l2` |
| L3 | `TwoRobotCleanDeskL3-v1` | `robot_H19_L3` | `L3_TwoRobotCleanDesk-v1` | RoboTwin `023_tissue-box` model 3 at `(0.2, 0.13, 0.06)` scale; YCB `061_foam_brick` | `imitator_game.h19.clean_desk.l3` |

All four IDs retain the original `panda_wristcam_pair` source embodiment. They use the RoboVerse
Franka pair realization, a 2.0×1.2 m worktable, a 224×224 ZED2i pinhole camera, 100 Hz physics and
20 Hz control. The adapter uses the source nominal object positions and omits its small per-reset
position jitter. The L0/L2 sponge nominal XY is `(0.08, -0.17)`; L1 is `(-0.02, -0.17)`; the brick
nominal XY is `(-0.02, -0.02)`. H19's source sponge and L2 replacement both override actor mass to
0.5 kg. The foam-brick wrapper uses the source YCB density (1000 kg/m³) and the volume of its
scaled convex collision hull. Each task-specific YCB wrapper has its source scale baked into its
mesh so the generated mass and geometry refer to the same variant.

The source success predicate requires the sponge (L0-L2) or tissue box (L3) to be grasped while within
0.12 m of the brick, then accumulates at least 90 contact steps and 0.3 m of XY travel. RoboVerse
records the same geometric distance and wipe thresholds, but does not yet have a dual-gripper grasp
predicate with parity on all three backends. Therefore these H19 task IDs support scene/control
integration but must not be reported as exact source success evaluations. The ManiSkill planner and
demonstration trajectory are also not ported. L3 uses the exact pinned tissue-box source model; no
substitute geometry is used.

### H19 asset acquisition and licenses

The locally available `assets.tar.zst` from the pinned Imitator Game asset revision is 27,154,679
bytes, revision `2d7a339b27aff14a0780db4bd4d5e3a68c6358bc`, SHA-256
`1a4a5a2e4b6b9866ab5d75969ae0c8cec7dadf57ee20a2907d694f43a4fd0d92`. It contains the exact
ManiSkill YCB assets used for H19 L0-L2. ManiSkill identifies its assets as CC BY-NC 4.0. Retain
that attribution and non-commercial restriction; the RoboVerse/Imitator code license does not
relicense the meshes.

Prepare the three H19-only YCB variants into `roboverse_data/icl/objects/imitator_game/h19/ycb/`:

```powershell
python tools/icl/prepare_imitator_game_h19_assets.py `
  --archive D:\PycharmProjects\RoboVerse\RoboVerse\roboverse_data\icl\ig10k-source\assets.tar.zst
```

Set `$env:ROBOVERSE_DATA_DIR` first to place the generated cache outside the checkout. Preparation
checks the pinned archive digest, extracts only `026_sponge`, `036_wood_block` and `061_foam_brick`,
bakes their source scales into both visual/collision meshes, records task-instance mass and bounds,
and refuses to overwrite existing files. It changes no shared YCB cache entries.

H19 L3 has a scene adapter for the exact RoboTwin `023_tissue-box` model 3. Its source visual GLB,
collision GLB and metadata are pinned individually to the official RoboTwin 2.0 dataset revision;
the dataset card and object README declare MIT. Run the converter above to populate
`robotwin_prepared/023_tissue-box/model_3`. The adapter verifies source and generated checksums before
loading it and never substitutes another object.

### H19 use

```python
from metasim.utils.setup_util import get_task_class

Task = get_task_class("imitator_game.h19.clean_desk.l0")
env = Task(simulator="mujoco", headless=True)
```

The L0-L3 scene adapters accept `mujoco`, `genesis` and `newton`. L3 construction requires the exact
prepared tissue-box model and fails before simulator launch when the verified wrapper is absent.
The three backends have not yet been launched for this H19 slice.

## H7 Use

```python
from metasim.utils.setup_util import get_task_class

Task = get_task_class("imitator_game.h7.pick_apple_basket.l0")
env = Task(simulator="mujoco", headless=True)
```

Use `imitator_game.h7.pick_apple_basket.l3` for the plastic-box variant. Both tasks support
`mujoco`, `genesis`, and `newton`. The source environment runs at 100 Hz physics and 20 Hz control;
MetaSim uses the same timing. The current source's dual-arm planner has eight phases (reach,
grasp, lift, transport, lower, release, retract, return); the left arm moves while the right arm
holds its gripper state. This first adapter exposes the dual-robot MetaSim scene and source placement
predicate. It does not port ManiSkill's cuRobo planner or source scripted trajectory; supply actions
through the normal RoboVerse control/demonstration pipeline.

The state-based success check transforms the source object's local collision bounds into world AABBs.
L0 checks apple XY inclusion and the source basket-height threshold; L3 checks the source plastic-box
XY bounds and Z tolerance. Asset bounds are read from the prepared provenance file, not approximated.
