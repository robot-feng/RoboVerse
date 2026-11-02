# RoboVerse ICL static scenes

This module describes four initial object compositions for static environment
inspection: `pick_and_place_gn`, `push_gn`, `pour_gn`, and `random_gn`. These
profiles create a robot, an explicit tabletop, prepared rigid-object wrappers,
lighting, and RGB cameras. They only define reset poses. There are no action
sequences, waypoint samplers, planners, or success predicates in this phase.
They are four visual environment profiles, not the complete SynthICL object or
placement sampling distribution. In particular, `pour_gn` currently places an
empty cup and an empty bowl; it contains no liquid and defines no pouring
behavior or transfer semantics.

## Local assets

The four prepared object wrappers are expected below
`$ROBOVERSE_DATA_DIR/icl/objects/`:

| Asset directory | Default scene use |
| --- | --- |
| `synthicl_tea_box_base2` | Pick/place source; one random-scene object |
| `synthicl_bowl_base1` | Pick/place and pour target |
| `synthicl_cup_base1` | Pour source; optional random-scene object |
| `synthicl_woodenblock_small_base3` | Push object; one random-scene object |

Every selected directory must contain `object.urdf`, `object.xml`,
`textured.obj`, `collision.obj`, `material.mtl`, `material_0.png`, and
`provenance.json`. The scene builder reads only the small bounds record from
provenance and passes ordinary URDF/MJCF paths to MetaSim. Newton's existing
backend URDF sanitizer handles local mesh paths. The probe does not download,
convert, hash, or otherwise process mesh files.

The source object package does not provide a confirmed redistribution license
in the local provenance. Keep its meshes, material textures, grasp arrays, and
derived wrappers in local ignored data storage; do not commit or publish them
until the rights holder's terms are confirmed. A source revision or checksum is
provenance, not a license grant. The optional DTD-retextured tea-box variant is
not used by the default scenes, and its image terms also need confirmation
before redistribution. The pseudo-demonstration generator code is not imported
or copied by this static-scene module.

## Robot and camera profiles

G1 is the default profile. G1 and G2 both reuse their prepared GenieSim robot
configurations and the head RGB plus two fisheye camera builders; `--robot all`
renders both robots for every selected scene. Each is positioned behind the
table for environment inspection. G1 delegates lens projection to
`geniesim_g1_cameras` and uses that helper's default calibrated mode: G1 camera
mount transforms with G2-borrowed fisheye optical parameters. That is still an
approximation until G1 receives physical lens calibration. The helper's
`fisheye_projection="pinhole"` option is a separate explicit fallback; this
scene builder does not select it. The `franka` profile remains available as a
baseline associated with the original data collection setup. Every profile
includes a separate rear-side three-quarter `overview` camera for scene review.
Camera names can be selected with `--camera-names`.

The cup's source `z_negative` orientation flag is read from its asset
provenance and converted to the same +1.57 rad Z yaw used by the source
SynthICL loader. Configuring only `overview` avoids loading the robot-mounted
camera calibration files; the default view set still loads the three head
cameras.

## Static render smoke

Point `ROBOVERSE_DATA_DIR` at the existing local data tree, then render all
four scenes on one backend. For example, in PowerShell:

```powershell
$env:ROBOVERSE_DATA_DIR = 'D:\PycharmProjects\RoboVerse\RoboVerse\roboverse_data'
python tools/icl/probe_icl_static_scenes.py `
  --sim mujoco `
  --scene all `
  --robot all `
  --image-size 640 `
  --output examples/output/qa_runs/icl_phase1_static_scenes
```

Repeat with `--sim genesis` and `--sim newton`. The probe writes one PNG
thumbnail and one MP4 per selected camera, plus one `report.json` for each
robot and scene. `--robot all` selects G1 and G2, producing both views for all
four scene profiles. Each MP4 repeats the same reset image for 30 frames at 30 fps
(about one second); it is a static inspection video, not a rollout. The report
records `physics_stepped: false`, frame/video dimensions, frame count, file
size, and the RoboVerse Git HEAD and `git status --porcelain` when available.
HEAD identifies the commit only; it does not include uncommitted worktree
changes. The report records whether the checkout is dirty, its status lines,
and a SHA-256 of the raw status output. It also records a SHA-256 for each
successfully written MP4. MP4 encoding reuses
`tools.icl.video_utils.RGBVideoWriter` (PyAV with `libx264`). Asset source
hashes are copied from each local `provenance.json`; the probe does not
recompute them. A successful scene probe establishes startup and camera
output only, not manipulation feasibility or task completion.
