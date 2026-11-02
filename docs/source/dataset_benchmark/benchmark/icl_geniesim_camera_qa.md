# GenieSim Genesis and camera QA

This note records a short G1 GenieSim scene capture on Genesis, MuJoCo and
Newton after the visual-scene import fix. The captures verify scene loading and
camera rendering; they do not validate task success, real-camera calibration,
or cross-engine physics parity. Run artifacts and logs are kept under
`examples/output/qa_runs/20261031_genesis_warning_final/` and
`examples/output/qa_runs/20261031_camera_return_after_static_fix/`.

## Genesis warning diagnosis

Genesis computes geometry-derived inertial hints while it resolves an MJCF
link. For a non-watertight mesh, `Mesh.inertial` uses a convex hull for that
estimate. The warning describes this estimate; it does not, by itself, mean
that the rendered mesh or the collision shape was replaced by a convex hull.
The prepared G1 links also contain authored mass and inertia, but Genesis 1.4.5
still builds geometry hints for its import-time consistency checks.

The prepared G1 MJCF has 18 collision-mesh references backed by STL files.
Auditing them with vertex welding makes 6 watertight; 12 remain non-watertight
or non-manifold. Therefore, a file-format conversion or blanket warning filter
is not a sufficient asset fix. The durable path is to keep the visual meshes
unchanged, produce separately validated collision proxies for the remaining
meshes, and compare their contact envelopes against the source before using
them for grasp evaluation. Convexifying a whole gripper link without checking
its shape could close finger gaps and change grasp contacts.

The currently scoped warning filter is an interim Genesis 1.4.5 workaround: it
only suppresses the exact mesh-estimator message while importing the
hash-verified prepared G1 or explicitly static visual-only MJCF assets. It does
not modify geometry or physical parameters. One separate check reports
`body_link2`'s authored COM at x=0.194 m while the available geometry bounds end
at x=0.104 m. This discrepancy remains unresolved; it must be compared against
the source USD/CAD inertial frame before claiming physical calibration. The
adapter does not suppress this diagnostic.

The panorama and source environment are visual-only assets. They are imported
as fixed-base entities. Collisions are disabled only when every geom explicitly
opts out with `contype="0"` and `conaffinity="0"`; unspecified or active
collision geometry remains enabled. Warning filters are local to the entity
load and do not suppress unrelated Genesis diagnostics.

The importer-level improvement is for Genesis to expose a validation policy
that skips geometry-derived inertial estimates only when every required
mass/COM/inertia field is authored and has passed asset validation. Links with
missing or invalid inertials must continue through the estimator. RoboVerse
should keep its runtime workaround narrow until such an API is available, and
use asset preparation and collision-envelope tests to address the actual
non-watertight contact meshes.

## Three-backend capture

The test recorded 8 frames at 30 FPS with `--image-size 640` using the same G1
source instance in hold mode and the default head RGB plus two fisheye cameras.
All three backends exited successfully, produced nonblank views, and decoded all
frames without warning/error or traceback lines in their captured run logs.

| Backend | Head RGB | Left fisheye | Right fisheye | Object state |
| --- | --- | --- | --- | --- |
| Genesis | 640x400 | 640x512 | 640x512 | unchanged during hold |
| MuJoCo | 640x400 | 640x512 | 640x512 | unchanged to numerical precision |
| Newton | 640x400 | 640x512 | 640x512 | chip settles about 0.09 mm |

Newton's small chip displacement is dynamic settling during the hold; the
backdrop, source environment and table remain fixed. Genesis' task checker
remained `running`, so no task success is claimed. Renderers still differ in
exposure, shadows and material response. G1 fisheye optics continue to borrow
G2 lens parameters until measured G1 calibration is available.

The focused tests passed:

```text
pytest tests/test_icl_geniesim.py -q -k "genesis_expected_inertial_diagnostics or geniesim_observation_returns_distorted_rgb"
2 passed
```

Ruff passed for the touched Genesis adapter, scene builder and test file. The
runtime capture reports, MP4s and logs are local QA artifacts, not source assets
or benchmark data.
