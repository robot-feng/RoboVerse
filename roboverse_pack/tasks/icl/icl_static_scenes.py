"""Build RoboVerse ICL object scenes for static environment inspection.

These configurations describe reset poses and rendering only. They do not
contain controllers, waypoint samplers, task-success checks, or action logic.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from metasim.constants import PhysicStateType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.lights import DistantLightCfg, DomeLightCfg
from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.utils.hf_util import LOCAL_DIR

SUPPORTED_SIMULATORS = ("mujoco", "genesis", "newton")
ROBOT_PROFILES = ("g1", "g2", "franka")
DEFAULT_ROBOT_PROFILE = "g1"
DEFAULT_ASSET_ROOT = Path(LOCAL_DIR) / "icl" / "objects"

TABLE_SIZE_M = (1.0, 0.82, 0.06)
TABLE_CENTER_M = (0.18, 0.0, 0.72)
TABLE_TOP_M = TABLE_CENTER_M[2] + TABLE_SIZE_M[2] / 2.0
OBJECT_CLEARANCE_M = 0.003

ICL_OBJECT_ASSET_IDS = (
    "synthicl_bowl_base1",
    "synthicl_cup_base1",
    "synthicl_tea_box_base2",
    "synthicl_woodenblock_small_base3",
)
ROBOT_CAMERA_NAMES = ("head_rgb", "head_fisheye_left", "head_fisheye_right")


@dataclass(frozen=True)
class SceneObjectPlacement:
    """One object role and its authored tabletop reset pose."""

    name: str
    asset_id: str
    xy: tuple[float, float]
    fixed: bool = False


@dataclass(frozen=True)
class ICLStaticSceneSpec:
    """Static object composition for one ICL task family."""

    scene_id: str
    objects: tuple[SceneObjectPlacement, ...]


_SCENE_SPECS = {
    "pick_and_place_gn": ICLStaticSceneSpec(
        scene_id="pick_and_place_gn",
        objects=(
            SceneObjectPlacement("source", "synthicl_tea_box_base2", (0.0, -0.16)),
            SceneObjectPlacement("target", "synthicl_bowl_base1", (0.38, 0.16), fixed=True),
        ),
    ),
    "push_gn": ICLStaticSceneSpec(
        scene_id="push_gn",
        objects=(SceneObjectPlacement("object", "synthicl_woodenblock_small_base3", (0.20, 0.0)),),
    ),
    "pour_gn": ICLStaticSceneSpec(
        scene_id="pour_gn",
        objects=(
            SceneObjectPlacement("source", "synthicl_cup_base1", (0.0, -0.16)),
            SceneObjectPlacement("target", "synthicl_bowl_base1", (0.38, 0.16), fixed=True),
        ),
    ),
    "random_gn": ICLStaticSceneSpec(
        scene_id="random_gn",
        objects=(
            SceneObjectPlacement("object_1", "synthicl_tea_box_base2", (0.0, -0.16)),
            SceneObjectPlacement("object_2", "synthicl_woodenblock_small_base3", (0.38, 0.16)),
        ),
    ),
}
ICL_STATIC_SCENE_IDS = tuple(_SCENE_SPECS)


def get_icl_static_scene_spec(scene_id: str, object_ids: Sequence[str] | None = None) -> ICLStaticSceneSpec:
    """Return a validated, deterministic object composition for ``scene_id``.

    ``random_gn`` accepts one or two explicit object IDs. The other profiles
    have fixed object counts so that their scene roles remain unambiguous.
    """
    try:
        base = _SCENE_SPECS[scene_id]
    except KeyError as exc:
        raise ValueError(f"Unknown ICL static scene {scene_id!r}; choose from {tuple(_SCENE_SPECS)}") from exc
    if object_ids is None:
        return base

    if isinstance(object_ids, (str, bytes)):
        raise ValueError("object_ids must be a sequence of object ID strings, not one string")
    ids = tuple(object_ids)
    expected = len(base.objects)
    if scene_id == "random_gn":
        if len(ids) not in (1, 2):
            raise ValueError("random_gn accepts one or two object IDs")
    elif len(ids) != expected:
        raise ValueError(f"{scene_id} requires exactly {expected} object IDs")
    if any(not isinstance(asset_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", asset_id) for asset_id in ids):
        raise ValueError("Object IDs must contain only letters, digits, underscores, and hyphens")

    placements = tuple(
        SceneObjectPlacement(
            name=base.objects[index].name if index < len(base.objects) else f"object_{index + 1}",
            asset_id=asset_id,
            xy=base.objects[index].xy if index < len(base.objects) else (0.0, -0.16),
            fixed=base.objects[index].fixed if index < len(base.objects) else False,
        )
        for index, asset_id in enumerate(ids)
    )
    return ICLStaticSceneSpec(scene_id=scene_id, objects=placements)


def _bounds_from_provenance(path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Read the small bounds record without loading or transforming mesh data."""
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        bounds = metadata["bounds_m"]
        if len(bounds) != 2 or any(len(row) != 3 for row in bounds):
            raise ValueError
        lower = tuple(float(value) for value in bounds[0])
        upper = tuple(float(value) for value in bounds[1])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid bounds_m in {path}") from exc
    if not all(math.isfinite(value) for value in (*lower, *upper)) or any(
        upper[index] <= lower[index] for index in range(3)
    ):
        raise ValueError(f"Invalid bounds_m in {path}")
    return lower, upper


def _orientation_from_provenance(path: Path) -> tuple[float, float, float, float]:
    """Apply the SynthICL source yaw convention recorded by the asset adapter."""
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        source_orientation = metadata.get("source_base_orientation")
        if source_orientation is None:
            source_orientation = {}
        if not isinstance(source_orientation, dict):
            raise ValueError
        z_negative = source_orientation.get("z_negative", False)
        if not isinstance(z_negative, bool):
            raise ValueError
        if z_negative and source_orientation.get("z_index") != 2:
            raise ValueError
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid source_base_orientation in {path}") from exc
    yaw = 1.57 if z_negative else 0.0
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def _asset_object(placement: SceneObjectPlacement, asset_root: Path) -> RigidObjCfg:
    """Construct one rigid object from an existing local RoboVerse wrapper."""
    root = asset_root.resolve()
    asset_dir = (root / placement.asset_id).resolve()
    if not asset_dir.is_relative_to(root):
        raise ValueError(f"Object asset path escapes the asset root: {placement.asset_id}")
    required = (
        "object.urdf",
        "object.xml",
        "provenance.json",
        "textured.obj",
        "collision.obj",
        "material.mtl",
        "material_0.png",
    )
    missing = [name for name in required if not (asset_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Prepared object {placement.asset_id!r} is incomplete; missing: {missing}")
    lower, _ = _bounds_from_provenance(asset_dir / "provenance.json")
    orientation = _orientation_from_provenance(asset_dir / "provenance.json")
    z = TABLE_TOP_M - lower[2] + OBJECT_CLEARANCE_M
    if not math.isfinite(z):
        raise ValueError(f"Invalid table support height for {placement.asset_id!r}")
    resources = [
        str(asset_dir / name)
        for name in ("textured.obj", "collision.obj", "material.mtl", "material_0.png")
    ]
    return RigidObjCfg(
        name=placement.name,
        urdf_path=str(asset_dir / "object.urdf"),
        mjcf_path=str(asset_dir / "object.xml"),
        extra_resources=resources,
        physics=PhysicStateType.GEOM if placement.fixed else PhysicStateType.RIGIDBODY,
        fix_base_link=placement.fixed,
        default_position=(placement.xy[0], placement.xy[1], z),
        default_orientation=orientation,
    )


def _franka_cameras(image_size: int) -> list[PinholeCameraCfg]:
    height = max(1, round(image_size * 0.625))
    return [
        PinholeCameraCfg(
            name=name,
            pos=pos,
            look_at=look_at,
            width=image_size,
            height=height,
            data_types=["rgb"],
        )
        for name, pos, look_at in (
            ("front", (1.3, -1.2, 1.35), (0.20, 0.0, 0.70)),
            ("side", (0.45, 1.15, 1.1), (0.20, 0.0, 0.70)),
            ("overhead", (0.20, 0.0, 1.75), (0.20, 0.0, 0.73)),
        )
    ]


def _overview_camera(image_size: int) -> PinholeCameraCfg:
    return PinholeCameraCfg(
        name="overview",
        pos=(1.7, -1.8, 1.9),
        look_at=(0.0, 0.15, 0.75),
        width=image_size,
        height=image_size,
        data_types=["rgb"],
        focal_length=20.0,
    )


def _select_cameras(cameras: Sequence[PinholeCameraCfg], camera_names: Sequence[str] | None):
    available = {camera.name: camera for camera in cameras}
    if camera_names is None:
        return list(available.values())
    requested = tuple(camera_names)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("camera_names must contain unique non-empty names")
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown camera names {unknown}; available: {tuple(available)}")
    return [available[name] for name in requested]


def build_icl_static_scenario(
    *,
    simulator: Literal["mujoco", "genesis", "newton"],
    scene_id: str,
    asset_root: Path | None = None,
    object_ids: Sequence[str] | None = None,
    robot_profile: Literal["g1", "g2", "franka"] = DEFAULT_ROBOT_PROFILE,
    robot_root: Path | None = None,
    image_size: int = 640,
    camera_names: Sequence[str] | None = None,
    headless: bool = True,
) -> ScenarioCfg:
    """Build a static object scene for MuJoCo, Genesis, or Newton.

    This function validates local wrappers, uses their provenance bounds to
    place them on the explicit work surface, and returns a normal MetaSim
    ``ScenarioCfg``. It does not download assets or execute simulation steps.
    """
    if simulator not in SUPPORTED_SIMULATORS:
        raise ValueError(f"Unsupported simulator {simulator!r}; choose from {SUPPORTED_SIMULATORS}")
    if robot_profile not in ROBOT_PROFILES:
        raise ValueError(f"Unsupported robot profile {robot_profile!r}; choose from {ROBOT_PROFILES}")
    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size < 16:
        raise ValueError("image_size must be an integer of at least 16 pixels")
    if not isinstance(headless, bool):
        raise ValueError("headless must be a boolean")

    spec = get_icl_static_scene_spec(scene_id, object_ids)
    assets = Path(asset_root) if asset_root is not None else DEFAULT_ASSET_ROOT
    objects = [
        PrimitiveCubeCfg(
            name="work_surface",
            size=TABLE_SIZE_M,
            color=(0.24, 0.26, 0.29),
            physics=PhysicStateType.GEOM,
            default_position=TABLE_CENTER_M,
        ),
        *(_asset_object(placement, assets) for placement in spec.objects),
    ]

    if robot_profile == "g1":
        # Reuse the existing RoboVerse G1 and camera profile lazily so a
        # configuration-only Franka test does not import the robot task stack.
        from roboverse_pack.tasks.icl.fruit_sorting import ROBOT_ROOT, build_geniesim_g1_robot
        from roboverse_pack.tasks.icl.geniesim_scene import geniesim_g1_cameras

        selected_robot_root = Path(robot_root) if robot_root is not None else Path(ROBOT_ROOT)
        robot = build_geniesim_g1_robot(selected_robot_root)
        needs_robot_cameras = camera_names is None or bool(set(camera_names) & set(ROBOT_CAMERA_NAMES))
        robot_cameras = geniesim_g1_cameras(selected_robot_root, image_size) if needs_robot_cameras else []
        cameras = [*robot_cameras, _overview_camera(image_size)]
    elif robot_profile == "g2":
        # Reuse the pinned G2 model and GenieSim hardware camera profile.
        # Keep the robot behind the table, matching the G1 static layout.
        from roboverse_pack.tasks.icl.geniesim_benchmark import G2_ROBOT_ROOT, build_geniesim_g2_robot
        from roboverse_pack.tasks.icl.geniesim_scene import geniesim_g2_cameras

        selected_robot_root = Path(robot_root) if robot_root is not None else Path(G2_ROBOT_ROOT)
        robot = build_geniesim_g2_robot(selected_robot_root)
        robot.default_position = (-0.8, 0.0, 0.0)
        needs_robot_cameras = camera_names is None or bool(set(camera_names) & set(ROBOT_CAMERA_NAMES))
        robot_cameras = (
            geniesim_g2_cameras(image_size=image_size, camera_profile="hardware") if needs_robot_cameras else []
        )
        cameras = [*robot_cameras, _overview_camera(image_size)]
    else:
        from roboverse_pack.robots.franka_cfg import FrankaCfg

        robot = FrankaCfg()
        robot.default_position = (-0.30, 0.0, TABLE_TOP_M)
        cameras = [*_franka_cameras(image_size), _overview_camera(image_size)]

    sim_params = SimParamCfg(mujoco_use_scenario_lights=True)
    if hasattr(sim_params, "newton_use_scenario_lights"):
        sim_params.newton_use_scenario_lights = True
    scenario = ScenarioCfg(
        robots=[robot],
        objects=objects,
        cameras=_select_cameras(cameras, camera_names),
        lights=[DomeLightCfg(intensity=450), DistantLightCfg(intensity=1100, polar=30, azimuth=45)],
        simulator=simulator,
        num_envs=1,
        headless=headless,
        decimation=1,
        sim_params=sim_params,
    )
    return scenario
