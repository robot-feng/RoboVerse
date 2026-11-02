# Copyright (c) 2026 RoboVerse contributors
# SPDX-License-Identifier: Apache-2.0
#
# Reimplemented from The Imitator Game H19 task specification; no upstream source code is copied.
# Full upstream license: roboverse_pack/tasks/imitator_game/LICENSE.The-Imitator-Game
"""MetaSim-native scene adapter for The Imitator Game H19 clean-desk task."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree as ET

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.cameras import BaseCameraCfg, PinholeCameraCfg
from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.task.base import BaseTaskEnv
from metasim.task.registry import register_task
from metasim.utils.hf_util import LOCAL_DIR
from metasim.utils.parse_util import extract_mesh_paths_from_urdf, extract_paths_from_mjcf
from roboverse_pack.robots import FrankaCfg
from roboverse_pack.tasks.imitator_game.robotwin_assets import (
    ROBOTWIN_PROFILES,
    validate_prepared_robotwin_asset,
)

SOURCE = "The-Imitator-Game"
UPSTREAM_SHA = "3c3404af98727f6d4783537732d0db8e973dfba5"
LOCAL_SOURCE_TREE_SHA256 = "b67adb166eb70411fc341985b4fa6fe48d5bf78428282d099587ae997e905ce5"
SOURCE_EMBODIMENT = "panda_wristcam_pair"
EMBODIMENT = "roboverse_franka_pair"
ASSET_ROOT = Path(LOCAL_DIR) / "icl" / "objects" / "imitator_game" / "h19"
ASSET_ARCHIVE_SHA256 = "1a4a5a2e4b6b9866ab5d75969ae0c8cec7dadf57ee20a2907d694f43a4fd0d92"
ASSET_REVISION = "2d7a339b27aff14a0780db4bd4d5e3a68c6358bc"
ASSET_LICENSE = "CC-BY-NC-4.0"
SUPPORTED_SIMULATORS = ("mujoco", "genesis", "newton")

LEFT_ROBOT = "franka_left"
RIGHT_ROBOT = "franka_right"
SPONGE_NAME = "sponge"
BRICK_NAME = "brick"
TABLE_SIZE = (1.2, 2.0, 0.05)
TABLE_POSITION = (-0.12, 0.0, -0.025)
TABLE_HEIGHT = 0.75
PANDA_INITIAL_JOINT_POSITIONS = {
    "panda_joint1": 0.0,
    "panda_joint2": math.pi / 8,
    "panda_joint3": 0.0,
    "panda_joint4": -5 * math.pi / 8,
    "panda_joint5": 0.0,
    "panda_joint6": 3 * math.pi / 4,
    "panda_joint7": math.pi / 4,
    "panda_finger_joint1": 0.04,
    "panda_finger_joint2": 0.04,
}
ROBOT_POSES = {
    LEFT_ROBOT: ((-0.615, -1.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    RIGHT_ROBOT: ((-0.615, 1.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
}
CLEAN_CONTACT_DISTANCE_M = 0.12
CLEAN_TARGET_STEPS = 90
CLEAN_TARGET_DISTANCE_M = 0.3


@dataclass(frozen=True)
class AssetSpec:
    """One exact YCB/Robotwin source object variant required by an H19 level."""

    key: str
    source_id: str
    namespace: Literal["ycb", "robotwin"]
    model_id: int | None = None
    source_scale: float | tuple[float, float, float] = 1.0
    mass_kg: float | None = None

    def directory(self, root: Path) -> Path:
        """Return the task-local prepared asset directory for this source variant."""
        if self.namespace == "ycb":
            return root / "ycb" / self.source_id
        if self.model_id is None:
            raise ValueError(f"Robotwin asset {self.source_id} must pin a model_id")
        return root / "robotwin_prepared" / self.source_id / f"model_{self.model_id}"


@dataclass(frozen=True)
class LevelSpec:
    """Public source task mapping and physical object variants for one level."""

    level: Literal["L0", "L1", "L2", "L3"]
    human_task_id: str
    simulator_env_id: str
    robot_task_aliases: tuple[str, ...]
    sim_task_aliases: tuple[str, ...]
    scene_id: str
    sponge_position: tuple[float, float]
    sponge_rotation_x: float
    sponge_rotation_z: float
    task_object_name: str
    assets: tuple[AssetSpec, ...]


_BRICK = AssetSpec("brick", "061_foam_brick", "ycb", source_scale=0.4)
LEVEL_SPECS: dict[str, LevelSpec] = {
    "L0": LevelSpec(
        level="L0",
        human_task_id="human_H19",
        simulator_env_id="TwoRobotCleanDesk-v1",
        robot_task_aliases=("robot_H19_L0",),
        sim_task_aliases=("L0_TwoRobotCleanDesk-v1",),
        scene_id="h19_clean_desk_l0",
        sponge_position=(0.08, -0.17),
        sponge_rotation_x=0.0,
        sponge_rotation_z=0.0,
        task_object_name="sponge",
        assets=(AssetSpec("sponge", "026_sponge", "ycb", source_scale=0.8, mass_kg=0.5), _BRICK),
    ),
    "L1": LevelSpec(
        level="L1",
        human_task_id="human_H19",
        simulator_env_id="TwoRobotCleanDesk-v1",
        robot_task_aliases=("robot_H19_L1",),
        sim_task_aliases=("L1_TwoRobotCleanDesk-v1",),
        scene_id="h19_clean_desk_l1",
        sponge_position=(-0.02, -0.17),
        sponge_rotation_x=0.0,
        sponge_rotation_z=0.0,
        task_object_name="sponge",
        assets=(AssetSpec("sponge", "026_sponge", "ycb", source_scale=0.8, mass_kg=0.5), _BRICK),
    ),
    "L2": LevelSpec(
        level="L2",
        human_task_id="human_H19",
        simulator_env_id="TwoRobotCleanDesk-v1",
        robot_task_aliases=("robot_H19_L2",),
        sim_task_aliases=("L2_TwoRobotCleanDesk-v1",),
        scene_id="h19_clean_desk_l2",
        sponge_position=(0.08, -0.17),
        sponge_rotation_x=math.pi / 2,
        sponge_rotation_z=0.0,
        task_object_name="sponge",
        assets=(AssetSpec("sponge", "036_wood_block", "ycb", source_scale=0.5, mass_kg=0.5), _BRICK),
    ),
    "L3": LevelSpec(
        level="L3",
        human_task_id="human_H19",
        simulator_env_id="TwoRobotCleanDeskL3-v1",
        robot_task_aliases=("robot_H19_L3",),
        sim_task_aliases=("L3_TwoRobotCleanDesk-v1",),
        scene_id="h19_clean_desk_l3",
        sponge_position=(-0.12, -0.22),
        sponge_rotation_x=math.pi / 2,
        sponge_rotation_z=math.pi / 4,
        task_object_name="tissue_box",
        assets=(
            AssetSpec(
                "tissue_box", "023_tissue-box", "robotwin", model_id=3, source_scale=(0.2, 0.13, 0.06), mass_kg=0.5
            ),
            _BRICK,
        ),
    ),
}


@dataclass(frozen=True)
class PreparedAsset:
    """Validated exact mesh wrapper and provenance for one task object."""

    spec: AssetSpec
    directory: Path
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    provenance: dict


def validate_assets(level: Literal["L0", "L1", "L2", "L3"], asset_root: Path = ASSET_ROOT) -> dict[str, PreparedAsset]:
    """Validate source IDs, baked scale, mass overrides and all referenced mesh files."""
    try:
        spec = LEVEL_SPECS[level]
    except KeyError as exc:
        raise ValueError(f"Unsupported Imitator Game H19 level {level!r}; choose L0, L1, L2 or L3") from exc

    root = Path(asset_root)
    prepared: dict[str, PreparedAsset] = {}
    missing: list[str] = []
    for asset in spec.assets:
        directory = asset.directory(root)
        required = (directory / "object.urdf", directory / "object.xml", directory / "provenance.json")
        absent = [path for path in required if not path.is_file()]
        missing.extend(str(path) for path in absent)
        if absent:
            continue

        try:
            provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
            if not isinstance(provenance, dict):
                raise ValueError("provenance.json must contain a JSON object")
            if provenance.get("source_object") != asset.source_id:
                raise ValueError("source_object does not match the requested source asset")
            if asset.namespace == "robotwin":
                if asset.source_id != "023_tissue-box" or asset.model_id != 3:
                    raise ValueError(
                        f"No pinned RoboTwin conversion profile exists for {asset.source_id} model {asset.model_id}"
                    )
                provenance = validate_prepared_robotwin_asset(directory, ROBOTWIN_PROFILES["h19_tissue_box"])
            else:
                if provenance.get("source_archive_sha256") != ASSET_ARCHIVE_SHA256:
                    raise ValueError("source_archive_sha256 does not match the pinned YCB asset archive")
                if provenance.get("source_revision") != ASSET_REVISION:
                    raise ValueError("source_revision does not match the pinned YCB asset revision")
                if provenance.get("asset_license") != ASSET_LICENSE:
                    raise ValueError(f"asset_license must be {ASSET_LICENSE} for ManiSkill YCB assets")
                recorded_scale = provenance.get("task_scale_baked_in")
                if isinstance(asset.source_scale, tuple):
                    scale_matches = (
                        isinstance(recorded_scale, list)
                        and tuple(float(v) for v in recorded_scale) == asset.source_scale
                    )
                else:
                    scale_matches = isinstance(recorded_scale, (float, int)) and math.isclose(
                        float(recorded_scale), asset.source_scale, rel_tol=0.0, abs_tol=1e-9
                    )
                if not scale_matches:
                    raise ValueError(f"task_scale_baked_in must be {asset.source_scale}")
            if asset.model_id is not None and provenance.get("model_id") != asset.model_id:
                raise ValueError(f"model_id must be {asset.model_id}")

            bounds_tensor = torch.as_tensor(provenance["bounds_m"], dtype=torch.float64)
            if bounds_tensor.shape != (2, 3) or not torch.isfinite(bounds_tensor).all():
                raise ValueError("bounds_m must be a finite 2x3 array")
            if not torch.all(bounds_tensor[0] < bounds_tensor[1]):
                raise ValueError("bounds_m lower bounds must be smaller than upper bounds")
            if asset.mass_kg is not None:
                recorded_mass = float(provenance["mass_kg"])
                if not math.isclose(recorded_mass, asset.mass_kg, rel_tol=0.0, abs_tol=1e-6):
                    raise ValueError(f"provenance mass_kg must be {asset.mass_kg} kg")
                urdf_root = ET.parse(directory / "object.urdf").getroot()
                urdf_masses = [float(item.get("value")) for item in urdf_root.findall(".//inertial/mass")]
                mjcf_root = ET.parse(directory / "object.xml").getroot()
                mjcf_masses = [float(item.get("mass")) for item in mjcf_root.findall(".//inertial") if item.get("mass")]
                if not urdf_masses or not mjcf_masses:
                    raise ValueError("source mass must be declared in both URDF and MJCF")
                if any(not math.isclose(value, asset.mass_kg, rel_tol=0.0, abs_tol=1e-6) for value in urdf_masses):
                    raise ValueError(f"URDF mass must be {asset.mass_kg} kg")
                if any(not math.isclose(value, asset.mass_kg, rel_tol=0.0, abs_tol=1e-6) for value in mjcf_masses):
                    raise ValueError(f"MJCF mass must be {asset.mass_kg} kg")
        except (ET.ParseError, OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid provenance for {asset.source_id} at {directory}: {exc}") from exc

        try:
            referenced = extract_mesh_paths_from_urdf(str(directory / "object.urdf"))
            referenced.extend(extract_paths_from_mjcf(str(directory / "object.xml")))
        except (ET.ParseError, OSError) as exc:
            raise ValueError(f"Invalid URDF/MJCF wrapper for {asset.source_id} at {directory}: {exc}") from exc
        missing_resources = sorted({path for path in referenced if not Path(path).is_file()})
        if missing_resources:
            raise FileNotFoundError(
                f"Prepared {asset.source_id} is missing referenced mesh/material resources: {missing_resources}"
            )

        prepared[asset.key] = PreparedAsset(
            spec=asset,
            directory=directory,
            bounds=(tuple(float(v) for v in bounds_tensor[0]), tuple(float(v) for v in bounds_tensor[1])),
            provenance=provenance,
        )

    if missing:
        paths = "\n  ".join(missing)
        raise FileNotFoundError(
            f"H19 {level} requires exact Imitator Game source assets under {root}. Missing files:\n  {paths}\n"
            "Prepare the exact YCB/Robotwin variants first; see roboverse_pack/tasks/imitator_game/README.md."
        )
    return prepared


def _euler_xyz_quaternion(x: float, y: float, z: float) -> tuple[float, float, float, float]:
    """Return a scalar-first quaternion for the source transforms3d ``sxyz`` rotation."""
    cx, sx = math.cos(x / 2), math.sin(x / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    cz, sz = math.cos(z / 2), math.sin(z / 2)
    return (
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    )


def _bottom_offset(bounds, orientation: tuple[float, float, float, float]) -> float:
    """Return the Z translation that puts the rotated collision bounds on the tabletop."""
    lower, upper = bounds
    corners = torch.tensor(
        [(x, y, z) for x in (lower[0], upper[0]) for y in (lower[1], upper[1]) for z in (lower[2], upper[2])],
        dtype=torch.float64,
    )
    w, x, y, z = orientation
    rotation = torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=torch.float64,
    )
    return float(-torch.min(corners @ rotation.T, dim=0).values[2])


def _table_objects() -> list[PrimitiveCubeCfg]:
    """Recreate the source 2.0 x 1.2 m table, four legs, rear wall and lower ground."""
    objects = [
        PrimitiveCubeCfg(
            name="table",
            size=TABLE_SIZE,
            color=(0.68, 0.62, 0.54),
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
            default_position=TABLE_POSITION,
        ),
        PrimitiveCubeCfg(
            name="ground",
            size=(100.0, 100.0, 0.05),
            color=(0.45, 0.45, 0.45),
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
            default_position=(0.0, 0.0, -TABLE_HEIGHT - 0.025),
        ),
        PrimitiveCubeCfg(
            name="wall",
            size=(0.1, 6.0, 3.0),
            color=(1.0, 0.9, 0.9),
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
            default_position=(0.8, 0.0, -0.3),
        ),
    ]
    for x_index, x in enumerate((-0.65, 0.41)):
        for y_index, y in enumerate((-0.93, 0.93)):
            objects.append(
                PrimitiveCubeCfg(
                    name=f"table_leg_{x_index}_{y_index}",
                    size=(0.08, 0.08, 0.7),
                    color=(0.68, 0.62, 0.54),
                    physics=PhysicStateType.GEOM,
                    fix_base_link=True,
                    default_position=(x, y, -0.4),
                )
            )
    return objects


def _build_scenario(
    *,
    level: Literal["L0", "L1", "L2", "L3"],
    assets: dict[str, PreparedAsset],
    simulator: str,
    headless: bool,
    image_size: int,
    cameras: tuple[BaseCameraCfg, ...] | None,
) -> ScenarioCfg:
    """Construct one dual-Panda scene from already validated, scale-baked source assets."""
    if cameras is None:
        aperture = 20.955
        horizontal_fov = math.pi * 1.1 / 2
        focal_length = aperture / (2 * math.tan(horizontal_fov / 2))
        cameras = (
            PinholeCameraCfg(
                name="zed2i",
                width=image_size,
                height=image_size,
                data_types=["rgb"],
                pos=(-0.4, 0.0, 0.6),
                look_at=(0.0, 0.0, 0.2),
                horizontal_aperture=aperture,
                focal_length=focal_length,
            ),
        )

    level_spec = LEVEL_SPECS[level]
    task_object = assets[level_spec.task_object_name]
    task_orientation = _euler_xyz_quaternion(level_spec.sponge_rotation_x, 0.0, level_spec.sponge_rotation_z)
    task_object_z = _bottom_offset(task_object.bounds, task_orientation)
    brick = assets["brick"]
    brick_z = -brick.bounds[0][2]
    objects = [
        *_table_objects(),
        RigidObjCfg(
            name=level_spec.task_object_name,
            urdf_path=str(task_object.directory / "object.urdf"),
            mjcf_path=str(task_object.directory / "object.xml"),
            physics=PhysicStateType.RIGIDBODY,
            default_position=(*level_spec.sponge_position, task_object_z),
            default_orientation=task_orientation,
        ),
        RigidObjCfg(
            name=BRICK_NAME,
            urdf_path=str(brick.directory / "object.urdf"),
            mjcf_path=str(brick.directory / "object.xml"),
            physics=PhysicStateType.RIGIDBODY,
            default_position=(-0.02, -0.02, brick_z),
            default_orientation=(1.0, 0.0, 0.0, 0.0),
        ),
    ]

    robots = []
    for name in (LEFT_ROBOT, RIGHT_ROBOT):
        robot = FrankaCfg(name=name)
        robot.default_position, robot.default_orientation = ROBOT_POSES[name]
        robot.default_joint_positions = dict(PANDA_INITIAL_JOINT_POSITIONS)
        robots.append(robot)

    return ScenarioCfg(
        robots=robots,
        objects=objects,
        cameras=list(cameras),
        simulator=simulator,
        num_envs=1,
        headless=headless,
        sim_params=SimParamCfg(dt=1.0 / 100.0),
        decimation=5,
        add_default_ground=False,
    )


def build_scenario(
    *,
    level: Literal["L0", "L1", "L2", "L3"],
    simulator: str,
    asset_root: Path = ASSET_ROOT,
    headless: bool = True,
    image_size: int = 224,
    cameras: tuple[BaseCameraCfg, ...] | None = None,
) -> ScenarioCfg:
    """Build one H19 tabletop scene after validating its exact source asset variants."""
    if simulator not in SUPPORTED_SIMULATORS:
        raise ValueError(f"Unsupported simulator {simulator!r}; choose from {SUPPORTED_SIMULATORS}")
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    assets = validate_assets(level, asset_root)
    return _build_scenario(
        level=level,
        assets=assets,
        simulator=simulator,
        headless=headless,
        image_size=image_size,
        cameras=cameras,
    )


class _CleanDeskBase(BaseTaskEnv):
    """Scene and geometric cleaning-progress adapter for H19 L0-L3."""

    scenario = None
    supported_simulators = SUPPORTED_SIMULATORS
    max_episode_steps = 100
    level: Literal["L0", "L1", "L2", "L3"]

    def __init__(
        self,
        scenario: ScenarioCfg | None = None,
        device=None,
        *,
        simulator: str = "mujoco",
        asset_root: Path = ASSET_ROOT,
        headless: bool = True,
        image_size: int = 224,
        cameras: tuple[BaseCameraCfg, ...] | None = None,
    ):
        if simulator not in SUPPORTED_SIMULATORS:
            raise ValueError(f"Unsupported simulator {simulator!r}; choose from {SUPPORTED_SIMULATORS}")
        self._assets = validate_assets(self.level, asset_root)
        self._level_spec = LEVEL_SPECS[self.level]
        if scenario is None:
            scenario = _build_scenario(
                level=self.level,
                assets=self._assets,
                simulator=simulator,
                headless=headless,
                image_size=image_size,
                cameras=cameras,
            )
        elif scenario.simulator != simulator:
            raise ValueError(f"Scenario backend {scenario.simulator!r} does not match requested {simulator!r}")
        super().__init__(scenario, device)
        self._clean_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._clean_distance = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._last_task_object_xy = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self._has_last_task_object = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_callback.append(self._reset_cleaning_progress)

    def _get_initial_states(self) -> list[dict]:
        """Set nominal poses and Panda joint state; source reset jitter is separately documented."""
        object_states = {
            item.name: {"pos": torch.tensor(item.default_position), "rot": torch.tensor(item.default_orientation)}
            for item in self.scenario.objects
        }
        return [
            {
                "objects": object_states,
                "robots": {
                    robot.name: {
                        "pos": torch.tensor(robot.default_position),
                        "rot": torch.tensor(robot.default_orientation),
                        "dof_pos": dict(robot.default_joint_positions),
                    }
                    for robot in self.scenario.robots
                },
                "cameras": {},
                "extras": {},
            }
        ]

    def _reset_cleaning_progress(self, env_ids: list[int]) -> None:
        """Clear H19 clean-distance accumulators when selected environments reset."""
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._clean_steps[ids] = 0
        self._clean_distance[ids] = 0
        self._has_last_task_object[ids] = False

    def _terminated(self, states) -> torch.Tensor:
        """Use the source 0.12 m proximity and 90-step/0.3 m wipe thresholds.

        The source SAPIEN task also gates this counter on its gripper-specific grasp detector. MetaSim
        does not expose equivalent dual-gripper grasp state on all three target backends yet; this
        adapter therefore records geometric proximity only and does not claim grasp-predicate parity.
        """
        task_object = states.objects[self._level_spec.task_object_name].root_state[:, :3]
        brick = states.objects[BRICK_NAME].root_state[:, :3]
        contact_region = torch.linalg.vector_norm(task_object - brick, dim=-1) <= CLEAN_CONTACT_DISTANCE_M
        delta = torch.linalg.vector_norm(task_object[:, :2] - self._last_task_object_xy, dim=-1)
        delta = delta * self._has_last_task_object.to(delta.dtype)
        self._clean_distance += delta * contact_region.to(delta.dtype)
        self._clean_steps += contact_region.to(torch.int32)
        self._last_task_object_xy = task_object[:, :2].clone()
        self._has_last_task_object[:] = True
        return (self._clean_steps >= CLEAN_TARGET_STEPS) & (self._clean_distance >= CLEAN_TARGET_DISTANCE_M)


@register_task("imitator_game.h19.clean_desk.l0")
class ImitatorGameCleanDeskL0(_CleanDeskBase):
    """H19 L0: use the source YCB sponge to wipe the YCB foam-brick target."""

    level = "L0"


@register_task("imitator_game.h19.clean_desk.l1")
class ImitatorGameCleanDeskL1(_CleanDeskBase):
    """H19 L1: apply the source -0.1 m sponge X offset before wiping."""

    level = "L1"


@register_task("imitator_game.h19.clean_desk.l2")
class ImitatorGameCleanDeskL2(_CleanDeskBase):
    """H19 L2: replace the sponge with the source YCB wood-block variant."""

    level = "L2"


@register_task("imitator_game.h19.clean_desk.l3")
class ImitatorGameCleanDeskL3(_CleanDeskBase):
    """H19 L3 tissue-box scene with the documented geometric-only cleaning metric."""

    level = "L3"
