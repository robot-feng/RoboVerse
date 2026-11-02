# Copyright (c) 2026 RoboVerse contributors
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from The Imitator Game task specification (https://github.com/imitator-game/The-Imitator-Game).
# Changes: reimplemented H7 L0/L3 task configuration and placement checks for MetaSim; no upstream
# source code is copied. Full upstream license: roboverse_pack/tasks/imitator_game/LICENSE.The-Imitator-Game
"""MetaSim-native H7 apple placement task for the Imitator Game benchmark."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.cameras import BaseCameraCfg, PinholeCameraCfg
from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.task.base import BaseTaskEnv
from metasim.task.registry import register_task
from metasim.utils.hf_util import LOCAL_DIR
from metasim.utils.math import quat_apply
from metasim.utils.parse_util import extract_mesh_paths_from_urdf, extract_paths_from_mjcf
from metasim.utils.xml_safe import ET
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
ASSET_ROOT = Path(LOCAL_DIR) / "icl" / "objects" / "imitator_game"
SUPPORTED_SIMULATORS = ("mujoco", "genesis", "newton")

LEFT_ROBOT = "franka_left"
RIGHT_ROBOT = "franka_right"
APPLE_NAME = "apple"
TABLE_NAME = "table"
APPLE_SCALE = 0.7
TABLE_SIZE = (1.2, 2.0, 0.05)  # Source 2.0 x 1.2 m tabletop rotated 90 degrees in the world.
TABLE_POSITION = (-0.12, 0.0, -0.025)
ROBOT_POSES = {
    LEFT_ROBOT: ((-0.60, -0.35, 0.0), (1.0, 0.0, 0.0, 0.0)),
    RIGHT_ROBOT: ((-0.60, 0.35, 0.0), (1.0, 0.0, 0.0, 0.0)),
}
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


@dataclass(frozen=True)
class AssetSpec:
    """One upstream object and the local RoboVerse cache layout required to load it."""

    key: str
    source_id: str
    namespace: Literal["ycb", "robotwin"]
    model_id: int | None = None
    scale: float = 1.0
    fixed: bool = False
    mass_kg: float | None = None

    def directory(self, root: Path) -> Path:
        """Return the prepared asset directory for this exact source object variant."""
        if self.namespace == "ycb":
            return root / "ycb" / self.source_id
        if self.model_id is None:
            raise ValueError(f"Robotwin asset {self.source_id} must pin a model_id")
        return root / "robotwin_prepared" / self.source_id / f"model_{self.model_id}"


@dataclass(frozen=True)
class LevelSpec:
    """H7's public source task identity and the assets used at a mismatch level."""

    level: Literal["L0", "L3"]
    human_task_id: str
    simulator_env_id: str
    robot_task_aliases: tuple[str, ...]
    sim_task_aliases: tuple[str, ...]
    scene_id: str
    goal_name: str
    assets: tuple[AssetSpec, ...]


LEVEL_SPECS = {
    "L0": LevelSpec(
        level="L0",
        human_task_id="human_H7",
        simulator_env_id="TwoRobotPickAppleBasket-v1",
        robot_task_aliases=("robot_H7_L0", "robot_H7_L1", "robot_H7_L2"),
        sim_task_aliases=(
            "L0_TwoRobotPickAppleBasket-v1",
            "L1_TwoRobotPickAppleBasket-v1",
            "L2_TwoRobotPickAppleBasket-v1",
        ),
        scene_id="h7_pick_apple_basket_l0",
        goal_name="breadbasket",
        assets=(
            AssetSpec("apple", "013_apple", "ycb", scale=APPLE_SCALE, mass_kg=0.5),
            AssetSpec("banana", "011_banana", "ycb"),
            AssetSpec("lemon", "014_lemon", "ycb"),
            AssetSpec("breadbasket", "076_breadbasket", "robotwin", model_id=3, scale=1.8, fixed=True),
        ),
    ),
    "L3": LevelSpec(
        level="L3",
        human_task_id="human_H7",
        simulator_env_id="TwoRobotPickAppleBasketL3-v1",
        robot_task_aliases=("robot_H7_L3", "robot_H7_L3_1"),
        sim_task_aliases=("L3_TwoRobotPickAppleBasket-v1",),
        scene_id="h7_pick_apple_basket_l3",
        goal_name="box",
        assets=(
            AssetSpec("apple", "013_apple", "ycb", scale=APPLE_SCALE, mass_kg=0.5),
            AssetSpec("box", "062_plasticbox", "robotwin", model_id=2, fixed=True),
        ),
    ),
}


@dataclass(frozen=True)
class PreparedAsset:
    """Validated paths and local-space bounds for one prepared upstream asset."""

    spec: AssetSpec
    directory: Path
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    provenance: dict


def validate_assets(level: Literal["L0", "L3"], asset_root: Path = ASSET_ROOT) -> dict[str, PreparedAsset]:
    """Validate every mesh wrapper and bounds record required by one H7 level.

    The function checks assets before a simulator is started. It never substitutes a primitive,
    another object ID, or a different model variant for a missing source asset.
    """
    try:
        spec = LEVEL_SPECS[level]
    except KeyError as exc:
        raise ValueError(f"Unsupported Imitator Game H7 level {level!r}; choose L0 or L3") from exc

    root = Path(asset_root)
    prepared: dict[str, PreparedAsset] = {}
    missing: list[str] = []
    for asset in spec.assets:
        directory = asset.directory(root)
        required = (directory / "object.urdf", directory / "object.xml", directory / "provenance.json")
        missing.extend(str(path) for path in required if not path.is_file())
        if any(not path.is_file() for path in required):
            continue

        try:
            provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
            if not isinstance(provenance, dict):
                raise ValueError("provenance.json must contain a JSON object")
            if provenance.get("source_object") != asset.source_id:
                raise ValueError("source_object does not match the requested asset ID")
            if asset.source_id == "076_breadbasket" and asset.model_id == 3:
                provenance = validate_prepared_robotwin_asset(directory, ROBOTWIN_PROFILES["h7_breadbasket"])
            if asset.model_id is not None and provenance.get("model_id") != asset.model_id:
                raise ValueError(f"model_id must be {asset.model_id}")
            raw_bounds = provenance["bounds_m"]
            bounds_tensor = torch.as_tensor(raw_bounds, dtype=torch.float64)
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
                    raise ValueError("apple wrapper must declare mass in both URDF and MJCF")
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
            f"H7 {level} requires the pinned Imitator Game assets under {root}. Missing files:\n  {paths}\n"
            "Prepare the source YCB/RoboTwin object wrappers first; see "
            "roboverse_pack/tasks/imitator_game/README.md. No object substitution is performed."
        )
    return prepared


def _euler_x_quaternion(angle: float) -> tuple[float, float, float, float]:
    """Return a scalar-first quaternion for an X-axis rotation."""
    return (math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0)


def _euler_z_quaternion(angle: float) -> tuple[float, float, float, float]:
    """Return a scalar-first quaternion for a Z-axis rotation."""
    return (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))


def _object_cfg(
    *,
    asset: PreparedAsset,
    position: tuple[float, float, float],
    orientation: tuple[float, float, float, float],
) -> RigidObjCfg:
    """Create a standard MetaSim object config for a validated source asset."""
    spec = asset.spec
    return RigidObjCfg(
        name=spec.key,
        urdf_path=str(asset.directory / "object.urdf"),
        mjcf_path=str(asset.directory / "object.xml"),
        physics=PhysicStateType.GEOM if spec.fixed else PhysicStateType.RIGIDBODY,
        fix_base_link=spec.fixed,
        scale=spec.scale,
        default_position=position,
        default_orientation=orientation,
    )


def _table_scene_objects() -> list[PrimitiveCubeCfg]:
    """Recreate the source workcell's procedural tabletop, legs, wall, and lower ground plane."""
    objects = [
        PrimitiveCubeCfg(
            name=TABLE_NAME,
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
            default_position=(0.0, 0.0, -0.775),
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
    for index, x in enumerate((-0.12 - 0.53, -0.12 + 0.53)):
        for y_index, y in enumerate((-0.93, 0.93)):
            objects.append(
                PrimitiveCubeCfg(
                    name=f"table_leg_{index}_{y_index}",
                    size=(0.08, 0.08, 0.7),
                    color=(0.68, 0.62, 0.54),
                    physics=PhysicStateType.GEOM,
                    fix_base_link=True,
                    default_position=(x, y, -0.4),
                )
            )
    return objects


def _validate_scenario_options(*, simulator: str, image_size: int) -> None:
    """Validate backend and image settings before touching the local asset cache."""
    if simulator not in SUPPORTED_SIMULATORS:
        raise ValueError(f"Unsupported simulator {simulator!r}; choose from {SUPPORTED_SIMULATORS}")
    if image_size <= 0:
        raise ValueError("image_size must be positive")


def _build_scenario(
    *,
    level: Literal["L0", "L3"],
    assets: dict[str, PreparedAsset],
    simulator: str,
    headless: bool,
    image_size: int,
    cameras: tuple[BaseCameraCfg, ...] | None,
) -> ScenarioCfg:
    """Build a dual-Panda tabletop scenario from already validated assets."""
    _validate_scenario_options(simulator=simulator, image_size=image_size)

    apple = assets["apple"]
    apple_z = -apple.bounds[0][2] * APPLE_SCALE
    apple_quat = _euler_z_quaternion(math.pi / 6)
    if level == "L0":
        basket_z = 0.0  # Source task places this Robotwin container at the tabletop origin.
        objects = [
            _object_cfg(asset=apple, position=(-0.1, -0.15, apple_z), orientation=apple_quat),
            _object_cfg(
                asset=assets["banana"], position=(0.06, 0.08, basket_z + 0.05), orientation=_euler_z_quaternion(math.pi)
            ),
            _object_cfg(
                asset=assets["lemon"],
                position=(0.03, 0.08, basket_z + 0.05),
                orientation=_euler_z_quaternion(-math.pi / 6),
            ),
            _object_cfg(
                asset=assets["breadbasket"],
                position=(0.0, 0.1, basket_z),
                orientation=_euler_x_quaternion(math.pi / 2),
            ),
        ]
    else:
        box = assets["box"]
        box_quat = _euler_x_quaternion(math.pi / 2)
        box_bounds = torch.tensor(box.bounds, dtype=torch.float64)
        box_corners = torch.stack(torch.meshgrid(*[box_bounds[:, i] for i in range(3)], indexing="ij"), dim=-1).reshape(
            -1, 3
        )
        box_corners = box_corners * box.spec.scale
        rotated = quat_apply(torch.tensor([box_quat] * len(box_corners), dtype=torch.float64), box_corners)
        box_z = -float(rotated[:, 2].min())
        objects = [
            _object_cfg(asset=apple, position=(-0.1, -0.25, apple_z), orientation=apple_quat),
            _object_cfg(asset=box, position=(0.0, 0.0, box_z), orientation=box_quat),
        ]

    objects = [*_table_scene_objects(), *objects]
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

    robots = []
    for name in (LEFT_ROBOT, RIGHT_ROBOT):
        robot = FrankaCfg(name=name)
        robot.default_position = ROBOT_POSES[name][0]
        robot.default_orientation = ROBOT_POSES[name][1]
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
        decimation=5,  # Source simulation/control frequencies are 100 Hz / 20 Hz.
        add_default_ground=False,
    )


def build_scenario(
    *,
    level: Literal["L0", "L3"],
    simulator: str,
    asset_root: Path = ASSET_ROOT,
    headless: bool = True,
    image_size: int = 224,
    cameras: tuple[BaseCameraCfg, ...] | None = None,
) -> ScenarioCfg:
    """Build one H7 L0 or L3 scenario after validating all required source assets."""
    _validate_scenario_options(simulator=simulator, image_size=image_size)
    assets = validate_assets(level, asset_root)
    return _build_scenario(
        level=level,
        assets=assets,
        simulator=simulator,
        headless=headless,
        image_size=image_size,
        cameras=cameras,
    )


def _world_bounds(local_bounds: tuple, pose: torch.Tensor, scale: float) -> torch.Tensor:
    """Transform a local axis-aligned box into batched world-aligned bounds."""
    bounds = torch.as_tensor(local_bounds, dtype=pose.dtype, device=pose.device)
    corners = torch.stack(torch.meshgrid(*[bounds[:, i] for i in range(3)], indexing="ij"), dim=-1).reshape(-1, 3)
    corners = corners * scale
    batch = pose.shape[0]
    rotated = quat_apply(
        pose[:, None, 3:7].expand(-1, len(corners), -1).reshape(-1, 4),
        corners[None].expand(batch, -1, -1).reshape(-1, 3),
    ).reshape(batch, len(corners), 3)
    world = rotated + pose[:, None, :3]
    return torch.stack((world.amin(dim=1), world.amax(dim=1)), dim=1)


class _PickAppleBasketBase(BaseTaskEnv):
    """Shared action, reset, and placement evaluation for the H7 base and L3 tasks."""

    scenario = None
    supported_simulators = SUPPORTED_SIMULATORS
    max_episode_steps = 100
    level: Literal["L0", "L3"]

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
    ) -> None:
        _validate_scenario_options(simulator=simulator, image_size=image_size)
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

    def _get_initial_states(self) -> list[dict]:
        """Initialize the two source-equivalent Panda arms and prepared objects."""
        return [
            {
                "objects": {
                    obj.name: {"pos": torch.tensor(obj.default_position), "rot": torch.tensor(obj.default_orientation)}
                    for obj in self.scenario.objects
                },
                "robots": {
                    name: {
                        "pos": torch.tensor(ROBOT_POSES[name][0]),
                        "rot": torch.tensor(ROBOT_POSES[name][1]),
                        "dof_pos": dict(PANDA_INITIAL_JOINT_POSITIONS),
                    }
                    for name in (LEFT_ROBOT, RIGHT_ROBOT)
                },
                "cameras": {},
                "extras": {},
            }
            for _ in range(self.num_envs)
        ]

    def _terminated(self, states) -> torch.Tensor:
        """Apply the source H7 level's geometry-based apple placement predicate."""
        apple_asset = self._assets["apple"]
        goal_asset = self._assets["breadbasket" if self.level == "L0" else "box"]
        apple_pose = states.objects[APPLE_NAME].root_state[:, :7]
        apple_bounds = torch.as_tensor(apple_asset.bounds, dtype=apple_pose.dtype, device=apple_pose.device)
        local_center = apple_bounds.mean(dim=0) * APPLE_SCALE
        center = quat_apply(apple_pose[:, 3:7], local_center.expand(apple_pose.shape[0], -1)) + apple_pose[:, :3]

        goal = states.objects[self._level_spec.goal_name].root_state[:, :7]
        goal_bounds = _world_bounds(goal_asset.bounds, goal, goal_asset.spec.scale)
        inside_xy = ((center[:, :2] >= goal_bounds[:, 0, :2]) & (center[:, :2] <= goal_bounds[:, 1, :2])).all(dim=-1)
        if self.level == "L0":
            inside_z = center[:, 2] <= goal_bounds[:, 1, 2] + 0.10
        else:
            inside_z = (center[:, 2] >= goal_bounds[:, 0, 2] - 0.02) & (center[:, 2] <= goal_bounds[:, 1, 2] + 0.08)
        return inside_xy & inside_z


@register_task("imitator_game.h7.pick_apple_basket.l0")
class ImitatorGamePickAppleBasketL0(_PickAppleBasketBase):
    """H7 L0: place a YCB apple into the source breadbasket with a dual-Panda cell."""

    level = "L0"


@register_task("imitator_game.h7.pick_apple_basket.l3")
class ImitatorGamePickAppleBasketL3(_PickAppleBasketBase):
    """H7 L3: reuse the apple-placement intent with the source plastic-box asset."""

    level = "L3"
