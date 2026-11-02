"""Simulator-free contract tests for Imitator Game H7 and H19 migrations."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from imitator_game_robotwin_fixtures import make_synthetic_robotwin_source

import roboverse_pack.tasks.imitator_game.clean_desk as h19_module
import roboverse_pack.tasks.imitator_game.pick_apple_basket as h7_module
from metasim.task.registry import get_task_class
from roboverse_pack.tasks.imitator_game.clean_desk import (
    ASSET_ARCHIVE_SHA256 as H19_ASSET_ARCHIVE_SHA256,
)
from roboverse_pack.tasks.imitator_game.clean_desk import (
    ASSET_LICENSE as H19_ASSET_LICENSE,
)
from roboverse_pack.tasks.imitator_game.clean_desk import (
    ASSET_REVISION as H19_ASSET_REVISION,
)
from roboverse_pack.tasks.imitator_game.clean_desk import (
    CLEAN_CONTACT_DISTANCE_M,
    CLEAN_TARGET_DISTANCE_M,
    CLEAN_TARGET_STEPS,
    ImitatorGameCleanDeskL0,
    ImitatorGameCleanDeskL1,
    ImitatorGameCleanDeskL2,
    ImitatorGameCleanDeskL3,
)
from roboverse_pack.tasks.imitator_game.clean_desk import (
    LEVEL_SPECS as H19_LEVEL_SPECS,
)
from roboverse_pack.tasks.imitator_game.clean_desk import (
    build_scenario as build_h19_scenario,
)
from roboverse_pack.tasks.imitator_game.pick_apple_basket import (
    APPLE_SCALE,
    LEFT_ROBOT,
    LEVEL_SPECS,
    RIGHT_ROBOT,
    ImitatorGamePickAppleBasketL0,
    ImitatorGamePickAppleBasketL3,
    build_scenario,
)
from roboverse_pack.tasks.imitator_game.robotwin_assets import (
    validate_prepared_robotwin_asset,
)
from tools.icl.prepare_imitator_game_h7_assets import _set_apple_mass
from tools.icl.prepare_imitator_game_h19_assets import _set_task_mass_and_provenance
from tools.icl.prepare_robotwin_imitator_assets import prepare_robotwin_asset


def _write_test_assets(root, level: str, *, apple_mass: float = 0.5) -> None:
    """Write config-only asset stubs; these fixtures are never opened by a simulator."""
    for spec in LEVEL_SPECS[level].assets:
        directory = spec.directory(root)
        directory.mkdir(parents=True, exist_ok=True)
        mass = apple_mass if spec.key == "apple" else 0.1
        (directory / "object.urdf").write_text(
            f'<robot name="test"><link name="body"><inertial><mass value="{mass}"/></inertial></link></robot>',
            encoding="utf-8",
        )
        (directory / "object.xml").write_text(
            f'<mujoco><worldbody><body name="body"><inertial mass="{mass}"/></body></worldbody></mujoco>',
            encoding="utf-8",
        )
        provenance = {
            "source_object": spec.source_id,
            "bounds_m": [[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]],
        }
        if spec.mass_kg is not None:
            provenance["mass_kg"] = mass
        if spec.model_id is not None:
            provenance["model_id"] = spec.model_id
        (directory / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")


def _write_h19_test_assets(root, level: str, *, sponge_mass: float = 0.5) -> None:
    """Write simulator-free wrappers with the exact H19 source IDs and variants."""
    for spec in H19_LEVEL_SPECS[level].assets:
        directory = spec.directory(root)
        directory.mkdir(parents=True, exist_ok=True)
        mass = sponge_mass if spec.key in {"sponge", "tissue_box"} else 0.1
        (directory / "object.urdf").write_text(
            f'<robot name="test"><link name="body"><inertial><mass value="{mass}"/>'
            '<inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link></robot>',
            encoding="utf-8",
        )
        (directory / "object.xml").write_text(
            f'<mujoco><worldbody><body name="body"><inertial mass="{mass}" '
            'diaginertia="1 1 1"/></body></worldbody></mujoco>',
            encoding="utf-8",
        )
        scale = list(spec.source_scale) if isinstance(spec.source_scale, tuple) else spec.source_scale
        provenance = {
            "source_object": spec.source_id,
            "source_archive_sha256": H19_ASSET_ARCHIVE_SHA256,
            "source_revision": H19_ASSET_REVISION,
            "asset_license": H19_ASSET_LICENSE,
            "task_scale_baked_in": scale,
            "bounds_m": [[-0.03, -0.04, -0.02], [0.03, 0.04, 0.02]],
            "mass_kg": mass,
        }
        if spec.model_id is not None:
            provenance["model_id"] = spec.model_id
        (directory / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")


@pytest.mark.parametrize(
    ("level", "task_type", "goal_name", "goal_source_id"),
    [
        ("L0", ImitatorGamePickAppleBasketL0, "breadbasket", "076_breadbasket"),
        ("L3", ImitatorGamePickAppleBasketL3, "box", "062_plasticbox"),
    ],
)
def test_h7_level_mapping_and_dual_panda_scenario(tmp_path, monkeypatch, level, task_type, goal_name, goal_source_id):
    """Base and L3 source mappings produce a distinct two-Panda MetaSim config."""
    _write_test_assets(tmp_path, level)
    if level == "L0":
        monkeypatch.setattr(
            h7_module,
            "validate_prepared_robotwin_asset",
            lambda directory, profile: json.loads((directory / "provenance.json").read_text(encoding="utf-8")),
        )
    scenario = build_scenario(level=level, simulator="mujoco", asset_root=tmp_path)

    mapping = LEVEL_SPECS[level]
    assert mapping.human_task_id == "human_H7"
    assert mapping.goal_name == goal_name
    if level == "L3":
        assert mapping.simulator_env_id.endswith("L3-v1")
        assert mapping.robot_task_aliases == ("robot_H7_L3", "robot_H7_L3_1")
        assert mapping.sim_task_aliases == ("L3_TwoRobotPickAppleBasket-v1",)
    else:
        assert mapping.simulator_env_id == "TwoRobotPickAppleBasket-v1"
        assert mapping.robot_task_aliases == ("robot_H7_L0", "robot_H7_L1", "robot_H7_L2")
        assert mapping.sim_task_aliases == (
            "L0_TwoRobotPickAppleBasket-v1",
            "L1_TwoRobotPickAppleBasket-v1",
            "L2_TwoRobotPickAppleBasket-v1",
        )
    assert task_type.level == level
    assert len(scenario.robots) == 2
    assert [robot.name for robot in scenario.robots] == [LEFT_ROBOT, RIGHT_ROBOT]
    assert all(robot.default_position[0] == pytest.approx(-0.60) for robot in scenario.robots)
    assert all(robot.default_joint_positions["panda_joint2"] == pytest.approx(0.392699) for robot in scenario.robots)
    objects = {obj.name: obj for obj in scenario.objects}
    assert goal_name in objects
    assert objects["table"].size == pytest.approx((1.2, 2.0, 0.05))
    assert {"wall", "ground", "table_leg_0_0", "table_leg_1_1"}.issubset(objects)
    assert scenario.add_default_ground is False
    assert objects["apple"].scale == pytest.approx((APPLE_SCALE,) * 3)
    if level == "L0":
        assert objects["breadbasket"].scale == pytest.approx((1.8,) * 3)
    assert goal_source_id in str(objects[goal_name].urdf_path)
    assert scenario.cameras[0].name == "zed2i"
    assert (scenario.cameras[0].width, scenario.cameras[0].height) == (224, 224)
    assert scenario.sim_params.dt == pytest.approx(0.01)
    assert scenario.decimation == 5


def test_asset_validation_fails_fast_without_source_assets(tmp_path):
    """Missing source files name the expected variants instead of loading proxy geometry."""
    with pytest.raises(FileNotFoundError) as error:
        build_scenario(level="L0", simulator="mujoco", asset_root=tmp_path)

    message = str(error.value)
    for required in ("013_apple", "011_banana", "014_lemon", "076_breadbasket", "model_3"):
        assert required in message
    assert "No object substitution is performed" in message


def test_h7_tasks_register_under_roboversenative_ids():
    """MetaSim task discovery resolves both levels without importing a simulator backend."""
    assert get_task_class("imitator_game.h7.pick_apple_basket.l0") is ImitatorGamePickAppleBasketL0
    assert get_task_class("imitator_game.h7.pick_apple_basket.l3") is ImitatorGamePickAppleBasketL3


def test_robotwin_asset_model_id_is_pinned(tmp_path):
    """A schema-valid synthetic model 3 wrapper rejects model ID 2 before scene construction."""
    source, synthetic_profile = make_synthetic_robotwin_source(tmp_path, "h7_breadbasket")
    spec = next(item for item in LEVEL_SPECS["L0"].assets if item.key == "breadbasket")
    prepared_directory = spec.directory(tmp_path)
    prepare_robotwin_asset(
        profile=synthetic_profile,
        source_directory=source,
        output_directory=prepared_directory,
    )
    provenance_path = prepared_directory / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert validate_prepared_robotwin_asset(prepared_directory, synthetic_profile)["model_id"] == 3
    provenance["model_id"] = 2
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="prepared provenance model_id must match pinned value 3"):
        build_scenario(level="L0", simulator="mujoco", asset_root=tmp_path)


def test_cached_apple_wrapper_mass_must_match_source_override(tmp_path):
    """The shared 0.1 kg apple wrapper is rejected for H7's 0.5 kg apple instance."""
    _write_test_assets(tmp_path, "L0", apple_mass=0.1)

    with pytest.raises(ValueError, match=r"mass_kg must be 0\.5 kg"):
        build_scenario(level="L0", simulator="mujoco", asset_root=tmp_path)


def test_apple_preparation_applies_source_mass_to_both_wrappers(tmp_path):
    """The H7 preparer updates mass and inertia consistently in URDF, MJCF, and provenance."""
    apple = tmp_path / "013_apple"
    apple.mkdir()
    (apple / "object.urdf").write_text(
        '<robot name="apple"><link name="object"><inertial><mass value="0.1"/>'
        '<inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link></robot>',
        encoding="utf-8",
    )
    (apple / "object.xml").write_text(
        '<mujoco><worldbody><body name="object"><inertial mass="0.1" diaginertia="1 1 1"/></body></worldbody></mujoco>',
        encoding="utf-8",
    )
    (apple / "provenance.json").write_text(json.dumps({"source_object": "013_apple", "mass_kg": 0.1}), encoding="utf-8")

    _set_apple_mass(apple)

    assert 'value="0.5"' in (apple / "object.urdf").read_text(encoding="utf-8")
    assert 'mass="0.5"' in (apple / "object.xml").read_text(encoding="utf-8")
    assert 'ixx="5.0"' in (apple / "object.urdf").read_text(encoding="utf-8")
    assert 'diaginertia="5.0 5.0 5.0"' in (apple / "object.xml").read_text(encoding="utf-8")
    provenance = json.loads((apple / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["mass_kg"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("level", "task_type", "sponge_id", "expected_x"),
    [
        ("L0", ImitatorGameCleanDeskL0, "026_sponge", 0.08),
        ("L1", ImitatorGameCleanDeskL1, "026_sponge", -0.02),
        ("L2", ImitatorGameCleanDeskL2, "036_wood_block", 0.08),
    ],
)
def test_h19_source_level_maps_to_exact_dual_panda_scene(tmp_path, level, task_type, sponge_id, expected_x):
    """H19 L0-L2 retain original aliases, object substitutions and source workcell timing."""
    _write_h19_test_assets(tmp_path, level)
    scenario = build_h19_scenario(level=level, simulator="mujoco", asset_root=tmp_path)
    mapping = H19_LEVEL_SPECS[level]

    assert mapping.human_task_id == "human_H19"
    assert mapping.robot_task_aliases == (f"robot_H19_{level}",)
    assert mapping.sim_task_aliases == (f"{level}_TwoRobotCleanDesk-v1",)
    assert mapping.simulator_env_id == "TwoRobotCleanDesk-v1"
    assert task_type.level == level
    assert len(scenario.robots) == 2
    assert [robot.name for robot in scenario.robots] == ["franka_left", "franka_right"]
    assert scenario.robots[0].default_position == pytest.approx((-0.615, -1.0, 0.0))
    assert scenario.robots[1].default_position == pytest.approx((-0.615, 1.0, 0.0))
    objects = {item.name: item for item in scenario.objects}
    assert objects["sponge"].urdf_path.endswith(f"{sponge_id}\\object.urdf")
    assert objects["sponge"].default_position[0] == pytest.approx(expected_x)
    assert objects["brick"].urdf_path.endswith("061_foam_brick\\object.urdf")
    assert objects["table"].size == pytest.approx((1.2, 2.0, 0.05))
    assert {"wall", "ground", "table_leg_0_0", "table_leg_1_1"}.issubset(objects)
    assert scenario.cameras[0].name == "zed2i"
    assert (scenario.cameras[0].width, scenario.cameras[0].height) == (224, 224)
    assert scenario.sim_params.dt == pytest.approx(0.01)
    assert scenario.decimation == 5
    assert (CLEAN_CONTACT_DISTANCE_M, CLEAN_TARGET_STEPS, CLEAN_TARGET_DISTANCE_M) == pytest.approx((0.12, 90, 0.3))
    if level == "L2":
        assert objects["sponge"].default_orientation == pytest.approx((math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0))


def test_h19_mapping_records_distinct_l3_tissue_box_variant():
    """L3 is a separate upstream environment, not an alias for the sponge levels."""
    mapping = H19_LEVEL_SPECS["L3"]
    assert mapping.human_task_id == "human_H19"
    assert mapping.simulator_env_id == "TwoRobotCleanDeskL3-v1"
    assert mapping.robot_task_aliases == ("robot_H19_L3",)
    assert mapping.sim_task_aliases == ("L3_TwoRobotCleanDesk-v1",)
    tissue_box = next(asset for asset in mapping.assets if asset.key == "tissue_box")
    assert tissue_box.source_id == "023_tissue-box"
    assert tissue_box.model_id == 3
    assert tissue_box.source_scale == (0.2, 0.13, 0.06)
    assert H19_LEVEL_SPECS["L3"].task_object_name == "tissue_box"
    assert H19_LEVEL_SPECS["L3"].sponge_rotation_z == pytest.approx(math.pi / 4)


def test_h19_l3_fails_fast_on_missing_exact_robotwin_assets(tmp_path):
    """No proxy object is accepted when the prepared exact L3 model is absent."""
    with pytest.raises(FileNotFoundError) as error:
        build_h19_scenario(level="L3", simulator="mujoco", asset_root=tmp_path)
    assert "023_tissue-box" in str(error.value)
    assert "model_3" in str(error.value)


def test_h19_l3_builds_exact_tissue_scene_with_source_pose(tmp_path, monkeypatch):
    """L3 uses its own exact tissue-box asset name, scale-baked bounds and source orientation."""
    _write_h19_test_assets(tmp_path, "L3")
    monkeypatch.setattr(
        h19_module,
        "validate_prepared_robotwin_asset",
        lambda directory, profile: json.loads((directory / "provenance.json").read_text(encoding="utf-8")),
    )
    scenario = build_h19_scenario(level="L3", simulator="mujoco", asset_root=tmp_path)
    objects = {item.name: item for item in scenario.objects}
    tissue = objects["tissue_box"]
    bounds = ((-0.03, -0.04, -0.02), (0.03, 0.04, 0.02))
    expected_orientation = h19_module._euler_xyz_quaternion(math.pi / 2, 0.0, math.pi / 4)
    assert expected_orientation == pytest.approx((
        math.cos(math.pi / 8) / math.sqrt(2),
        math.cos(math.pi / 8) / math.sqrt(2),
        math.sin(math.pi / 8) / math.sqrt(2),
        math.sin(math.pi / 8) / math.sqrt(2),
    ))
    assert tissue.default_position[:2] == pytest.approx((-0.12, -0.22))
    assert tissue.scale == pytest.approx((1.0, 1.0, 1.0))
    assert tissue.default_orientation == pytest.approx(expected_orientation)
    assert tissue.default_position[2] == pytest.approx(h19_module._bottom_offset(bounds, expected_orientation))
    assert tissue.default_position[2] == pytest.approx(0.04)
    assert tissue.urdf_path.endswith("023_tissue-box\\model_3\\object.urdf")
    assert "sponge" not in objects
    assert objects["brick"].default_position[2] == pytest.approx(0.02)


def test_h19_source_mass_override_rejects_generic_ycb_wrapper(tmp_path):
    """The generic 0.1 kg wrapper is rejected for either H19 sponge variant."""
    _write_h19_test_assets(tmp_path, "L0", sponge_mass=0.1)
    with pytest.raises(ValueError, match=r"mass_kg must be 0\.5 kg"):
        build_h19_scenario(level="L0", simulator="mujoco", asset_root=tmp_path)


def test_h19_asset_preparer_bakes_scale_and_task_mass(tmp_path):
    """YCB preparation keeps its mesh scale and source mass in both simulator wrappers."""
    asset = tmp_path / "026_sponge"
    asset.mkdir()
    cube = (
        "v 1 1 1\nv -1 1 1\nv -1 -1 1\nv 1 -1 1\n"
        "v 1 1 -1\nv -1 1 -1\nv -1 -1 -1\nv 1 -1 -1\n"
        "f 1 2 3\nf 1 3 4\nf 5 7 6\nf 5 8 7\n"
        "f 1 5 6\nf 1 6 2\nf 2 6 7\nf 2 7 3\n"
        "f 3 7 8\nf 3 8 4\nf 4 8 5\nf 4 5 1\n"
    )
    for name in ("textured.obj", "collision.obj"):
        (asset / name).write_text(cube, encoding="utf-8")
    (asset / "object.urdf").write_text(
        '<robot name="sponge"><link name="body"><inertial><mass value="0.1"/>'
        '<inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link></robot>',
        encoding="utf-8",
    )
    (asset / "object.xml").write_text(
        '<mujoco><worldbody><body name="body"><inertial mass="0.1" diaginertia="1 1 1"/></body></worldbody></mujoco>',
        encoding="utf-8",
    )
    provenance = {"source_files_sha256": {"textured.obj": "fixture"}}
    (asset / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")

    _set_task_mass_and_provenance(asset, "026_sponge")

    assert "v 0.8 0.8 0.8" in (asset / "textured.obj").read_text(encoding="utf-8")
    assert 'value="0.5"' in (asset / "object.urdf").read_text(encoding="utf-8")
    assert 'mass="0.5"' in (asset / "object.xml").read_text(encoding="utf-8")
    provenance = json.loads((asset / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["task_scale_baked_in"] == pytest.approx(0.8)
    assert provenance["mass_kg"] == pytest.approx(0.5)
    assert provenance["asset_license"] == H19_ASSET_LICENSE


def test_h19_task_ids_register_for_all_upstream_levels():
    """Every source task-map level has a stable RoboVerse task ID."""
    task_types = (ImitatorGameCleanDeskL0, ImitatorGameCleanDeskL1, ImitatorGameCleanDeskL2, ImitatorGameCleanDeskL3)
    for level, task_type in zip(("L0", "L1", "L2", "L3"), task_types, strict=True):
        assert get_task_class(f"imitator_game.h19.clean_desk.{level.lower()}") is task_type


def test_h19_asset_manifest_matches_code_mapping_and_source_provenance():
    """The manifest records the verified exact RoboTwin source and wrapper preparation status."""
    manifest_path = Path(__file__).parents[1] / "roboverse_pack" / "tasks" / "imitator_game" / "h19_asset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["ycb_asset_archive"]["sha256"] == H19_ASSET_ARCHIVE_SHA256
    for level, spec in H19_LEVEL_SPECS.items():
        entry = manifest["levels"][level]
        assert entry["simulator_env_id"] == spec.simulator_env_id
        assert tuple(entry["robot_task_aliases"]) == spec.robot_task_aliases
        assert tuple(entry["sim_task_aliases"]) == spec.sim_task_aliases
    l3 = manifest["levels"]["L3"]
    assert l3["asset_status"] == "source_verified_wrapper_conversion_required"
    assert l3["asset"]["source_id"] == "023_tissue-box"
    assert l3["asset"]["model_id"] == 3
    assert l3["asset"]["license_status"] == "verified_in_pinned_source_metadata"
    assert l3["evaluation_status"] == "geometric_only_grasp_parity_unavailable"
