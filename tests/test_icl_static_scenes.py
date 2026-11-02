"""Configuration-only checks for ICL static scene profiles."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from roboverse_pack.tasks.icl.icl_static_scenes import (
    DEFAULT_ROBOT_PROFILE,
    ICL_STATIC_SCENE_IDS,
    ICL_OBJECT_ASSET_IDS,
    ROBOT_PROFILES,
    TABLE_TOP_M,
    build_icl_static_scenario,
    get_icl_static_scene_spec,
)


def _write_asset_root(root: Path) -> None:
    for asset_id in ICL_OBJECT_ASSET_IDS:
        asset = root / asset_id
        asset.mkdir(parents=True)
        for name in (
            "object.urdf",
            "object.xml",
            "textured.obj",
            "collision.obj",
            "material.mtl",
            "material_0.png",
        ):
            (asset / name).write_text("fixture", encoding="utf-8")
        (asset / "provenance.json").write_text(
            json.dumps({"bounds_m": [[-0.02, -0.03, -0.04], [0.02, 0.03, 0.04]]}), encoding="utf-8"
        )


@pytest.mark.general
@pytest.mark.parametrize(
    ("scene_id", "expected_assets"),
    [
        ("pick_and_place_gn", ("synthicl_tea_box_base2", "synthicl_bowl_base1")),
        ("push_gn", ("synthicl_woodenblock_small_base3",)),
        ("pour_gn", ("synthicl_cup_base1", "synthicl_bowl_base1")),
        ("random_gn", ("synthicl_tea_box_base2", "synthicl_woodenblock_small_base3")),
    ],
)
def test_static_profiles_build_native_scenarios_without_simulator(scene_id, expected_assets, tmp_path):
    _write_asset_root(tmp_path)

    scenario = build_icl_static_scenario(
        simulator="mujoco",
        scene_id=scene_id,
        asset_root=tmp_path,
        robot_profile="franka",
        image_size=64,
        camera_names=("overview",),
    )

    assert scenario.simulator == "mujoco"
    assert scenario.robots[0].name == "franka"
    assert [camera.name for camera in scenario.cameras] == ["overview"]
    assert [Path(obj.urdf_path).parent.name for obj in scenario.objects if getattr(obj, "urdf_path", None)] == list(
        expected_assets
    )
    assert scenario.objects[0].name == "work_surface"
    object_config = next(obj for obj in scenario.objects if getattr(obj, "urdf_path", None))
    assert object_config.default_position[2] == pytest.approx(TABLE_TOP_M + 0.043)


@pytest.mark.general
def test_static_profiles_cover_all_four_skill_scene_ids_and_g1_is_default():
    assert set(ICL_STATIC_SCENE_IDS) == {"pick_and_place_gn", "push_gn", "pour_gn", "random_gn"}
    assert DEFAULT_ROBOT_PROFILE == "g1"
    assert set(ROBOT_PROFILES) == {"g1", "g2", "franka"}
    assert get_icl_static_scene_spec("pour_gn").objects[0].asset_id == "synthicl_cup_base1"


@pytest.mark.general
def test_random_scene_accepts_one_or_two_explicit_objects():
    single = get_icl_static_scene_spec("random_gn", ("synthicl_cup_base1",))
    pair = get_icl_static_scene_spec("random_gn", ("synthicl_cup_base1", "synthicl_bowl_base1"))
    assert len(single.objects) == 1
    assert len(pair.objects) == 2


@pytest.mark.general
@pytest.mark.parametrize(
    ("scene_id", "object_ids"),
    [
        ("unknown", None),
        ("push_gn", ("synthicl_cup_base1", "synthicl_bowl_base1")),
        ("random_gn", ICL_OBJECT_ASSET_IDS[:3]),
        ("push_gn", ("../outside",)),
    ],
)
def test_static_scene_spec_rejects_unknown_or_invalid_object_sets(scene_id, object_ids):
    with pytest.raises(ValueError):
        get_icl_static_scene_spec(scene_id, object_ids)


@pytest.mark.general
def test_static_scene_requires_all_local_wrappers(tmp_path):
    _write_asset_root(tmp_path)
    (tmp_path / "synthicl_woodenblock_small_base3" / "material_0.png").unlink()

    with pytest.raises(FileNotFoundError, match="material_0.png"):
        build_icl_static_scenario(
            simulator="newton",
            scene_id="push_gn",
            asset_root=tmp_path,
            robot_profile="franka",
            camera_names=("overview",),
        )


@pytest.mark.general
def test_g1_static_profile_reuses_fruit_scene_robot_root(tmp_path, monkeypatch):
    _write_asset_root(tmp_path)
    from roboverse_pack.robots.franka_cfg import FrankaCfg
    from roboverse_pack.tasks.icl import fruit_sorting, geniesim_scene

    received_roots = []
    camera_calls = []
    monkeypatch.setattr(
        fruit_sorting,
        "build_geniesim_g1_robot",
        lambda robot_root: received_roots.append(Path(robot_root)) or FrankaCfg(),
    )
    monkeypatch.setattr(
        geniesim_scene,
        "geniesim_g1_cameras",
        lambda robot_root, image_size: camera_calls.append((robot_root, image_size)) or [],
    )

    scenario = build_icl_static_scenario(
        simulator="newton",
        scene_id="push_gn",
        asset_root=tmp_path,
        robot_profile="g1",
        image_size=64,
        camera_names=("overview",),
    )

    assert received_roots == [Path(fruit_sorting.ROBOT_ROOT)]
    assert camera_calls == []
    assert [camera.name for camera in scenario.cameras] == ["overview"]


@pytest.mark.general
def test_g2_static_profile_reuses_prepared_robot_and_hardware_cameras(tmp_path, monkeypatch):
    _write_asset_root(tmp_path)
    from roboverse_pack.robots.franka_cfg import FrankaCfg
    from roboverse_pack.tasks.icl import geniesim_benchmark, geniesim_scene

    received_roots = []
    camera_calls = []
    monkeypatch.setattr(
        geniesim_benchmark,
        "build_geniesim_g2_robot",
        lambda robot_root: received_roots.append(Path(robot_root)) or FrankaCfg(),
    )
    monkeypatch.setattr(
        geniesim_scene,
        "geniesim_g2_cameras",
        lambda **kwargs: camera_calls.append(kwargs) or [],
    )

    scenario = build_icl_static_scenario(
        simulator="newton",
        scene_id="push_gn",
        asset_root=tmp_path,
        robot_profile="g2",
        image_size=64,
        camera_names=("overview",),
    )

    assert received_roots == [geniesim_benchmark.G2_ROBOT_ROOT]
    assert camera_calls == []
    assert scenario.robots[0].default_position == (-0.8, 0.0, 0.0)
    assert [camera.name for camera in scenario.cameras] == ["overview"]


@pytest.mark.general
def test_static_cup_applies_source_z_negative_yaw(tmp_path):
    _write_asset_root(tmp_path)
    cup_provenance_path = tmp_path / "synthicl_cup_base1" / "provenance.json"
    provenance = json.loads(cup_provenance_path.read_text(encoding="utf-8"))
    provenance["source_base_orientation"] = {"value": [0, 0, 0], "z_index": 2, "z_negative": True}
    cup_provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    scenario = build_icl_static_scenario(
        simulator="mujoco",
        scene_id="pour_gn",
        asset_root=tmp_path,
        robot_profile="franka",
        image_size=64,
        camera_names=("overview",),
    )

    cup = next(obj for obj in scenario.objects if obj.name == "source")
    half_yaw = 1.57 / 2.0
    assert cup.default_orientation == pytest.approx((math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)))
