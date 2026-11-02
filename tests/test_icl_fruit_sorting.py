"""Contracts for the ICL tabletop fruit-sorting scene."""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from roboverse_pack.tasks.icl.fruit_sorting import (
    APPLE_DELIVERY_SCENE_ROOT,
    FruitSortingTask,
    GenieSimG1AppleDeliveryTask,
    point_inside_world_aabb,
)


def test_world_aabb_uses_box_rotation_and_includes_boundaries():
    bounds = torch.tensor([[-1.0, -2.0, -0.5], [1.0, 2.0, 0.5]])
    half = 2**-0.5
    pose = torch.tensor([[3.0, 4.0, 1.0, half, 0.0, 0.0, half]]).repeat(3, 1)
    points = torch.tensor([[5.0, 4.0, 1.0], [3.0, 5.1, 1.0], [3.0, 4.0, 1.6]])
    assert point_inside_world_aabb(points, pose, bounds).tolist() == [True, False, False]


def test_scoring_requires_two_consecutive_frames_and_failure_is_distinct():
    task = object.__new__(FruitSortingTask)
    task._metadata = {
        "target": "peach",
        "destination": "box",
        "objects": {"box": {"bounds_m": [[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]]}},
    }
    task._inside_frames = torch.zeros(1, dtype=torch.int32)
    task._success = torch.zeros(1, dtype=torch.bool)
    task._failure = torch.zeros(1, dtype=torch.bool)
    box = torch.tensor([[0, 0, 0.8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)
    peach = box.clone()
    state = SimpleNamespace(
        objects={"box": SimpleNamespace(root_state=box), "peach": SimpleNamespace(root_state=peach)}
    )
    task._observe_outcome(state)
    assert task._inside_frames.item() == 1 and not task._success.item()
    peach[0, 0] = 0.2
    task._observe_outcome(state)
    assert task._inside_frames.item() == 0 and not task._success.item()
    peach[0, 0] = 0.0
    task._observe_outcome(state)
    task._observe_outcome(state)
    assert task._success.item() and not task._failure.item()
    task._reset_outcome([0])
    peach[0, 2] = 0.1
    task._observe_outcome(state)
    assert task._failure.item() and not task._success.item()


def test_g1_apple_scene_checker_has_positive_and_negative_placement_cases():
    scene_path = APPLE_DELIVERY_SCENE_ROOT / "scene.json"
    if not scene_path.is_file():
        pytest.skip("Prepare the supplied G1 apple scene before checking its predicate")
    task = object.__new__(GenieSimG1AppleDeliveryTask)
    task._metadata = json.loads(scene_path.read_text(encoding="utf-8"))
    task._inside_frames = torch.zeros(1, dtype=torch.int32)
    task._success = torch.zeros(1, dtype=torch.bool)
    task._failure = torch.zeros(1, dtype=torch.bool)
    plate = torch.tensor([[0.13, 0.525, 0.78, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    apple = torch.tensor([[0.13, 0.525, 0.82, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    state = SimpleNamespace(
        objects={"apple": SimpleNamespace(root_state=apple), "pink_plate": SimpleNamespace(root_state=plate)}
    )

    task._observe_outcome(state)
    assert task._inside_frames.item() == 1 and not task._success.item()
    task._observe_outcome(state)
    assert task._success.item() and not task._failure.item()

    task._reset_outcome([0])
    apple[0, 0] = 0.13 + 0.066
    task._observe_outcome(state)
    assert task._inside_frames.item() == 0 and not task._success.item()

    task._reset_outcome([0])
    apple[0, 2] = 0.1
    task._observe_outcome(state)
    assert task._failure.item() and not task._success.item()


def test_joint_target_vector_is_bounded_and_converted_to_named_meta_sim_actions():
    task = object.__new__(FruitSortingTask)
    from roboverse_pack.tasks.icl.fruit_sorting import CONTROL_JOINTS

    robot = SimpleNamespace(
        name="agibot_g1",
        joint_limits={name: (-1.0, 1.0) for name in CONTROL_JOINTS},
    )
    task.scenario = SimpleNamespace(robots=[robot])
    task.num_envs = 1

    bounds = task.action_space
    assert bounds.shape == (len(CONTROL_JOINTS),)
    converted = task._process_action(np.zeros(len(CONTROL_JOINTS), dtype=np.float32))
    assert len(converted) == 1
    assert set(converted[0]) == {"agibot_g1"}
    assert list(converted[0]["agibot_g1"]["dof_pos_target"]) == list(CONTROL_JOINTS)
    assert set(converted[0]["agibot_g1"]["dof_pos_target"].values()) == {0.0}

    named_action = [{"agibot_g1": {"dof_pos_target": {CONTROL_JOINTS[0]: 0.25}}}]
    assert task._process_action(named_action) is named_action
    with pytest.raises(ValueError, match="shape"):
        task._process_action(np.zeros(len(CONTROL_JOINTS) - 1, dtype=np.float32))
    invalid = np.zeros(len(CONTROL_JOINTS), dtype=np.float32)
    invalid[0] = 2.0
    with pytest.raises(ValueError, match="outside the robot's limits"):
        task._process_action(invalid)


def test_prepared_scene_uses_unique_instance_names_and_open_box_colliders():
    scene_path = os.environ.get("ROBOVERSE_ICL_FRUIT_SCENE")
    if not scene_path:
        pytest.skip("Set ROBOVERSE_ICL_FRUIT_SCENE to the prepared scene")
    scene = Path(scene_path)
    meta = json.loads((scene / "scene.json").read_text(encoding="utf-8"))
    for name in meta["objects"]:
        root = ET.parse(scene / name / "object.urdf").getroot()
        assert root.get("name") == name and root.find("link").get("name") == name
    for name in ("box_left", "box_right"):
        root = ET.parse(scene / name / "object.urdf").getroot()
        assert len(root.findall("link/collision/geometry/box")) == 5
        assert not root.findall("link/collision/geometry/mesh")


def test_prepared_objects_keep_source_world_frame():
    pytest.importorskip("pxr")
    source_path = os.environ.get("ROBOVERSE_GENIESIM_SOURCE")
    scene_path = os.environ.get("ROBOVERSE_ICL_FRUIT_SCENE")
    if not source_path or not scene_path:
        pytest.skip("Set ROBOVERSE_GENIESIM_SOURCE and ROBOVERSE_ICL_FRUIT_SCENE")
    import trimesh
    from pxr import Usd, UsdGeom

    scene = Path(scene_path)
    meta = json.loads((scene / "scene.json").read_text(encoding="utf-8"))
    for name in ("box_right", "peach_target"):
        source = Path(source_path) / meta["objects"][name]["source_asset"] / "Aligned.usda"
        stage = Usd.Stage.Open(str(source))
        prim = stage.GetPrimAtPath("/World/entity/body/visual")
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
        transform = np.asarray(UsdGeom.XformCache().ComputeRelativeTransform(prim, stage.GetPrimAtPath("/World"))[0])
        world = (np.c_[points, np.ones(len(points))] @ transform)[:, :3]
        actual = trimesh.load(scene / name / "collision.ply", force="mesh", process=False)
        np.testing.assert_allclose(actual.bounds, [world.min(axis=0), world.max(axis=0)], atol=1e-6)


def test_g1_cartesian_control_uses_common_joint_order_and_source_loops():
    pytest.importorskip("mujoco")
    asset = os.environ.get("ROBOVERSE_GENIESIM_G1_CLOSED_ASSET")
    if not asset:
        pytest.skip("Set ROBOVERSE_GENIESIM_G1_CLOSED_ASSET to the closed-loop G1 asset")
    import mujoco

    from roboverse_pack.tasks.icl.control import G1CartesianController
    from roboverse_pack.tasks.icl.fruit_sorting import CONTROL_JOINTS, _robot

    robot = _robot(Path(asset))
    controller = G1CartesianController(robot, CONTROL_JOINTS)
    assert controller.model.neq == 6
    assert sum(controller.model.eq_type == mujoco.mjtEq.mjEQ_CONNECT) == 4
    assert sum(controller.model.eq_type == mujoco.mjtEq.mjEQ_JOINT) == 2
    start = np.array([robot.default_joint_positions[name] for name in CONTROL_JOINTS])
    pos, _ = controller.tcp_pose(start)
    action = np.array([0.01, 0, 0, 0, 0, 0, 0.5])
    targets = controller.joint_targets(start, action)
    end, _ = controller.tcp_pose(targets)
    assert np.linalg.norm(end - pos - action[:3]) < 0.001
    assert targets[CONTROL_JOINTS.index("idx81_gripper_r_outer_joint1")] == pytest.approx(
        0.5 * robot.joint_limits["idx81_gripper_r_outer_joint1"][1]
    )
    assert robot.file_type["genesis"] == "mjcf"
