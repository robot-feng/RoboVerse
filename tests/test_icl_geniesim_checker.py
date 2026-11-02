"""Behavioral tests for the imported Genie Sim checker graph evaluator."""

import numpy as np
import pytest

from roboverse_pack.tasks.icl.geniesim_checker import (
    CheckerFrame,
    CheckerStatus,
    GenieSimChecker,
    ObjectObservation,
    available_problem_ids,
    checker_graph_vocabulary,
    compile_problem_graph,
)


def _pose(position=(0.0, 0.0, 0.0), rotation=None):
    result = np.eye(4)
    if rotation is not None:
        result[:3, :3] = rotation
    result[:3, 3] = position
    return result


def _frame(step, *, objects=None, grippers=None, robot_pose=None, vlm_scores=None):
    return CheckerFrame(
        step=step,
        objects=objects or {},
        grippers=grippers or {},
        robot_pose=robot_pose if robot_pose is not None else np.eye(4),
        vlm_scores=vlm_scores or {},
    )


def test_inside_matches_source_scale_and_two_frame_hold():
    graph = compile_problem_graph({"Acts": [{"ActionList": [{"ActionSetWaitAny": [{"Inside": "fruit|box|0.8"}]}]}]})
    checker = GenieSimChecker(graph)
    box = ObjectObservation(_pose(), bounds=np.array([[-1, -1, -1], [1, 1, 1]]))
    fruit = ObjectObservation(_pose((0.75, 0.0, 0.0)))

    assert checker.step(_frame(0, objects={"fruit": fruit, "box": box})).status == CheckerStatus.RUNNING
    assert checker.step(_frame(1, objects={"fruit": fruit, "box": box})).status == CheckerStatus.SUCCESS


def test_problem_sequence_only_activates_the_current_stage():
    graph = compile_problem_graph({
        "Acts": [
            {
                "ActionList": [
                    {"ActionSetWaitAny": [{"LiftUp": "cube|0.1"}, {"StepOut": 1}]},
                    {"ActionSetWaitAny": [{"Inside": "cube|bin|1.0"}, {"StepOut": 5}]},
                ]
            }
        ]
    })
    checker = GenieSimChecker(graph)
    cube = ObjectObservation(_pose((0, 0, 0)))
    bin_obj = ObjectObservation(_pose(), bounds=np.array([[-0.2] * 3, [0.2] * 3]))
    frame0 = {"cube": cube, "bin": bin_obj}
    assert checker.step(_frame(0, objects=frame0)).status == CheckerStatus.RUNNING

    lifted = ObjectObservation(_pose((0, 0, 0.11)))
    assert checker.step(_frame(1, objects={"cube": lifted, "bin": bin_obj})).status == CheckerStatus.RUNNING
    assert checker.step(_frame(2, objects={"cube": lifted, "bin": bin_obj})).status == CheckerStatus.SUCCESS


def test_failure_predicates_propagate_out_of_wait_any():
    checker = GenieSimChecker(
        compile_problem_graph({
            "Acts": [{"ActionSetWaitAny": [{"Inside": "item|box|1.0"}, {"Onfloor": "item|0"}, {"StepOut": 10}]}]
        })
    )
    item = ObjectObservation(_pose((0, 0, 0.1)))
    box = ObjectObservation(_pose(), bounds=np.array([[-0.3] * 3, [0.3] * 3]))
    assert checker.step(_frame(0, objects={"item": item, "box": box})).status == CheckerStatus.FAILURE


def test_pickup_checker_uses_initial_height_and_named_gripper():
    checker = GenieSimChecker(compile_problem_graph({"Acts": [{"PickUpOnGripper": "cube|right_gripper"}]}))
    cube = ObjectObservation(_pose((0, 0, 0)))
    gripper = _pose((0, 0, 0))
    assert checker.step(_frame(0, objects={"cube": cube}, grippers={"right_gripper": gripper})).status == (
        CheckerStatus.RUNNING
    )
    lifted = ObjectObservation(_pose((0, 0, 0.021)))
    assert checker.step(_frame(1, objects={"cube": lifted}, grippers={"right_gripper": gripper})).status == (
        CheckerStatus.SUCCESS
    )


def test_pickup_source_requires_strictly_more_than_two_centimeters():
    checker = GenieSimChecker(compile_problem_graph({"Acts": [{"PickUpOnGripper": "cube|right_gripper"}]}))
    grip = {"right_gripper": _pose()}
    checker.step(_frame(0, objects={"cube": ObjectObservation(_pose())}, grippers=grip))
    assert checker.step(_frame(1, objects={"cube": ObjectObservation(_pose((0, 0, 0.02)))}, grippers=grip)).status == (
        CheckerStatus.RUNNING
    )
    assert checker.step(_frame(2, objects={"cube": ObjectObservation(_pose((0, 0, 0.021)))}, grippers=grip)).status == (
        CheckerStatus.SUCCESS
    )


def test_multi_object_follow_pickup_and_placeholder_propagation():
    checker = GenieSimChecker(
        compile_problem_graph({
            "Acts": [
                {
                    "ActionList": [
                        {"Follow": "first,second|[0.2,0.2,0.2]|right_gripper"},
                        {"PickUpOnGripper": "first,second|right_gripper"},
                        {"Inside": "{@placeholder_str1}|bin|0.8"},
                    ]
                }
            ]
        })
    )
    first = ObjectObservation(_pose((0.0, 0.0, 0.0)))
    second = ObjectObservation(_pose((1.0, 0.0, 0.0)))
    bin_obj = ObjectObservation(_pose((1.0, 0.0, 0.03)), bounds=np.array([[-0.25] * 3, [0.25] * 3]))
    initial = {"first": first, "second": second, "bin": bin_obj}

    # Follow selects the second candidate and the pickup checker records each candidate's start height.
    assert checker.step(_frame(0, objects=initial, grippers={"right_gripper": _pose((1.0, 0.0, 0.0))})).status == (
        CheckerStatus.RUNNING
    )
    raised = ObjectObservation(_pose((1.0, 0.0, 0.03)))
    lifted = {**initial, "second": raised}
    assert (
        checker.step(_frame(1, objects=lifted, grippers={"right_gripper": _pose((1.0, 0.0, 0.03))})).status
        == CheckerStatus.RUNNING
    )
    assert (
        checker.step(_frame(2, objects=lifted, grippers={"right_gripper": _pose((1.0, 0.0, 0.03))})).status
        == CheckerStatus.SUCCESS
    )
    assert checker._placeholders["placeholder_str1"] == "second"


def test_follow_skips_objects_matched_in_previous_rounds():
    checker = GenieSimChecker(
        compile_problem_graph({
            "Acts": [
                {
                    "ActionList": [
                        {"Follow": "first,second|[0.2,0.2,0.2]|right_gripper"},
                        {"Follow": "first,second|[0.2,0.2,0.2]|right_gripper"},
                    ]
                }
            ]
        })
    )
    objects = {"first": ObjectObservation(_pose()), "second": ObjectObservation(_pose((1, 0, 0)))}
    assert checker.step(_frame(0, objects=objects, grippers={"right_gripper": _pose((1, 0, 0))})).status == (
        CheckerStatus.RUNNING
    )
    assert checker.step(_frame(1, objects=objects, grippers={"right_gripper": _pose()})).status == (
        CheckerStatus.SUCCESS
    )
    assert checker._followed_objects == {"first", "second"}


def test_relative_position_uses_the_robot_frame_and_needs_two_frames():
    robot = _pose(rotation=np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]))
    bounds = np.array([[-0.01] * 3, [0.01] * 3])
    item_a, item_b = ObjectObservation(_pose((-0.2, 0, 0)), bounds), ObjectObservation(_pose(), bounds)
    checker = GenieSimChecker(compile_problem_graph({"Acts": [{"RelativePosition": "a|b|leftof"}]}))
    assert checker.step(_frame(0, objects={"a": item_a, "b": item_b}, robot_pose=robot)).status == (
        CheckerStatus.RUNNING
    )
    assert checker.step(_frame(1, objects={"a": item_a, "b": item_b}, robot_pose=robot)).status == (
        CheckerStatus.SUCCESS
    )


def test_relative_position_uses_world_geometry_centers_and_source_threshold_rules():
    # Both roots coincide.  After the 90-degree rotation, the offset visual
    # center of A is 0.1 m in robot-left direction from B.
    rotated = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    obj_a = ObjectObservation(_pose(rotation=rotated), np.array([[0.09, -0.01, -0.01], [0.11, 0.01, 0.01]]))
    obj_b = ObjectObservation(_pose(), np.array([[-0.01] * 3, [0.01] * 3]))
    objects = {"a": obj_a, "b": obj_b}
    # A custom directional threshold is ignored by the source implementation.
    directional = GenieSimChecker(compile_problem_graph({"Acts": [{"RelativePosition": "a|b|leftof|0.5"}]}))
    assert directional.step(_frame(0, objects=objects)).status == CheckerStatus.RUNNING
    assert directional.step(_frame(1, objects=objects)).status == CheckerStatus.SUCCESS

    # In contrast, custom alignment thresholds are honored.
    aligned = GenieSimChecker(compile_problem_graph({"Acts": [{"RelativePosition": "a|b|aligned_y|0.05"}]}))
    assert aligned.step(_frame(0, objects=objects)).status == CheckerStatus.RUNNING
    assert aligned.step(_frame(1, objects=objects)).status == CheckerStatus.RUNNING


def test_source_entity_path_uses_child_rotation_for_follow_and_upright():
    # The source beverage assets have an entity frame rotated 90 degrees about
    # X, so its local Y points upwards even when the object root is identity.
    entity = _pose(rotation=np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))
    obj = ObjectObservation(_pose(), subframes={"entity": entity})
    ref = "/Workspace/Objects/bottle/entity"
    checker = GenieSimChecker(compile_problem_graph({"Acts": [{"ActionList": [
        {"Follow": f"{ref}|[0.02,0.2,0.02]|left_gripper"},
        {"Upright": f"{ref}|15"},
    ]}]}))
    assert checker.step(_frame(0, objects={"bottle": obj}, grippers={"left_gripper": _pose((0, 0, 0.09))})).status == (
        CheckerStatus.SUCCESS
    )
    assert checker._placeholders["placeholder_str1"] == ref


def test_source_child_pose_is_not_silently_replaced_with_object_root():
    checker = GenieSimChecker(compile_problem_graph({"Acts": [{
        "Upright": "/Workspace/Objects/bottle/entity|15"
    }]}))
    with pytest.raises(KeyError, match="no verified source subframe"):
        checker.step(_frame(0, objects={"bottle": ObjectObservation(_pose())}))


def test_stock_shelf_source_entity_frames_match_pinned_usd():
    import json
    from pathlib import Path

    pytest.importorskip("pxr")
    from pxr import Usd

    from roboverse_pack.tasks.icl.geniesim_scene import geniesim_checker_subframes
    from tools.icl.prepare_geniesim_objects import _source_subframes

    root = Path(__file__).resolve().parents[1]
    spec = root / "roboverse_data/icl/geniesim_benchmark_specs_source_scene_v2/benchmark.json"
    asset_root = root / "roboverse_data/icl/objects/geniesim"
    source = root / "datasets/icl/sources/geniesim-assets/objects/genie/beverage_bottle"
    if not spec.is_file() or not (source / "genie_beverage_bottle_008/Aligned.usda").is_file():
        pytest.skip("Prepare source-scene and beverage assets first")
    instance = json.loads(spec.read_text(encoding="utf-8"))["instances"]["stock_and_straighten_shelf/0"]
    transforms = geniesim_checker_subframes(instance, asset_root)
    for instance_name, asset_id in (
        ("genie_beverage_bottle_008", "genie_beverage_bottle_008"),
        ("genie_beverage_bottle_009_06", "genie_beverage_bottle_009"),
    ):
        source_matrix = _source_subframes(Usd.Stage.Open(str(source / asset_id / "Aligned.usda")))["entity"]
        np.testing.assert_allclose(transforms[instance_name]["entity"], source_matrix, atol=1e-8)


def test_source_background_bounds_are_available_to_checker_without_fabricating_collisions(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom

    from roboverse_pack.tasks.icl.geniesim_scene import geniesim_checker_background_objects

    source_path = tmp_path / "background.usda"
    stage = Usd.Stage.CreateNew(str(source_path))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    cube = UsdGeom.Cube.Define(stage, "/World/background/tray")
    cube.CreateSizeAttr(2.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
    stage.GetRootLayer().Save()
    problem = {"Acts": [{"Inside": "fruit|/World/background/tray|1.0"}]}
    instance = {"checker_graphs": {"problem0": problem}}
    static = geniesim_checker_background_objects(instance, "background.usda", tmp_path)
    np.testing.assert_allclose(static["/World/background/tray"].bounds, [[0, 1, 2], [2, 3, 4]])
    checker = GenieSimChecker(compile_problem_graph(problem))
    observations = {**static, "fruit": ObjectObservation(_pose((1, 2, 3)))}
    assert checker.step(_frame(0, objects=observations)).status == CheckerStatus.RUNNING
    assert checker.step(_frame(1, objects=observations)).status == CheckerStatus.SUCCESS


def test_mixed_rules_and_vlm_have_explicit_scoring_inputs():
    checker = GenieSimChecker(
        compile_problem_graph({
            "Acts": [
                {
                    "MixedRules": {
                        "mode": "and",
                        "check_interval": 1,
                        "rules": [
                            {"name": "InBBox", "params": "cube|0,0,0|0.2,0.2,0.2"},
                            {"name": "Upright", "params": "cube|15.0"},
                        ],
                    }
                }
            ]
        })
    )
    upright_rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    cube = ObjectObservation(_pose(rotation=upright_rotation), bounds=np.array([[-0.1] * 3, [0.1] * 3]))
    assert checker.step(_frame(0, objects={"cube": cube})).status == CheckerStatus.RUNNING
    assert checker.step(_frame(1, objects={"cube": cube})).status == CheckerStatus.SUCCESS

    visual = GenieSimChecker(compile_problem_graph({"Acts": [{"VLM": "arrange|1"}]}))
    with pytest.raises(RuntimeError, match="needs a score"):
        visual.step(_frame(0))
    assert visual.step(_frame(1, vlm_scores={"arrange": 1.0})).status == CheckerStatus.SUCCESS


def test_mixed_rules_or_mode_accumulates_all_checker_passes():
    checker = GenieSimChecker(
        compile_problem_graph({
            "Acts": [
                {
                    "MixedRules": {
                        "mode": "or",
                        "rules": [
                            {"name": "Upright", "params": "cube|15.0"},
                            {"name": "InBBox", "params": "cube|1,0,0|0.2,0.2,0.2"},
                        ],
                    }
                }
            ]
        })
    )
    upright = ObjectObservation(_pose(rotation=np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])))
    assert checker.step(_frame(0, objects={"cube": upright})).status == CheckerStatus.RUNNING

    moved_and_tilted = ObjectObservation(_pose((1, 0, 0)))
    assert checker.step(_frame(1, objects={"cube": moved_and_tilted})).status == CheckerStatus.RUNNING
    assert checker.step(_frame(2, objects={"cube": moved_and_tilted})).status == CheckerStatus.SUCCESS


def test_push_pull_counts_successful_samples_even_when_nonconsecutive():
    checker = GenieSimChecker(compile_problem_graph({"Acts": [{"PushPull": "drawer|0.2|0.4"}]}))
    for step, joint in enumerate((0.3, 0.0)):
        obj = ObjectObservation(_pose(), joint_positions=np.array([joint]))
        assert checker.step(_frame(step, objects={"drawer": obj})).status == CheckerStatus.RUNNING
    obj = ObjectObservation(_pose(), joint_positions=np.array([0.3]))
    assert checker.step(_frame(2, objects={"drawer": obj})).status == CheckerStatus.SUCCESS


def test_push_pull_resolves_source_background_object_path():
    checker = GenieSimChecker(
        compile_problem_graph({"Acts": [{"PushPull": "/World/objects/benchmark_door_000|0.05|1.5708|0"}]})
    )
    door = ObjectObservation(_pose(), joint_positions=np.array([0.5, 0.0]))
    assert checker.step(_frame(0, objects={"benchmark_door_000": door})).status == CheckerStatus.RUNNING
    assert checker.step(_frame(1, objects={"benchmark_door_000": door})).status == CheckerStatus.SUCCESS


def test_checker_frame_reorders_object_joint_state_to_source_order():
    from types import SimpleNamespace

    from roboverse_pack.tasks.icl.geniesim_checker import checker_frame_from_states

    states = SimpleNamespace(
        objects={
            "door": SimpleNamespace(
                root_state=np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
                joint_pos=np.array([[0.7, 0.2]]),
            )
        },
        robots={
            "robot": SimpleNamespace(
                root_state=np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
                body_names=["base"],
                body_state=np.array([[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]]),
            )
        },
    )
    frame = checker_frame_from_states(
        states,
        robot_name="robot",
        step=0,
        object_joint_indices={"door": [1, 0]},
        gripper_links={"test_gripper": "base"},
    )
    assert frame.objects["door"].joint_positions.tolist() == [0.2, 0.7]


def test_follow_stack_and_stable_grasp_predicates():
    follow = GenieSimChecker(compile_problem_graph({"Acts": [{"Follow": "cube|[0.2,0.2,0.2]|left_gripper"}]}))
    cube = ObjectObservation(_pose())
    assert follow.step(_frame(0, objects={"cube": cube}, grippers={"left_gripper": _pose()})).status == (
        CheckerStatus.SUCCESS
    )

    stack = GenieSimChecker(compile_problem_graph({"Acts": [{"Stack": "[a,b]|[0.05,0.05]"}]}))
    objects = {"a": ObjectObservation(_pose()), "b": ObjectObservation(_pose((0.02, 0.01, 0.5)))}
    assert stack.step(_frame(0, objects=objects)).status == CheckerStatus.RUNNING
    assert stack.step(_frame(1, objects=objects)).status == CheckerStatus.SUCCESS

    stable = GenieSimChecker(compile_problem_graph({"Acts": [{"StableGrasp": "cube|right_gripper|0.1|0.02|0.1"}]}))
    for step in range(3):
        position = np.array([step * 0.06, 0.0, 0.0])
        obj = ObjectObservation(_pose(position))
        hand = _pose(position + np.array([0.01, 0.0, 0.0]))
        result = stable.step(_frame(step, objects={"cube": obj}, grippers={"right_gripper": hand})).status
    assert result == CheckerStatus.SUCCESS


def test_ontop_and_cover_use_rotated_object_bounds():
    unit_bounds = np.array([[-0.1] * 3, [0.1] * 3])
    base = ObjectObservation(_pose(), bounds=unit_bounds)
    top = ObjectObservation(_pose((0.0, 0.0, 0.2)), bounds=unit_bounds)
    ontop = GenieSimChecker(compile_problem_graph({"Acts": [{"Ontop": "top|base"}]}))
    assert ontop.step(_frame(0, objects={"base": base, "top": top})).status == CheckerStatus.SUCCESS

    lid = ObjectObservation(_pose((0.0, 0.0, 0.0)), bounds=unit_bounds)
    lower = ObjectObservation(_pose((0.0, 0.0, -0.05)), bounds=unit_bounds)
    cover = GenieSimChecker(compile_problem_graph({"Acts": [{"Cover": "lid|lower"}]}))
    assert cover.step(_frame(0, objects={"lid": lid, "lower": lower})).status == CheckerStatus.SUCCESS


def test_wait_groups_timers_and_subtask_steps_compile():
    all_checker = GenieSimChecker(
        compile_problem_graph({"Acts": [{"ActionSetWaitAll": [{"Upright": "a|15"}, {"Upright": "b|15"}]}]})
    )
    upright = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    both = {"a": ObjectObservation(_pose(rotation=upright)), "b": ObjectObservation(_pose(rotation=upright))}
    assert all_checker.step(_frame(0, objects=both)).status == CheckerStatus.SUCCESS

    some = GenieSimChecker(
        compile_problem_graph({
            "Acts": [{"ActionSetWaitSome_2": [{"Upright": "a|15"}, {"Upright": "b|15"}, {"Upright": "c|15"}]}]
        })
    )
    three = {**both, "c": ObjectObservation(_pose())}
    assert some.step(_frame(0, objects=three)).status == CheckerStatus.SUCCESS

    timer = GenieSimChecker(compile_problem_graph({"Acts": [{"ActionWaitForTime": 0.05}]}))
    assert timer.step(CheckerFrame(step=0, objects={}, delta_time=0.03)).status == CheckerStatus.RUNNING
    assert timer.step(CheckerFrame(step=1, objects={}, delta_time=0.03)).status == CheckerStatus.SUCCESS

    subtask = GenieSimChecker.from_subtask_steps([{"checker": {"LiftUp": "item|0.1"}, "step_out": 10}])
    initial = ObjectObservation(_pose())
    assert subtask.step(_frame(0, objects={"item": initial})).status == CheckerStatus.RUNNING
    raised = ObjectObservation(_pose((0, 0, 0.1)))
    assert subtask.step(_frame(1, objects={"item": raised})).status == CheckerStatus.SUCCESS


def test_vocabulary_audit_includes_rules_nested_in_mixed_rules():
    assert checker_graph_vocabulary({"MixedRules": {"rules": [{"name": "Upright", "params": "cup|15"}]}}) == {
        "MixedRules",
        "Upright",
    }


def test_frame_adapter_reads_shared_tensor_state_and_named_grippers():
    from types import SimpleNamespace

    import torch

    from roboverse_pack.tasks.icl.geniesim_checker import checker_frame_from_states

    robot_bodies = torch.zeros((1, 2, 13), dtype=torch.float32)
    robot_bodies[0, 0, :3] = torch.tensor([0.1, 0.2, 0.3])
    robot_bodies[0, 0, 3] = 1.0
    robot_bodies[0, 1, :3] = torch.tensor([0.4, 0.5, 0.6])
    robot_bodies[0, 1, 3] = 1.0
    states = SimpleNamespace(
        objects={
            "cube": SimpleNamespace(
                root_state=torch.tensor([[0.7, 0.8, 0.9, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32),
                joint_pos=None,
            )
        },
        robots={
            "agibot_g1": SimpleNamespace(
                root_state=torch.tensor([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32),
                body_names=["agibot_g1/gripper_l_center_link", "agibot_g1/gripper_r_center_link"],
                body_state=robot_bodies,
            )
        },
    )
    frame = checker_frame_from_states(
        states,
        robot_name="agibot_g1",
        step=4,
        object_bounds={"cube": np.array([[-0.05] * 3, [0.05] * 3])},
    )
    assert frame.step == 4
    np.testing.assert_allclose(frame.objects["cube"].pose[:3, 3], [0.7, 0.8, 0.9])
    np.testing.assert_allclose(frame.grippers["left_gripper"][:3, 3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(frame.grippers["right_gripper"][:3, 3], [0.4, 0.5, 0.6])


def test_source_problem_selection_is_numeric_and_matches_instruction():
    instance = {
        "task": "pick",
        "instance_id": "0",
        "instructions": {"instructions": [{"instruction": "red"}, {"instruction": "blue"}]},
        "checker_graphs": {
            "problem2": {"Acts": [{"Upright": "blue|15"}]},
            "problem1": {"Acts": [{"Upright": "red|15"}]},
        },
    }
    assert available_problem_ids(instance) == ["problem1", "problem2"]
    selected = GenieSimChecker.from_instance(instance, problem_index=1)
    assert selected.problem_id == "problem2" and selected.instruction == "blue"
    with pytest.raises(IndexError):
        GenieSimChecker.from_instance(instance, problem_index=2)
