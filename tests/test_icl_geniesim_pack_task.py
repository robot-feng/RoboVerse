"""Synthetic checker cases for GenieSim's grasp, lift, and placement operators.

These tests validate checker semantics with synthetic state frames; they do
not simulate grasp contact, container geometry, or successful physics.
"""

import numpy as np

from roboverse_pack.tasks.icl.geniesim_checker import (
    CheckerFrame,
    CheckerStatus,
    GenieSimChecker,
    ObjectObservation,
)

APPLE = "synthetic_apple"
BAG = "synthetic_container"


def _pose(position):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = position
    return pose


def _frame(step, *, apple_position, right_position, bag_position=(1.0, 0.0, 0.0), left_position=(2, 2, 2)):
    return CheckerFrame(
        step=step,
        objects={
            APPLE: ObjectObservation(
                _pose(apple_position), bounds=np.asarray([[-0.04, -0.045, -0.04], [0.04, 0.045, 0.04]])
            ),
            BAG: ObjectObservation(_pose(bag_position), bounds=np.asarray([[-0.2, -0.16, -0.1], [0.2, 0.16, 0.1]])),
        },
        grippers={"right_gripper": _pose(right_position), "left_gripper": _pose(left_position)},
    )


def _case():
    instruction = {"instruction": "Pack the synthetic objects"}
    instance = {
        "task": "synthetic_pack",
        "instance_id": "0",
        "instructions": {"instructions": [instruction]},
        "checker_graphs": {
            "problem1": {
                "Acts": [
                    {
                        "ActionList": [
                            {
                                "ActionSetWaitAny": [
                                    {"Follow": f"{APPLE}|[0.2,0.2,0.2]|right_gripper"}, {"StepOut": 200}
                                ]
                            },
                            {"ActionSetWaitAny": [{"PickUpOnGripper": f"{APPLE}|right_gripper"}, {"StepOut": 200}]},
                            {"ActionSetWaitAny": [{"Inside": f"{APPLE}|{BAG}|1.0"}, {"StepOut": 400}]},
                        ]
                    }
                ],
                "Init": [],
                "Objects": [],
                "Problem": "synthetic_pack",
            }
        },
    }
    return {
        "task_key": {"task_id": "synthetic_pack", "scene_id": "synthetic_scene", "embodiment": "G1_omnipicker"},
        "scene_seed": 1,
        "target_object_id": APPLE,
        "container_object_id": BAG,
        "instance": instance,
    }


def test_synthetic_pack_graph_requires_right_grasp_lift_then_container_placement():
    case = _case()
    task_key = case["task_key"]
    instruction = case["instance"]["instructions"]["instructions"][0]["instruction"]
    checker = GenieSimChecker.from_instance(case["instance"], problem_index=0)

    assert task_key["task_id"] == "synthetic_pack"
    assert task_key["scene_id"] == "synthetic_scene"
    assert task_key["embodiment"] == "G1_omnipicker"
    assert case["scene_seed"] == 1
    assert case["target_object_id"] == APPLE
    assert case["container_object_id"] == BAG
    assert instruction == "Pack the synthetic objects"
    assert checker.problem_id == "problem1"
    assert checker.instruction == instruction

    # Right-hand proximity, same-object pickup, then the apple center inside
    # the container bounds for two consecutive frames.
    apple_start = (0.0, 0.0, 0.0)
    assert (
        checker.step(_frame(0, apple_position=apple_start, right_position=apple_start)).status == CheckerStatus.RUNNING
    )
    lifted = (0.0, 0.0, 0.021)
    assert checker.step(_frame(1, apple_position=lifted, right_position=lifted)).status == CheckerStatus.RUNNING
    packed = (1.0, 0.0, 0.0)
    assert checker.step(_frame(2, apple_position=packed, right_position=packed)).status == CheckerStatus.RUNNING
    assert checker.step(_frame(3, apple_position=packed, right_position=packed)).status == CheckerStatus.SUCCESS


def test_synthetic_pack_graph_rejects_left_hand_pickup_and_wrong_container():
    case = _case()
    apple_start = (0.0, 0.0, 0.0)

    wrong_hand = GenieSimChecker.from_instance(case["instance"], problem_index=0)
    assert (
        wrong_hand.step(
            _frame(0, apple_position=apple_start, right_position=(2, 2, 2), left_position=apple_start)
        ).status
        == CheckerStatus.RUNNING
    )
    assert (
        wrong_hand.step(
            _frame(1, apple_position=(0.0, 0.0, 0.021), right_position=(2, 2, 2), left_position=(0.0, 0.0, 0.021))
        ).status
        == CheckerStatus.RUNNING
    )

    wrong_container = GenieSimChecker.from_instance(case["instance"], problem_index=0)
    assert (
        wrong_container.step(_frame(0, apple_position=apple_start, right_position=apple_start)).status
        == CheckerStatus.RUNNING
    )
    lifted = (0.0, 0.0, 0.021)
    assert (
        wrong_container.step(_frame(1, apple_position=lifted, right_position=lifted)).status == CheckerStatus.RUNNING
    )
    outside_bag = (0.5, 0.0, 0.0)
    for step in range(2, 5):
        result = wrong_container.step(_frame(step, apple_position=outside_bag, right_position=outside_bag))
    assert result.status == CheckerStatus.RUNNING


def test_synthetic_pack_graph_times_out_when_right_hand_never_reaches_apple():
    case = _case()
    checker = GenieSimChecker.from_instance(case["instance"], problem_index=0)
    apple_start = (0.0, 0.0, 0.0)
    assert (
        checker.step(_frame(0, apple_position=apple_start, right_position=(2.0, 2.0, 2.0))).status
        == CheckerStatus.RUNNING
    )
    result = checker.step(_frame(201, apple_position=apple_start, right_position=(2.0, 2.0, 2.0)))
    assert result.status == CheckerStatus.FAILURE
