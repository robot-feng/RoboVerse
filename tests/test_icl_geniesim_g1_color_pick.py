"""Synthetic semantic check for GenieSim's color-pick checker operators."""

import numpy as np

from roboverse_pack.tasks.icl.geniesim_checker import (
    CheckerFrame,
    CheckerStatus,
    GenieSimChecker,
    ObjectObservation,
)

RED_BLOCK = "synthetic_red_block"
YELLOW_BLOCK = "synthetic_yellow_block"


def _pose(position):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = position
    return pose


def _frame(step, *, red_position, right_position, yellow_position=(0.0, 0.0, 0.9), left_position=(1, 1, 1)):
    return CheckerFrame(
        step=step,
        objects={
            RED_BLOCK: ObjectObservation(_pose(red_position)),
            YELLOW_BLOCK: ObjectObservation(_pose(yellow_position)),
        },
        grippers={
            "right_gripper": _pose(right_position),
            "left_gripper": _pose(left_position),
        },
    )


def _case():
    instruction = {
        "id": 1,
        "instruction": "Right arm picks up the red block",
        "target": {"id1": RED_BLOCK},
        "gripper": {"1": {"handover": False, "side": "right"}},
    }
    instance = {
        "task": "synthetic_pick_color",
        "instance_id": "0",
        "instructions": {"instructions": [instruction]},
        "checker_graphs": {
            "problem1": {
                "Acts": [
                    {
                        "ActionList": [
                            {
                                "ActionSetWaitAny": [
                                    {"Follow": f"{RED_BLOCK}|[0.1,0.2,0.1]|right_gripper"}, {"StepOut": 200}
                                ]
                            },
                            {
                                "ActionSetWaitAny": [
                                    {"PickUpOnGripper": f"{RED_BLOCK}|right_gripper"}, {"StepOut": 200}
                                ]
                            },
                        ]
                    }
                ],
                "Init": [],
                "Objects": [],
                "Problem": "synthetic_pick_color",
            }
        },
    }
    return {
        "task_key": {"task_id": "synthetic_pick_color"},
        "scene_seed": 1,
        "target_object_id": RED_BLOCK,
        "instance": instance,
    }


def test_synthetic_color_pick_selects_red_and_right_arm():
    case = _case()
    task_key = case["task_key"]
    instruction = case["instance"]["instructions"]["instructions"][0]
    checker = GenieSimChecker.from_instance(case["instance"], problem_index=0)

    assert task_key == {"task_id": "synthetic_pick_color"}
    assert case["scene_seed"] == 1
    assert instruction["id"] == 1
    assert instruction["instruction"] == "Right arm picks up the red block"
    assert instruction["target"]["id1"] == case["target_object_id"] == RED_BLOCK
    assert instruction["gripper"]["1"]["side"] == "right"
    assert checker.problem_id == "problem1"
    assert checker.instruction == instruction["instruction"]

    # The right gripper first enters the target's Follow box,
    # then that same target must rise at least 2 cm beside the right gripper.
    start = (0.044, -0.05, 0.885)
    assert checker.step(_frame(0, red_position=start, right_position=start)).status == CheckerStatus.RUNNING
    lifted_red = (start[0], start[1], start[2] + 0.021)
    assert checker.step(_frame(1, red_position=lifted_red, right_position=lifted_red)).status == CheckerStatus.SUCCESS


def test_synthetic_color_pick_rejects_wrong_block_or_wrong_arm():
    case = _case()

    wrong_object = GenieSimChecker.from_instance(case["instance"], problem_index=0)
    red_start = (0.044, -0.05, 0.885)
    assert (
        wrong_object.step(_frame(0, red_position=red_start, right_position=red_start)).status == CheckerStatus.RUNNING
    )
    assert wrong_object.step(
        _frame(
            1,
            red_position=red_start,
            right_position=red_start,
            yellow_position=(0.0, 0.0, 0.921),
        )
    ).status == CheckerStatus.RUNNING

    wrong_arm = GenieSimChecker.from_instance(case["instance"], problem_index=0)
    assert wrong_arm.step(
        _frame(0, red_position=red_start, right_position=(1, 1, 1), left_position=red_start)
    ).status == CheckerStatus.RUNNING
    assert wrong_arm.step(
        _frame(
            1,
            red_position=(red_start[0], red_start[1], red_start[2] + 0.021),
            right_position=(1, 1, 1),
            left_position=(red_start[0], red_start[1], red_start[2] + 0.021),
        )
    ).status == CheckerStatus.RUNNING
