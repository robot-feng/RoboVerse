"""Genie Sim evaluates task checkers at a slower cadence than control."""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from metasim.task.base import BaseTaskEnv
from roboverse_pack.tasks.icl.geniesim_benchmark import GenieSimBenchmarkTask
from roboverse_pack.tasks.icl.geniesim_checker import (
    CheckerFrame,
    CheckerResult,
    CheckerStatus,
    GenieSimChecker,
    ObjectObservation,
)


def _pose(x: float) -> np.ndarray:
    result = np.eye(4)
    result[0, 3] = x
    return result


@pytest.mark.general
def test_source_stepout_uses_action_steps_between_checker_samples():
    checker = GenieSimChecker.from_subtask_steps([{"checker": {"StepOut": 200}}])
    for step in range(0, 210, 30):
        assert checker.step(CheckerFrame(step=step, objects={})).status == CheckerStatus.RUNNING
    assert checker.step(CheckerFrame(step=210, objects={})).status == CheckerStatus.FAILURE


@pytest.mark.general
def test_source_stable_grasp_can_observe_motion_across_30_actions():
    checker = GenieSimChecker.from_subtask_steps([
        {"checker": {"StableGrasp": "chip|right_gripper|0.1|0.02|0.1"}}
    ])
    for step, x, expected in ((0, 0.0, CheckerStatus.RUNNING), (30, 0.06, CheckerStatus.RUNNING),
                              (60, 0.12, CheckerStatus.SUCCESS)):
        frame = CheckerFrame(
            step=step,
            objects={"chip": ObjectObservation(_pose(x))},
            grippers={"right_gripper": _pose(x)},
        )
        assert checker.step(frame).status == expected


@pytest.mark.general
def test_task_samples_checker_at_reset_and_before_each_30th_action(monkeypatch):
    task = object.__new__(GenieSimBenchmarkTask)
    task._episode_steps = torch.zeros(1, dtype=torch.int32)
    task._checker_step = 0
    task._checker_sim_step = 0
    task._delta_time = 1 / 30
    task.reset_settle_seconds = 0.0
    task._internal_target_clipping = {}
    task.run_id = "test"
    task.run = {}
    task.source_robot = "G1"
    task.robot_type = "G1"
    task.action_mode = "joint_position"
    task.device = torch.device("cpu")
    task._checker_result = CheckerResult(CheckerStatus.RUNNING, "problem1", "test")
    task.handler = SimpleNamespace(
        num_envs=1,
        get_states=lambda mode: f"after-{int(task._episode_steps[0])}-actions",
    )
    samples = []

    def observe(self, state, *, action_step, completed_sim_steps, delta_time):
        samples.append((action_step, completed_sim_steps, state, delta_time))
        self._checker_step = action_step
        self._checker_sim_step = completed_sim_steps

    task._observe_checker = MethodType(observe, task)

    def reset_base(self, states=None, env_ids=None, seed=None):
        self._episode_steps.zero_()
        return "reset-state", {}

    def step_base(self, action):
        self._episode_steps += 1
        return None, None, None, None, {}

    monkeypatch.setattr(BaseTaskEnv, "reset", reset_base)
    monkeypatch.setattr(BaseTaskEnv, "step", step_base)
    task.reset()
    for _ in range(61):
        task.step(None)
    assert samples == [(0, 0, "reset-state", 0.0), (30, 29, "after-29-actions", 29 / 30),
                       (60, 59, "after-59-actions", 1.0)]


@pytest.mark.examples
def test_g1_benchmark_checker_cadence_on_mujoco():
    """The actual task samples its source checker around the 30th action."""
    from roboverse_pack.tasks.icl.geniesim_benchmark import (
        ASSETS_ROOT,
        CACHE_ROOT,
        ROBOT_ROOT,
        SPEC_PATH,
        build_geniesim_g1_robot,
    )
    from roboverse_pack.tasks.icl.geniesim_scene import build_run_scenario

    if not SPEC_PATH.is_file() or not (ROBOT_ROOT / "provenance.json").is_file():
        pytest.skip("Prepare G1 and the source-scene benchmark snapshot first")
    scenario, _, _, _ = build_run_scenario(
        spec_path=SPEC_PATH,
        run_id="g1op_s2r_bimanual_chip_handover",
        simulator="mujoco",
        robot=build_geniesim_g1_robot(ROBOT_ROOT),
        assets_root=ASSETS_ROOT,
        cache_root=CACHE_ROOT,
        cameras=(),
    )
    task = GenieSimBenchmarkTask(run_id="g1op_s2r_bimanual_chip_handover", simulator="mujoco", scenario=scenario)
    try:
        task.reset()
        assert task.checker._last_step == 0
        controls = [scenario.robots[0].default_joint_positions[name] for name in task.control_joint_names]
        for _ in range(29):
            task.step(controls)
        assert task.checker._last_step == 0
        task.step(controls)
        assert task.checker._last_step == 30
    finally:
        task.close()
