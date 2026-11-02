"""Shared bounded joint-position action contract for G1 ICL tasks."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from metasim.task.base import BaseTaskEnv

CONTROL_JOINTS = (
    "idx01_body_joint1",
    "idx02_body_joint2",
    "idx11_head_joint1",
    "idx12_head_joint2",
    *(f"idx{20 + i:02d}_arm_l_joint{i}" for i in range(1, 8)),
    *(f"idx{60 + i:02d}_arm_r_joint{i}" for i in range(1, 8)),
    "idx41_gripper_l_outer_joint1",
    "idx81_gripper_r_outer_joint1",
)


class G1JointPositionTask(BaseTaskEnv):
    """Base lifecycle and joint-vector controls shared by fixed G1 ICL tasks."""

    supported_simulators = ("mujoco", "genesis", "newton")

    def __init__(self, scenario, device=None):
        super().__init__(scenario, device)
        self._inside_frames = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._failure = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.robot_name = self.scenario.robots[0].name
        names = [name.split("/")[-1] for name in self.handler.get_joint_names(self.robot_name, sort=True)]
        missing = set(CONTROL_JOINTS) - set(names)
        if missing:
            self.close()
            raise ValueError(f"Backend is missing required G1 control joints: {sorted(missing)}")
        self._control_indices = [names.index(name) for name in CONTROL_JOINTS]
        self.post_physics_step_callback.append(self._observe_outcome)
        self.reset_callback.append(self._reset_outcome)

    def _get_initial_states(self) -> list[dict]:
        robot = self.scenario.robots[0]
        return [
            {
                "objects": {
                    obj.name: {"pos": torch.tensor(obj.default_position), "rot": torch.tensor(obj.default_orientation)}
                    for obj in self.scenario.objects
                },
                "robots": {
                    robot.name: {
                        "pos": torch.tensor(robot.default_position),
                        "rot": torch.tensor(robot.default_orientation),
                        "dof_pos": dict(robot.default_joint_positions),
                    }
                },
            }
        ]

    def _reset_outcome(self, env_ids: list[int]) -> None:
        self._inside_frames[env_ids] = 0
        self._success[env_ids] = False
        self._failure[env_ids] = False

    def _observe_outcome(self, state) -> None:
        raise NotImplementedError("Each G1 ICL task must define its own task predicate")

    def _terminated(self, states) -> torch.Tensor:
        return self._success

    def control_state(self, states) -> torch.Tensor:
        """Return the 20 named control joints in the shared action order."""
        return states.robots[self.robot_name].joint_pos[:, self._control_indices]

    def _action_space(self) -> gym.spaces.Box:
        """Use finite joint-position limits as the bounded task action contract."""
        limits = self.scenario.robots[0].joint_limits
        low = np.asarray([limits[name][0] for name in CONTROL_JOINTS], dtype=np.float32)
        high = np.asarray([limits[name][1] for name in CONTROL_JOINTS], dtype=np.float32)
        if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high <= low):
            raise ValueError("G1 joint-position control requires finite, increasing limits for every controlled joint")
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _process_action(self, actions):
        """Map bounded joint vectors to the named position targets used by MetaSim."""
        if (
            isinstance(actions, (list, tuple))
            and len(actions) == self.num_envs
            and all(isinstance(action, dict) for action in actions)
        ):
            return actions

        command = torch.as_tensor(actions, dtype=torch.float64, device="cpu")
        if command.shape == (len(CONTROL_JOINTS),) and self.num_envs == 1:
            command = command.unsqueeze(0)
        if command.shape != (self.num_envs, len(CONTROL_JOINTS)) or not torch.isfinite(command).all():
            raise ValueError(f"Expected finite joint targets with shape ({self.num_envs}, {len(CONTROL_JOINTS)})")
        bounds = self._action_space()
        lower = torch.as_tensor(bounds.low, dtype=command.dtype)
        upper = torch.as_tensor(bounds.high, dtype=command.dtype)
        if torch.any(command < lower) or torch.any(command > upper):
            raise ValueError("Joint-position action contains a target outside the robot's limits")
        return [
            {self.robot_name: {"dof_pos_target": dict(zip(CONTROL_JOINTS, row.tolist(), strict=True))}}
            for row in command
        ]

    def position_action(self, values) -> list[dict]:
        """Build a validated single-environment named position command."""
        command = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
        if command.shape != (len(CONTROL_JOINTS),) or not torch.isfinite(command).all():
            raise ValueError(f"Expected {len(CONTROL_JOINTS)} finite position targets")
        limits = self.scenario.robots[0].joint_limits
        for name, value in zip(CONTROL_JOINTS, command.tolist(), strict=True):
            if not limits[name][0] <= value <= limits[name][1]:
                raise ValueError(f"Position target for {name} is outside its limits")
        return [{self.robot_name: {"dof_pos_target": dict(zip(CONTROL_JOINTS, command.tolist(), strict=True))}}]

    def step(self, actions):
        """Advance the task and report failure separately from success."""
        obs, reward, success, timeout, info = super().step(actions)
        info["task_failure"] = self._failure.clone()
        return obs, reward, success, timeout | self._failure, info
