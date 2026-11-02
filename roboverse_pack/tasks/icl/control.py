"""Cartesian delta control for the fixed-base Franka ICL task."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class GenieSimResetPose:
    """A source reset pose adjusted to the portable robot model's constraints."""

    joint_positions: dict[str, float]
    clipped_source_positions: dict[str, dict[str, float]]
    max_equality_residual: float


def prepare_geniesim_reset_pose(robot, source_positions: Mapping[str, float]) -> GenieSimResetPose:
    """Apply named source targets and solve passive gripper loop joints before reset.

    Source gripper commands beyond the prepared joint limits are clipped and
    recorded. Other out-of-range source joints and unsatisfied loop constraints
    raise an error instead of starting an invalid simulation state.
    """
    positions = dict(robot.default_joint_positions or {})
    limits = robot.joint_limits or {}
    gripper_leaders = {"idx41_gripper_l_outer_joint1", "idx81_gripper_r_outer_joint1"}
    missing = set(source_positions) - set(positions)
    if missing:
        raise ValueError(f"Source reset contains joints missing from the robot: {sorted(missing)}")
    clipped: dict[str, dict[str, float]] = {}
    for name, raw in source_positions.items():
        value = float(raw)
        if not np.isfinite(value) or name not in limits:
            raise ValueError(f"Source reset joint {name!r} has no finite value or limits")
        low, high = limits[name]
        if not low <= value <= high:
            if name not in gripper_leaders:
                raise ValueError(f"Source reset joint {name!r} is outside the prepared model's limits")
            adjusted = min(max(value, low), high)
            clipped[name] = {"source": value, "applied": float(adjusted)}
            value = adjusted
        positions[name] = value

    passive = [name for name in positions if "_gripper_" in name and name not in source_positions]
    if not passive:
        return GenieSimResetPose(positions, clipped, 0.0)

    model = mujoco.MjModel.from_xml_path(robot.mjcf_path)
    data = mujoco.MjData(model)
    joint_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in positions}
    absent = [name for name, joint_id in joint_ids.items() if joint_id < 0]
    if absent:
        raise ValueError(f"Prepared MJCF is missing reset joints: {sorted(absent)}")
    qpos = model.qpos0.copy()
    for name, value in positions.items():
        joint_id = joint_ids[name]
        if model.jnt_type[joint_id] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError(f"Reset joint {name!r} is not a scalar hinge or slide")
        qpos[model.jnt_qposadr[joint_id]] = value

    # Satisfy explicit mimic equalities exactly before optimizing ball-joint
    # closure. A controlled joint can appear on either side of the equation.
    known = set(source_positions)
    unresolved = set(passive)
    for _ in range(len(passive) + 1):
        changed = False
        for index in range(model.neq):
            if model.eq_type[index] != mujoco.mjtEq.mjEQ_JOINT or not model.eq_active0[index]:
                continue
            j1, j2 = int(model.eq_obj1id[index]), int(model.eq_obj2id[index])
            if min(j1, j2) < 0:
                continue
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j1)
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j2)
            coefficients = model.eq_data[index, :5]
            if n1 in unresolved and n2 in known:
                name, value = n1, np.polynomial.polynomial.polyval(qpos[model.jnt_qposadr[j2]], coefficients)
            elif n2 in unresolved and n1 in known and np.allclose(coefficients[2:], 0) and abs(coefficients[1]) > 1e-12:
                name = n2
                value = (qpos[model.jnt_qposadr[j1]] - coefficients[0]) / coefficients[1]
            else:
                continue
            low, high = limits[name]
            if value < low - 1e-8 or value > high + 1e-8:
                raise ValueError(f"Mimic reset for {name!r} is outside its joint limits")
            qpos[model.jnt_qposadr[joint_ids[name]]] = np.clip(value, low, high)
            known.add(name)
            unresolved.remove(name)
            changed = True
        if not changed:
            break

    free = [name for name in passive if name in unresolved]
    q_indices = np.asarray([model.jnt_qposadr[joint_ids[name]] for name in free], dtype=int)

    def residual(values: np.ndarray) -> np.ndarray:
        data.qpos[:] = qpos
        data.qpos[q_indices] = values
        mujoco.mj_forward(model, data)
        return data.efc_pos[: data.ne].copy()

    if free:
        lower = np.asarray([limits[name][0] for name in free], dtype=float)
        upper = np.asarray([limits[name][1] for name in free], dtype=float)
        if np.any(lower >= upper):
            raise ValueError("Passive gripper joints require nonzero position ranges")
        start = np.clip(qpos[q_indices], lower, upper)
        solution = least_squares(residual, start, bounds=(lower, upper), ftol=1e-12, xtol=1e-12, gtol=1e-12)
        if not solution.success:
            raise RuntimeError(f"Could not solve the gripper reset constraints: {solution.message}")
        qpos[q_indices] = solution.x
    errors = residual(qpos[q_indices])
    max_error = float(np.max(np.abs(errors), initial=0.0))
    if not np.isfinite(max_error) or max_error > 1e-4:
        raise RuntimeError(f"Gripper reset constraints remain inconsistent (maximum residual {max_error:.6g})")
    for name in passive:
        positions[name] = float(qpos[model.jnt_qposadr[joint_ids[name]]])
    return GenieSimResetPose(positions, clipped, max_error)


class FrankaCartesianController:
    """Convert world-frame TCP deltas into position targets using Franka kinematics.

    The kinematic model is independent of the simulation backend. The task
    fixes the robot base at the world origin; inputs must use the handler's
    sorted joint order. Gripper 0 means closed and 1 means open.
    """

    def __init__(self, robot, joint_names: list[str]):
        self.robot = robot
        self.joint_names = [name.split("/")[-1] for name in joint_names]
        self.arm_names = [f"panda_joint{i}" for i in range(1, 8)]
        if len(set(self.joint_names)) != len(self.joint_names) or not set(self.arm_names).issubset(self.joint_names):
            raise ValueError(f"Franka arm joints missing or ambiguous: {joint_names}")
        self.model = mujoco.MjModel.from_xml_path(robot.mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, robot.ee_body_name)
        if self.body_id < 0:
            raise ValueError(f"End-effector body {robot.ee_body_name!r} is missing from {robot.mjcf_path}")
        ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.arm_names]
        if min(ids) < 0:
            raise ValueError("Franka MJCF does not contain all seven arm joints")
        self.qpos_indices = np.array([self.model.jnt_qposadr[i] for i in ids], dtype=int)
        self.dof_indices = np.array([self.model.jnt_dofadr[i] for i in ids], dtype=int)
        self.input_indices = np.array([self.joint_names.index(name) for name in self.arm_names], dtype=int)
        self.limits = np.array([robot.joint_limits[name] for name in self.arm_names], dtype=float)
        self.tcp_offset = np.asarray(robot.curobo_tcp_rel_pos, dtype=float)

    def _forward(self, q: np.ndarray) -> tuple[np.ndarray, Rotation]:
        self.data.qpos[self.qpos_indices] = q
        mujoco.mj_forward(self.model, self.data)
        orientation = Rotation.from_matrix(self.data.xmat[self.body_id].reshape(3, 3))
        position = self.data.xpos[self.body_id].copy() + orientation.apply(self.tcp_offset)
        return position, orientation

    def tcp_pose(self, joint_pos) -> tuple[np.ndarray, Rotation]:
        """Return the measured joint configuration's TCP pose in world coordinates."""
        q = np.asarray(joint_pos, dtype=float)
        if q.shape != (len(self.joint_names),) or not np.isfinite(q).all():
            raise ValueError("Expected one finite value for each sorted robot joint")
        return self._forward(q[self.input_indices])

    def joint_targets(self, joint_pos, action) -> dict[str, float]:
        """Solve a 7D [translation delta, rotation-vector delta, gripper] command."""
        q_all = np.asarray(joint_pos, dtype=float)
        action = np.asarray(action, dtype=float)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Cartesian action must contain seven finite values")
        if np.linalg.norm(action[:3]) > 0.021 or np.linalg.norm(action[3:6]) > 0.21:
            raise ValueError("Cartesian delta exceeds the task's 0.021 m / 0.21 rad per-step limits")
        if not 0.0 <= action[6] <= 1.0:
            raise ValueError("Gripper command must be between 0 (closed) and 1 (open)")
        current_pos, current_rot = self.tcp_pose(q_all)
        target_pos = current_pos + action[:3]
        target_rot = Rotation.from_rotvec(action[3:6]) * current_rot
        q = q_all[self.input_indices].copy()
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        for _ in range(40):
            position, orientation = self._forward(q)
            error = np.concatenate((target_pos - position, (target_rot * orientation.inv()).as_rotvec()))
            if np.linalg.norm(error[:3]) < 0.001 and np.linalg.norm(error[3:]) < 0.02:
                break
            mujoco.mj_jac(self.model, self.data, jacp, jacr, position, self.body_id)
            jac = np.vstack((jacp[:, self.dof_indices], jacr[:, self.dof_indices]))
            delta = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * 0.0025, error)
            q = np.clip(q + np.clip(delta, -0.15, 0.15), self.limits[:, 0], self.limits[:, 1])
        else:
            raise ValueError(f"Cartesian target is unreachable: position error={np.linalg.norm(error[:3]):.4f} m")
        result = {name: float(value) for name, value in zip(self.arm_names, q, strict=True)}
        for name in self.robot.ee_joint_names:
            low, high = self.robot.joint_limits[name]
            result[name] = float(low + action[6] * (high - low))
        return result


class G1CartesianController:
    """Solve one-arm Cartesian deltas against the prepared G1 kinematic model.

    Gripper values map linearly from 0 (closed) to 1 (open) over the current
    simulation range. This preliminary mapping is not a real-hardware calibration.
    """

    POSITION_TOLERANCE_M = 0.001
    ORIENTATION_TOLERANCE_RAD = 0.02

    def __init__(
        self,
        robot,
        control_names: tuple[str, ...],
        arm_side: str = "right",
        allow_torso_motion: bool = False,
    ):
        if arm_side not in {"left", "right"}:
            raise ValueError("G1 Cartesian arm_side must be 'left' or 'right'")
        self.robot = robot
        self.control_names = tuple(control_names)
        self.arm_side = arm_side
        arm_letter, offset = ("l", 20) if arm_side == "left" else ("r", 60)
        self.arm_names = tuple(f"idx{offset + i:02d}_arm_{arm_letter}_joint{i}" for i in range(1, 8))
        self.gripper_name = f"idx{'41' if arm_side == 'left' else '81'}_gripper_{arm_letter}_outer_joint1"
        if len(set(self.control_names)) != len(self.control_names) or not set((
            *self.arm_names,
            self.gripper_name,
        )).issubset(self.control_names):
            raise ValueError(f"G1 control names are missing the {arm_side} arm or gripper")
        self.model = mujoco.MjModel.from_xml_path(robot.mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"gripper_{arm_letter}_center_link")
        if self.body_id < 0:
            raise ValueError(f"G1 {arm_side} gripper center link is missing")
        joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.control_names]
        if min(joint_ids) < 0:
            raise ValueError("G1 MJCF is missing a control joint")
        self.qpos_indices = np.array([self.model.jnt_qposadr[index] for index in joint_ids], dtype=int)
        self.arm_input_indices = np.array([self.control_names.index(name) for name in self.arm_names], dtype=int)
        self.arm_dof_indices = np.array(
            [self.model.jnt_dofadr[joint_ids[index]] for index in self.arm_input_indices], dtype=int
        )
        self.arm_limits = np.asarray([robot.joint_limits[name] for name in self.arm_names], dtype=float)
        torso_names = ("idx01_body_joint1", "idx02_body_joint2") if allow_torso_motion else ()
        self.ik_names = (*torso_names, *self.arm_names)
        self.ik_input_indices = np.asarray([self.control_names.index(name) for name in self.ik_names], dtype=int)
        self.ik_dof_indices = np.asarray([self.model.jnt_dofadr[joint_ids[index]] for index in self.ik_input_indices])
        self.ik_limits = np.asarray([robot.joint_limits[name] for name in self.ik_names], dtype=float)
        self.allow_torso_motion = allow_torso_motion
        self.base_position = np.asarray(robot.default_position, dtype=float)
        self.base_rotation = Rotation.from_quat(robot.default_orientation, scalar_first=True)

    def _forward(self, controls: np.ndarray) -> tuple[np.ndarray, Rotation]:
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[self.qpos_indices] = controls
        mujoco.mj_forward(self.model, self.data)
        local_pos = self.data.xpos[self.body_id].copy()
        local_rot = Rotation.from_matrix(self.data.xmat[self.body_id].reshape(3, 3))
        return self.base_position + self.base_rotation.apply(local_pos), self.base_rotation * local_rot

    def tcp_pose(self, controls) -> tuple[np.ndarray, Rotation]:
        """Return the selected gripper center pose in world coordinates."""
        values = np.asarray(controls, dtype=float)
        if values.shape != (len(self.control_names),) or not np.isfinite(values).all():
            raise ValueError("Expected one finite value for each G1 control joint")
        return self._forward(values)

    def joint_targets(self, controls, action) -> np.ndarray:
        """Return 20 position targets for a 7D world-frame delta and gripper command."""
        values = np.asarray(controls, dtype=float)
        command = np.asarray(action, dtype=float)
        if command.shape != (7,) or not np.isfinite(command).all():
            raise ValueError("Cartesian action must contain seven finite values")
        if np.linalg.norm(command[:3]) > 0.021 or np.linalg.norm(command[3:6]) > 0.21:
            raise ValueError("Cartesian delta exceeds 0.021 m / 0.21 rad per step")
        if not 0 <= command[6] <= 1:
            raise ValueError("Gripper command must be between 0 and 1")
        current_pos, current_rot = self.tcp_pose(values)
        target_pos = current_pos + command[:3]
        target_rot = Rotation.from_rotvec(command[3:6]) * current_rot
        result = values.copy()
        start_ik = np.clip(
            values[self.ik_input_indices], self.ik_limits[:, 0] + 1e-10, self.ik_limits[:, 1] - 1e-10
        )
        result[self.ik_input_indices] = start_ik
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        converged = False
        for _ in range(40):
            position, orientation = self._forward(result)
            error = np.r_[target_pos - position, (target_rot * orientation.inv()).as_rotvec()]
            if (
                np.linalg.norm(error[:3]) < self.POSITION_TOLERANCE_M
                and np.linalg.norm(error[3:]) < self.ORIENTATION_TOLERANCE_RAD
            ):
                converged = True
                break
            mujoco.mj_jac(self.model, self.data, jacp, jacr, self.data.xpos[self.body_id], self.body_id)
            rotation = self.base_rotation.as_matrix()
            jacobian = np.vstack((rotation @ jacp[:, self.ik_dof_indices], rotation @ jacr[:, self.ik_dof_indices]))
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + np.eye(6) * 0.0025, error)
            result[self.ik_input_indices] = np.clip(
                result[self.ik_input_indices] + np.clip(delta, -0.15, 0.15),
                self.ik_limits[:, 0],
                self.ik_limits[:, 1],
            )
        position, orientation = self._forward(result)
        error = np.r_[target_pos - position, (target_rot * orientation.inv()).as_rotvec()]
        ik_delta = result[self.ik_input_indices] - start_ik
        # A converged pose can still be an IK branch flip. Keep the fast DLS
        # result for ordinary steps and refine stalled or discontinuous ones
        # near the measured joint configuration.
        needs_refinement = (
            not converged
            or np.linalg.norm(error[:3]) >= self.POSITION_TOLERANCE_M
            or np.linalg.norm(error[3:]) >= self.ORIENTATION_TOLERANCE_RAD
            or np.max(np.abs(ik_delta), initial=0.0) > 0.5
        )
        if needs_refinement:
            def pose_residual(ik_positions: np.ndarray) -> np.ndarray:
                candidate = values.copy()
                candidate[self.ik_input_indices] = ik_positions
                candidate_position, candidate_rotation = self._forward(candidate)
                orientation_error = (target_rot * candidate_rotation.inv()).as_rotvec()
                continuity_error = 0.002 * (ik_positions - start_ik)
                return np.r_[target_pos - candidate_position, 0.1 * orientation_error, continuity_error]

            solution = least_squares(
                pose_residual,
                start_ik,
                bounds=(self.ik_limits[:, 0], self.ik_limits[:, 1]),
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
                max_nfev=200,
            )
            result[self.ik_input_indices] = solution.x
            position, orientation = self._forward(result)
            error = np.r_[target_pos - position, (target_rot * orientation.inv()).as_rotvec()]
            if (
                np.linalg.norm(error[:3]) >= self.POSITION_TOLERANCE_M
                or np.linalg.norm(error[3:]) >= self.ORIENTATION_TOLERANCE_RAD
            ):
                raise ValueError(
                    "G1 Cartesian target is unreachable: "
                    f"position error={np.linalg.norm(error[:3]):.4f} m, "
                    f"orientation error={np.linalg.norm(error[3:]):.4f} rad"
                )
        low, high = self.robot.joint_limits[self.gripper_name]
        result[self.control_names.index(self.gripper_name)] = low + command[6] * (high - low)
        return result


class G2CartesianController:
    """Solve one-arm Cartesian deltas against the prepared G2 Omnipicker MJCF."""

    def __init__(self, robot, control_names: tuple[str, ...], arm_side: str = "right"):
        if arm_side not in {"left", "right"}:
            raise ValueError("G2 Cartesian arm_side must be 'left' or 'right'")
        self.robot = robot
        self.control_names = tuple(control_names)
        self.arm_side = arm_side
        side_offset = 20 if arm_side == "left" else 60
        self.arm_names = tuple(
            f"idx{side_offset + index:02d}_arm_{'l' if arm_side == 'left' else 'r'}_joint{index}"
            for index in range(1, 8)
        )
        self.gripper_name = "idx41_gripper_l_outer_joint1" if arm_side == "left" else "idx81_gripper_r_outer_joint1"
        if not set((*self.arm_names, self.gripper_name)).issubset(self.control_names):
            raise ValueError(f"G2 {arm_side} arm or gripper is missing from the control vector")
        self.model = mujoco.MjModel.from_xml_path(robot.mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, f"gripper_{'l' if arm_side == 'left' else 'r'}_base_link"
        )
        if self.body_id < 0:
            raise ValueError(f"G2 {arm_side} gripper base link is missing")
        control_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.control_names]
        arm_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.arm_names]
        if min(control_ids) < 0 or min(arm_ids) < 0:
            raise ValueError("G2 MJCF is missing one or more controlled joints")
        self.qpos_indices = np.asarray([self.model.jnt_qposadr[index] for index in control_ids], dtype=int)
        self.arm_input_indices = np.asarray([self.control_names.index(name) for name in self.arm_names], dtype=int)
        self.arm_dof_indices = np.asarray([self.model.jnt_dofadr[index] for index in arm_ids], dtype=int)
        self.arm_limits = np.asarray([robot.joint_limits[name] for name in self.arm_names], dtype=float)
        self.base_position = np.asarray(robot.default_position, dtype=float)
        self.base_rotation = Rotation.from_quat(robot.default_orientation, scalar_first=True)
        self.tcp_offset = np.asarray([0.0, 0.0, 0.14308], dtype=float)
        self.tcp_rotation = Rotation.from_euler("z", -np.pi / 2)

    def _forward(self, controls: np.ndarray) -> tuple[np.ndarray, Rotation]:
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[self.qpos_indices] = controls
        mujoco.mj_forward(self.model, self.data)
        link_rotation = Rotation.from_matrix(self.data.xmat[self.body_id].reshape(3, 3))
        local_position = self.data.xpos[self.body_id].copy() + link_rotation.apply(self.tcp_offset)
        local_rotation = link_rotation * self.tcp_rotation
        return self.base_position + self.base_rotation.apply(local_position), self.base_rotation * local_rotation

    def tcp_pose(self, controls) -> tuple[np.ndarray, Rotation]:
        """Return the selected G2 gripper center pose in world coordinates."""
        values = np.asarray(controls, dtype=float)
        if values.shape != (len(self.control_names),) or not np.isfinite(values).all():
            raise ValueError("Expected one finite value for each G2 control joint")
        return self._forward(values)

    def joint_targets(self, controls, action) -> np.ndarray:
        """Return the G2 control vector for a 7D TCP delta and gripper command."""
        values = np.asarray(controls, dtype=float)
        command = np.asarray(action, dtype=float)
        if command.shape != (7,) or not np.isfinite(command).all():
            raise ValueError("Cartesian action must contain seven finite values")
        if np.linalg.norm(command[:3]) > 0.021 or np.linalg.norm(command[3:6]) > 0.21:
            raise ValueError("Cartesian delta exceeds 0.021 m / 0.21 rad per step")
        if not 0.0 <= command[6] <= 1.0:
            raise ValueError("Gripper command must be between 0 (closed) and 1 (open)")
        target_position, target_rotation = self.tcp_pose(values)
        target_position += command[:3]
        target_rotation = Rotation.from_rotvec(command[3:6]) * target_rotation
        result = values.copy()
        jacobian_position = np.zeros((3, self.model.nv))
        jacobian_rotation = np.zeros((3, self.model.nv))
        for _ in range(40):
            position, orientation = self._forward(result)
            error = np.concatenate((target_position - position, (target_rotation * orientation.inv()).as_rotvec()))
            if np.linalg.norm(error[:3]) < 0.001 and np.linalg.norm(error[3:]) < 0.02:
                break
            mujoco.mj_jac(
                self.model,
                self.data,
                jacobian_position,
                jacobian_rotation,
                self.data.xpos[self.body_id] + self.data.xmat[self.body_id].reshape(3, 3) @ self.tcp_offset,
                self.body_id,
            )
            rotation = self.base_rotation.as_matrix()
            jacobian = np.vstack((
                rotation @ jacobian_position[:, self.arm_dof_indices],
                rotation @ jacobian_rotation[:, self.arm_dof_indices],
            ))
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + np.eye(6) * 0.0025, error)
            result[self.arm_input_indices] = np.clip(
                result[self.arm_input_indices] + np.clip(delta, -0.15, 0.15),
                self.arm_limits[:, 0],
                self.arm_limits[:, 1],
            )
        else:
            raise ValueError(f"G2 Cartesian target is unreachable: position error={np.linalg.norm(error[:3]):.4f} m")
        gripper_index = self.control_names.index(self.gripper_name)
        low, high = self.robot.joint_limits[self.gripper_name]
        result[gripper_index] = low + command[6] * (high - low)
        return result
