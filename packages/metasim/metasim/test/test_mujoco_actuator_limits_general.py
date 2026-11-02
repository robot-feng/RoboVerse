"""Exercise the opt-in direct-actuator force-limit placement with MuJoCo."""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("dm_control")

from metasim.sim.mujoco.mujoco import _move_actuator_limits_to_joints

pytestmark = pytest.mark.general


def _model(*, gear: int = 1, extra_actuator: bool = False):
    extra = '<motor name="second" joint="j"/>' if extra_actuator else ""
    return mujoco.MjModel.from_xml_string(
        '<mujoco><option timestep="0.00166666666667" integrator="implicitfast" gravity="0 0 0"/>'
        '<worldbody><body><joint name="j" type="hinge" actuatorfrclimited="true" '
        'actuatorfrcrange="-2 3"/><geom type="sphere" size=".1" mass="1"/>'
        '</body></worldbody><actuator>'
        f'<position name="a" joint="j" gear="{gear}" kp="100000" kv="10000" '
        f'forcelimited="true" forcerange="-5 4"/>{extra}</actuator></mujoco>'
    )


def test_force_limit_move_preserves_clamped_joint_force_and_passive_force():
    original = _model()
    converted = _model()
    assert _move_actuator_limits_to_joints(converted, [0]) == 1
    np.testing.assert_allclose(converted.jnt_actfrcrange[0], [-2, 3])
    assert not converted.actuator_forcelimited[0]
    for q, velocity, target in ((0, 0, 10), (0, 0, -10), (0.1, 0.02, 0.100002), (0, 0, 0)):
        states = []
        for model in (original, converted):
            data = mujoco.MjData(model)
            data.qpos[0], data.qvel[0], data.ctrl[0] = q, velocity, target
            mujoco.mj_forward(model, data)
            states.append((data.qfrc_actuator[0], data.qfrc_passive[0]))
        np.testing.assert_allclose(states[0], states[1], atol=1e-10)


def test_force_limit_move_rejects_ambiguous_drive_without_partial_changes():
    for model in (_model(gear=2), _model(extra_actuator=True)):
        old_limited = model.actuator_forcelimited.copy()
        old_range = model.jnt_actfrcrange.copy()
        with pytest.raises(ValueError, match=r"gear|multiple"):
            _move_actuator_limits_to_joints(model, [0])
        np.testing.assert_array_equal(model.actuator_forcelimited, old_limited)
        np.testing.assert_array_equal(model.jnt_actfrcrange, old_range)


def test_force_limit_move_validates_every_joint_before_modifying_any():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint name="j1" type="hinge"/>'
        '<geom type="sphere" size=".1" mass="1"/></body><body pos="1 0 0">'
        '<joint name="j2" type="hinge"/><geom type="sphere" size=".1" mass="1"/>'
        '</body></worldbody><actuator>'
        '<motor name="a1" joint="j1" forcelimited="true" forcerange="-2 3"/>'
        '<motor name="a2" joint="j2" gear="2" forcelimited="true" forcerange="-2 3"/>'
        '</actuator></mujoco>'
    )
    before = model.actuator_forcelimited.copy()
    with pytest.raises(ValueError, match="gear"):
        _move_actuator_limits_to_joints(model, [0, 1])
    np.testing.assert_array_equal(model.actuator_forcelimited, before)
    assert not model.jnt_actfrclimited[0]


def test_force_limit_move_keeps_high_gain_implicit_drive_bounded():
    model = _model()
    _move_actuator_limits_to_joints(model, [0])
    data = mujoco.MjData(model)
    data.ctrl[0] = 0.01
    positions, forces = [], []
    for _ in range(120):
        mujoco.mj_step(model, data)
        positions.append(data.qpos[0])
        forces.append(data.qfrc_actuator[0])
    assert np.isfinite(positions).all()
    assert np.max(np.abs(positions)) < 0.05
    assert min(forces) >= -2 - 1e-9 and max(forces) <= 3 + 1e-9
