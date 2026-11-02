"""The newton compat shims forward renamed / removed attributes on both old and new newton APIs.

newton 1.5 replaced ``Model.joint_target_pos`` / ``joint_target_vel`` (and the same on ``Control``)
with ``joint_target_q`` / ``joint_target_qd`` behind a ``RemovedAttribute`` descriptor that raises on
read and write; 1.6 renamed ``Model.num_worlds`` to ``world_count`` outright. The shims are exercised
against a synthetic ``newton`` module so the test runs without the engine installed.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

import pytest


def test_collision_only_urdf_can_hide_newton_render_shapes_without_disabling_collision(tmp_path):
    newton = pytest.importorskip("newton")
    wp = pytest.importorskip("warp")
    if not hasattr(newton, "ShapeFlags"):
        pytest.skip("This Newton version does not expose shape flags")
    from metasim.constants import PhysicStateType
    from metasim.scenario.objects import RigidObjCfg
    from metasim.sim.newton.newton import NewtonHandler

    path = tmp_path / "collision_only.urdf"
    path.write_text(
        '<robot name="collision_only"><link name="base"><collision><geometry>'
        '<box size="1 1 0.1"/></geometry></collision></link></robot>', encoding="utf-8"
    )
    visible = int(newton.ShapeFlags.VISIBLE)
    collides = int(newton.ShapeFlags.COLLIDE_SHAPES)
    with wp.ScopedDevice("cpu"):
        for hide in (False, True):
            builder = newton.ModelBuilder()
            builder.add_shape_sphere(body=-1, radius=0.01)
            original = int(builder.shape_flags[0])
            cfg = RigidObjCfg(name="test", urdf_path=str(path), physics=PhysicStateType.GEOM,
                              newton_hide_collision_shapes=hide)
            NewtonHandler.__new__(NewtonHandler)._add_object_to_builder(builder, cfg)
            assert int(builder.shape_flags[0]) == original
            flags = int(builder.shape_flags[1])
            assert flags & collides
            assert bool(flags & visible) is not hide


def test_site_connect_preserves_both_authored_anchors(tmp_path):
    """A site pair may intentionally be noncoincident at the reference pose."""
    newton = pytest.importorskip("newton")
    pytest.importorskip("mujoco")
    import numpy as np

    from metasim.sim.newton._newton_compat import restore_site_connect_anchors

    path = tmp_path / "site_pair.xml"
    path.write_text(
        '<mujoco model="test"><worldbody><body name="base">'
        '<geom type="box" size=".05 .05 .05" mass="1"/>'
        '<body name="a"><joint name="a" type="hinge"/>'
        '<geom type="sphere" size=".1" mass="1"/><site name="a_site" pos=".12 0 0"/></body>'
        '<body name="b" pos=".1 0 0"><joint name="b" type="hinge"/>'
        '<geom type="sphere" size=".1" mass="1"/><site name="b_site" pos=".03 0 0"/></body>'
        '</body></worldbody><equality><connect name="loop" site1="a_site" site2="b_site"/></equality></mujoco>',
        encoding="utf-8",
    )
    builder = newton.ModelBuilder()
    if not hasattr(builder, "add_mjcf"):
        pytest.skip("This Newton version does not expose the current MJCF importer")
    builder.add_mjcf(str(path), floating=False)
    model = builder.finalize(device="cpu")
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True)
    result = restore_site_connect_anchors(model, solver, {"test/loop": "b_site"})
    if result == -1:
        pytest.skip("This Newton version does not expose equality mappings")
    assert result == 1
    np.testing.assert_allclose(solver.mj_model.eq_data[0, :6], [.12, 0, 0, .03, 0, 0], atol=1e-6)
    np.testing.assert_allclose(model.joint_X_c.numpy()[-1, :3], [.03, 0, 0], atol=1e-6)


def test_site_connect_maps_repeated_labels_to_each_world():
    import numpy as np

    from metasim.sim.newton._newton_compat import restore_site_connect_anchors

    class Array:
        def __init__(self, value):
            self.value = np.array(value)

        def numpy(self):
            return self.value.copy()

        def assign(self, value):
            self.value[:] = value

    model = types.SimpleNamespace(
        mujoco=types.SimpleNamespace(
            equality_constraint_label=["robot/loop", "robot/loop"],
            equality_constraint_body2=Array([1, 3]),
            equality_constraint_target=Array([0, 1]),
            equality_constraint_target_kind=Array([1, 1]),
        ),
        shape_label=["robot/body/site2", "robot/body/site2"],
        shape_body=Array([1, 3]),
        shape_transform=Array([[.03, 0, 0, 0, 0, 0, 1], [.03, 0, 0, 0, 0, 0, 1]]),
        joint_X_c=Array([[.02, 0, 0, 0, 0, 0, 1], [.02, 0, 0, 0, 0, 0, 1]]),
        joint_child=Array([1, 3]),
    )
    solver = types.SimpleNamespace(
        mjc_eq_to_newton_eq=Array([[0], [1]]),
        mjw_model=types.SimpleNamespace(eq_data=Array(np.zeros((2, 1, 11)))),
        use_mujoco_cpu=False,
    )
    assert restore_site_connect_anchors(model, solver, {"robot/loop": "site2"}) == 2
    np.testing.assert_allclose(solver.mjw_model.eq_data.value[:, 0, 3:6], [[.03, 0, 0], [.03, 0, 0]])
    np.testing.assert_allclose(model.joint_X_c.value[:, :3], [[.03, 0, 0], [.03, 0, 0]])


class RemovedAttribute:
    """Mirror of ``newton._src.utils.deprecation.RemovedAttribute`` (data descriptor that raises)."""

    def __init__(self, replacement: str):
        self._message = f"removed; use {replacement} instead"

    def __get__(self, instance, owner=None):
        raise AttributeError(self._message)

    def __set__(self, instance, value):
        raise AttributeError(self._message)


def _fake_newton(*, removed: bool, renamed: bool) -> types.ModuleType:
    mod = types.ModuleType("newton")

    class Model:
        def __init__(self):
            if renamed:
                self.world_count = 4
            else:
                self.num_worlds = 4
            if removed:
                self.joint_target_q = "q"
                self.joint_target_qd = "qd"
            else:
                self.joint_target_pos = "q"
                self.joint_target_vel = "qd"

    class Control:
        def __init__(self):
            if removed:
                self.joint_target_q = "cq"
            else:
                self.joint_target_pos = "cq"

    if removed:
        Model.joint_target_pos = RemovedAttribute("joint_target_q")
        Model.joint_target_vel = RemovedAttribute("joint_target_qd")
        Control.joint_target_pos = RemovedAttribute("joint_target_q")
        Control.joint_target_vel = RemovedAttribute("joint_target_qd")

    class ModelBuilder:
        pass

    mod.Model, mod.Control, mod.ModelBuilder = Model, Control, ModelBuilder
    mod.JointType = object
    mod.sensors = types.ModuleType("newton.sensors")  # no populate_contacts: exercises the >=1.2 path
    return mod


def _load_compat(monkeypatch, fake):
    """Load ``_newton_compat.py`` against ``fake`` as the ``newton`` package.

    Loaded from its file so ``metasim.sim.newton/__init__`` (which imports the real engine) stays out
    of the picture.
    """
    for name in list(sys.modules):
        if name == "newton" or name.startswith("newton."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "newton", fake)
    monkeypatch.setitem(sys.modules, "newton.sensors", fake.sensors)
    import metasim

    path = os.path.join(os.path.dirname(metasim.__file__), "sim", "newton", "_newton_compat.py")
    spec = importlib.util.spec_from_file_location("_newton_compat_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.general
def test_removed_joint_target_attributes_forward_to_new_names(monkeypatch):
    fake = _fake_newton(removed=True, renamed=True)
    compat = _load_compat(monkeypatch, fake)
    compat._install_removed_attr_aliases()
    compat._install_renamed_attr_aliases()
    model, control = fake.Model(), fake.Control()
    assert model.joint_target_pos == "q" and model.joint_target_vel == "qd"
    assert control.joint_target_pos == "cq"
    model.joint_target_pos = "new"
    assert model.joint_target_q == "new", "writes must land on the new attribute, not a dead shadow"
    assert model.num_worlds == 4


@pytest.mark.general
def test_old_newton_attributes_are_left_alone(monkeypatch):
    fake = _fake_newton(removed=False, renamed=False)
    compat = _load_compat(monkeypatch, fake)
    compat._install_removed_attr_aliases()
    compat._install_renamed_attr_aliases()
    model = fake.Model()
    assert model.joint_target_pos == "q" and model.num_worlds == 4
    model.num_worlds = 8
    assert model.num_worlds == 8


@pytest.mark.general
def test_finalized_root_body_lookup_survives_fixed_joint_collapse(monkeypatch):
    fake = _fake_newton(removed=True, renamed=True)
    compat = _load_compat(monkeypatch, fake)
    body_name_to_id = {
        "franka/worldbody/panda_link0/panda_link1": 0,
        "cube": 9,
        "sphere": 10,
        "converted_from_mjcf/bbq_sauce": 11,
        "box_base/robot_base": 12,
        "box_base/box_lid": 13,
    }
    body_parent_joint = {0: 0, 11: 1, 12: 2, 13: 3}
    joint_parents = [-1, -1, -1, 12]

    assert compat.object_root_body_candidates(body_name_to_id, body_parent_joint, joint_parents, "cube") == [9]
    assert compat.object_root_body_candidates(body_name_to_id, body_parent_joint, joint_parents, "sphere") == [10]
    assert compat.object_root_body_candidates(body_name_to_id, body_parent_joint, joint_parents, "bbq_sauce") == [11]
    assert compat.object_root_body_candidates(body_name_to_id, body_parent_joint, joint_parents, "box_base") == [12]


@pytest.mark.general
def test_newton_default_ground_respects_scenario_opt_out(monkeypatch):
    pytest.importorskip("newton")
    from types import SimpleNamespace

    from metasim.sim.newton import newton as newton_module
    from metasim.sim.newton.newton import NewtonHandler

    calls = []

    class FakeBuilder:
        def add_ground_plane(self, *, cfg, color):
            calls.append(("ground", cfg, color))
            return 4

    handler = NewtonHandler.__new__(NewtonHandler)
    handler.scenario = SimpleNamespace(add_default_ground=False)
    handler._shape_color_overrides = {}
    monkeypatch.setattr(newton_module, "set_current_world", lambda builder, world: calls.append(("world", world)))
    monkeypatch.setattr(newton_module, "_add_checkerboard_ground_visual", lambda builder: calls.append(("visual",)))
    monkeypatch.setattr(handler, "_get_ground_shape_cfg", lambda builder: "ground_cfg")

    handler._add_default_ground(FakeBuilder())
    assert calls == []
    assert handler._shape_color_overrides == {}


@pytest.mark.general
def test_newton_default_ground_keeps_legacy_behavior(monkeypatch):
    pytest.importorskip("newton")
    from types import SimpleNamespace

    from metasim.sim.newton import newton as newton_module
    from metasim.sim.newton.newton import NewtonHandler

    calls = []

    class FakeBuilder:
        def add_ground_plane(self, *, cfg, color):
            calls.append(("ground", cfg, color))
            return 4

    handler = NewtonHandler.__new__(NewtonHandler)
    handler.scenario = SimpleNamespace()
    handler._shape_color_overrides = {}
    monkeypatch.setattr(newton_module, "set_current_world", lambda builder, world: calls.append(("world", world)))
    monkeypatch.setattr(newton_module, "_add_checkerboard_ground_visual", lambda builder: calls.append(("visual",)))
    monkeypatch.setattr(handler, "_get_ground_shape_cfg", lambda builder: "ground_cfg")

    handler._add_default_ground(FakeBuilder())
    assert calls == [
        ("world", -1),
        ("ground", "ground_cfg", (0.18, 0.24, 0.32)),
        ("visual",),
    ]
    assert handler._shape_color_overrides[4].tolist() == pytest.approx([0.18, 0.24, 0.32, 1.0])


@pytest.mark.general
def test_mounted_camera_uses_each_world_link_pose():
    """Mounted camera transforms follow their corresponding world after a state change."""
    pytest.importorskip("newton")
    import numpy as np
    import warp as wp

    from metasim.scenario.cameras import PinholeCameraCfg
    from metasim.sim.newton.newton import NewtonHandler

    wp.init()
    camera = PinholeCameraCfg(
        name="head",
        mount_to="robot",
        mount_link="head_link",
        mount_pos=(0.0, 0.0, 1.0),
        mount_quat=(1.0, 0.0, 0.0, 0.0),
    )
    body_poses = np.array([[0, 0, 0, 0, 0, 0, 1], [1, 2, 3, 0, 0, 0, 1]], dtype=np.float32)
    handler = object.__new__(NewtonHandler)
    handler.scenario = types.SimpleNamespace(cameras=[camera])
    handler._num_envs = 2
    handler._model = types.SimpleNamespace(device="cpu")
    handler._state_0 = types.SimpleNamespace(body_q=types.SimpleNamespace(numpy=lambda: body_poses))
    handler._body_name_cache = {0: {"robot/base/head_link": 0}, 1: {"robot/base/head_link": 1}}
    transforms = handler._build_camera_transforms([0]).numpy()
    np.testing.assert_allclose(transforms[0, :, :3], [[0, 0, 1], [1, 2, 4]])
    body_poses[1, 0] = 4
    moved = handler._build_camera_transforms([0]).numpy()
    np.testing.assert_allclose(moved[0, 1, :3], [4, 2, 4])
    handler._body_name_cache[1]["robot/other/head_link"] = 1
    with pytest.raises(ValueError, match="uniquely resolve"):
        handler._build_camera_transforms([0])


@pytest.mark.general
def test_newton_camera_light_keeps_default_direction_unless_opted_in():
    pytest.importorskip("newton")
    from metasim.scenario.lights import DistantLightCfg
    from metasim.scenario.simulator_params import SimParamCfg
    from metasim.sim.newton.newton import _scenario_camera_light_direction

    scenario = types.SimpleNamespace(sim_params=SimParamCfg(), lights=[DistantLightCfg(polar=30, azimuth=45)])
    assert _scenario_camera_light_direction(scenario) is None


@pytest.mark.general
def test_newton_camera_light_uses_first_distant_light_direction():
    pytest.importorskip("newton")
    import numpy as np

    from metasim.scenario.lights import DistantLightCfg
    from metasim.scenario.simulator_params import SimParamCfg
    from metasim.sim.newton.newton import _scenario_camera_light_direction

    scenario = types.SimpleNamespace(
        sim_params=SimParamCfg(newton_use_scenario_lights=True),
        lights=[DistantLightCfg(polar=30, azimuth=45), DistantLightCfg(polar=0)],
    )
    np.testing.assert_allclose(
        _scenario_camera_light_direction(scenario),
        (-np.sqrt(2) / 4, np.sqrt(2) / 4, -np.sqrt(3) / 2),
        atol=1e-6,
    )


@pytest.mark.general
def test_newton_background_configuration_is_optional_and_validated():
    from metasim.scenario.simulator_params import SimParamCfg

    assert SimParamCfg().newton_camera_background_rgb is None
    assert SimParamCfg(newton_camera_background_rgb=(23, 36, 53)).newton_camera_background_rgb == (23, 36, 53)
    for invalid in [(-1, 0, 0), (0, 256, 0), (0, 0), (0, 0, 0.5)]:
        with pytest.raises(ValueError, match="newton_camera_background_rgb"):
            SimParamCfg(newton_camera_background_rgb=invalid)
