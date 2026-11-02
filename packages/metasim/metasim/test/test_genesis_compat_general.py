"""Tests for Genesis-version and configuration compatibility helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from metasim.sim.genesis._genesis_compat import (
    apply_genesis_actuator_settings,
    apply_genesis_mesh_texture_compat,
    genesis_absolute_positions,
    genesis_floating_mjcf,
    genesis_mjcf_armature_kwargs,
    genesis_position_offsets,
    genesis_primitive_options,
    genesis_supports_anisotropic_scale,
    genesis_vis_options_kwargs,
    normalize_genesis_scale,
    refresh_genesis_visualizer,
)


@pytest.mark.general
def test_floating_mjcf_preserves_source_and_resolves_assets(tmp_path):
    import xml.etree.ElementTree as ET

    source = tmp_path / "object.xml"
    original = '<mujoco><compiler meshdir="meshes"/><asset><mesh file="x.obj"/></asset><worldbody><body name="obj"/></worldbody></mujoco>'
    source.write_text(original, encoding="utf-8")
    converted = ET.fromstring(genesis_floating_mjcf(str(source)))
    assert converted.find("./worldbody/body/freejoint") is not None
    assert converted.find("compiler").get("meshdir") == str(tmp_path / "meshes")
    assert source.read_text(encoding="utf-8") == original

    source.write_text('<mujoco><worldbody><body><freejoint/></body></worldbody></mujoco>', encoding="utf-8")
    assert genesis_floating_mjcf(str(source)) == str(source)
    source.write_text('<mujoco><worldbody><body/><body/></worldbody></mujoco>', encoding="utf-8")
    with pytest.raises(ValueError, match="one top-level body"):
        genesis_floating_mjcf(str(source))


@pytest.mark.general
def test_mjcf_armature_options_preserve_source_and_support_legacy_genesis():
    class Current:
        model_fields = {"default_armature": SimpleNamespace(annotation=float | None)}

    class LegacyWithoutOption:
        __fields__ = {}

    class LegacyOptional:
        __fields__ = {"default_armature": SimpleNamespace(annotation=float, allow_none=True)}

    class NoNullableOption:
        model_fields = {"default_armature": SimpleNamespace(annotation=float)}

    assert genesis_mjcf_armature_kwargs(Current) == {"default_armature": None}
    assert genesis_mjcf_armature_kwargs(LegacyWithoutOption) == {}
    assert genesis_mjcf_armature_kwargs(LegacyOptional) == {"default_armature": None}
    assert genesis_mjcf_armature_kwargs(NoNullableOption) == {}


@pytest.mark.general
def test_refresh_genesis_visualizer_updates_headless_and_handles_legacy_signature():
    calls = []

    class Current:
        def update(self, force=False):
            calls.append(force)

    class Legacy:
        def update(self):
            calls.append("legacy")

    refresh_genesis_visualizer(SimpleNamespace(visualizer=Current()), headless=True)
    refresh_genesis_visualizer(SimpleNamespace(visualizer=Legacy()), headless=True)
    assert calls == [True, "legacy"]
    with pytest.raises(NotImplementedError):
        refresh_genesis_visualizer(SimpleNamespace(), headless=True)


@pytest.mark.general
def test_vis_options_use_current_argument_and_support_older_genesis():
    class CurrentVisOptions:
        model_fields = {"rendered_envs_idx": object()}

    class LegacyVisOptions:
        __fields__ = {"n_rendered_envs": object()}

    assert genesis_vis_options_kwargs(CurrentVisOptions, 3) == {"rendered_envs_idx": (0, 1, 2)}
    assert genesis_vis_options_kwargs(LegacyVisOptions, 2) == {"n_rendered_envs": 2}


@pytest.mark.general
def test_vis_options_translate_distant_and_dome_lights():
    from metasim.scenario.lights import DistantLightCfg, DomeLightCfg

    class CurrentVisOptions:
        model_fields = {"rendered_envs_idx": object(), "lights": object(), "ambient_light": object()}

    options = genesis_vis_options_kwargs(
        CurrentVisOptions,
        1,
        [DomeLightCfg(intensity=450), DistantLightCfg(intensity=1100, polar=30, azimuth=45)],
    )
    assert options["rendered_envs_idx"] == (0,)
    assert options["ambient_light"] == pytest.approx((0.2025, 0.2025, 0.2025))
    assert len(options["lights"]) == 1
    assert options["lights"][0]["intensity"] == pytest.approx(3.732142857)
    assert np.linalg.norm(options["lights"][0]["dir"]) == pytest.approx(1.0)


@pytest.mark.general
def test_genesis_scale_normalization_preserves_anisotropy_when_supported():
    class TupleScaleMorph:
        model_fields = {"scale": SimpleNamespace(annotation=tuple[float, float, float])}

    class IsotropicOnlyMorph:
        model_fields = {"scale": SimpleNamespace(annotation=tuple[float, float, float])}
        __pydantic_decorators__ = SimpleNamespace(model_validators={"_enforce_isotropic_scale": object()})

    class ScalarScaleMorph:
        __fields__ = {"scale": SimpleNamespace(outer_type_=float)}

    assert genesis_supports_anisotropic_scale(TupleScaleMorph)
    assert not genesis_supports_anisotropic_scale(IsotropicOnlyMorph)
    assert not genesis_supports_anisotropic_scale(ScalarScaleMorph)
    assert normalize_genesis_scale((2, 2, 2)) == (2.0, False)
    fallback_scale, anisotropy_was_discarded = normalize_genesis_scale((1, 2, 1))
    assert fallback_scale == pytest.approx(2 ** (1 / 3))
    assert anisotropy_was_discarded
    assert normalize_genesis_scale((1, 2, 1), supports_anisotropic=True) == ((1.0, 2.0, 1.0), False)
    assert normalize_genesis_scale(0.5) == (0.5, False)


@pytest.mark.general
def test_genesis_mesh_texture_compat_keeps_source_textures_and_explicit_surfaces():
    texture_fields = {
        "diffuse_texture",
        "emissive_texture",
        "metallic_texture",
        "normal_texture",
        "opacity_texture",
        "roughness_texture",
    }

    class ColorTexture:
        def __init__(self, color=(1.0, 1.0, 1.0)):
            self.color = color

    class ImageTexture:
        pass

    class DefaultSurface:
        def __init__(self, *, color=None, diffuse_texture=None):
            self.color = color
            self.roughness = None
            self.metallic = None
            self.emissive = None
            self.diffuse_texture = diffuse_texture
            self.model_fields_set = {"diffuse_texture"} if diffuse_texture is not None else set()

        @classmethod
        def engine_default(cls):
            surface = cls(diffuse_texture=ColorTexture())
            surface.model_fields_set = texture_fields.copy()
            return surface

    class FakeGenesis:
        surfaces = SimpleNamespace(Default=DefaultSurface)
        textures = SimpleNamespace(ColorTexture=ColorTexture, ImageTexture=ImageTexture)

    class FakeMesh:
        @classmethod
        def from_trimesh(cls, mesh, *, surface=None):
            return mesh, surface

    class TexturedMesh:
        visual = SimpleNamespace(kind="texture", material=SimpleNamespace(image=object()))

    FakeGenesis.Mesh = FakeMesh
    FakeMesh.__module__ = "genesis.engine.mesh"
    # The compatibility helper imports the runtime Mesh class, so temporarily
    # publish the same stand-in module shape it expects.
    import sys
    from types import ModuleType

    fake_module = ModuleType("genesis.engine.mesh")
    fake_module.Mesh = FakeMesh
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(sys.modules, "genesis.engine.mesh", fake_module)
    try:
        apply_genesis_mesh_texture_compat(FakeGenesis)
        _, imported_surface = FakeMesh.from_trimesh(TexturedMesh())
        default_surface = DefaultSurface.engine_default()
        _, repaired_default_surface = FakeMesh.from_trimesh(TexturedMesh(), surface=default_surface)
        requested_surface = DefaultSurface(diffuse_texture=ImageTexture())
        _, explicit_surface = FakeMesh.from_trimesh(TexturedMesh(), surface=requested_surface)
        colored_surface = DefaultSurface(color=(0.2, 0.4, 0.6))
        _, explicit_color_surface = FakeMesh.from_trimesh(TexturedMesh(), surface=colored_surface)
        apply_genesis_mesh_texture_compat(FakeGenesis)
        assert isinstance(imported_surface, DefaultSurface)
        assert repaired_default_surface is not default_surface
        assert explicit_surface is requested_surface
        assert explicit_color_surface is colored_surface
        assert FakeMesh._metasim_source_texture_compat
    finally:
        monkeypatch.undo()


@pytest.mark.general
def test_genesis_joint_positions_round_trip_through_qpos0_offsets():
    qpos0 = np.array([0.0, -0.785, 0.0, -2.356], dtype=np.float32)
    absolute = np.array([0.1, -0.7, -0.2, -2.1], dtype=np.float32)

    offsets = genesis_position_offsets(absolute, qpos0)
    round_trip = genesis_absolute_positions(offsets, qpos0)

    np.testing.assert_allclose(round_trip, absolute)


@pytest.mark.general
def test_robot_actuator_settings_are_applied_per_joint_and_control_mode():
    class FakeEntity:
        def __init__(self):
            self.joints = [
                SimpleNamespace(name="position_joint", dofs_idx_local=[2]),
                SimpleNamespace(name="effort_joint", dofs_idx_local=[5]),
            ]
            self.calls = []

        def set_dofs_kp(self, value, *, dofs_idx_local):
            self.calls.append(("kp", value, dofs_idx_local))

        def set_dofs_kv(self, value, *, dofs_idx_local):
            self.calls.append(("kv", value, dofs_idx_local))

        def set_dofs_force_range(self, lower, upper, *, dofs_idx_local):
            self.calls.append(("force_range", lower, upper, dofs_idx_local))

    entity = FakeEntity()
    actuators = {
        "position_joint": SimpleNamespace(stiffness=400.0, damping=40.0, effort_limit_sim=12.0),
        "effort_joint": SimpleNamespace(stiffness=100.0, damping=10.0, effort_limit_sim=4.0),
    }

    apply_genesis_actuator_settings(
        entity,
        actuators,
        {"position_joint": "position", "effort_joint": "effort"},
    )

    assert entity.calls == [
        ("kp", 400.0, [2]),
        ("kv", 40.0, [2]),
        ("force_range", -12.0, 12.0, [2]),
        ("kp", 0.0, [5]),
        ("kv", 0.0, [5]),
        ("force_range", -4.0, 4.0, [5]),
    ]


@pytest.mark.general
def test_genesis_control_mode_defaults_to_position_for_unconfigured_articulations():
    pytest.importorskip("genesis")

    from metasim.sim.genesis.genesis import GenesisHandler

    handler = GenesisHandler.__new__(GenesisHandler)
    handler.object_dict = {
        "passive": SimpleNamespace(control_type=None),
        "empty": SimpleNamespace(control_type={}),
        "effort": SimpleNamespace(control_type={"joint": "effort"}),
    }

    assert handler._get_control_mode("passive") == "position"
    assert handler._get_control_mode("empty") == "position"
    assert handler._get_control_mode("effort") == "effort"


@pytest.mark.general
def test_genesis_primitive_options_honour_scene_pose_and_physics():
    from metasim.constants import PhysicStateType
    from metasim.scenario.objects import PrimitiveCubeCfg

    obj = PrimitiveCubeCfg(
        name="marker",
        size=(0.1, 0.1, 0.01),
        color=(1.0, 0.0, 0.0),
        physics=PhysicStateType.XFORM,
        default_position=(0.4, 0.2, 0.1),
    )
    assert genesis_primitive_options(obj) == {
        "pos": (0.4, 0.2, 0.1),
        "quat": (1.0, 0.0, 0.0, 0.0),
        "fixed": True,
        "collision": False,
    }


@pytest.mark.general
def test_genesis_default_ground_respects_scenario_opt_out():
    pytest.importorskip("genesis")

    from metasim.sim.genesis.genesis import GenesisHandler

    class FakeScene:
        def __init__(self):
            self.entities = []

        def add_entity(self, entity):
            self.entities.append(entity)

    handler = GenesisHandler.__new__(GenesisHandler)
    handler.scenario = SimpleNamespace(add_default_ground=False)
    handler.scene_inst = FakeScene()
    handler._add_default_ground()
    assert handler.scene_inst.entities == []


@pytest.mark.general
def test_genesis_default_ground_keeps_legacy_behavior(monkeypatch):
    pytest.importorskip("genesis")

    import importlib

    from metasim.sim.genesis.genesis import GenesisHandler

    genesis_module = importlib.import_module("metasim.sim.genesis.genesis")
    monkeypatch.setattr(genesis_module, "gs", SimpleNamespace(morphs=SimpleNamespace(Plane=lambda: "plane")))

    class FakeScene:
        def __init__(self):
            self.entities = []

        def add_entity(self, entity):
            self.entities.append(entity)

    handler = GenesisHandler.__new__(GenesisHandler)
    handler.scenario = SimpleNamespace()
    handler.scene_inst = FakeScene()
    handler._add_default_ground()
    assert handler.scene_inst.entities == ["plane"]


@pytest.mark.general
def test_genesis_articulation_states_expose_sorted_per_link_pose_and_velocity():
    torch = pytest.importorskip("torch")

    from metasim.sim.genesis.genesis import GenesisHandler

    class FakeLink:
        def __init__(self, name, idx_local):
            self.name = name
            self.idx_local = idx_local

    class FakeEntity:
        links = [FakeLink("z_link", 0), FakeLink("a_link", 1)]

        def get_links_pos(self, *, links_idx_local, envs_idx, relative):
            assert envs_idx == [0] and relative is True
            return torch.tensor([[[float(index), 0.0, 0.0] for index in links_idx_local]])

        def get_links_quat(self, *, links_idx_local, envs_idx, relative):
            return torch.tensor([[[1.0, 0.0, 0.0, 0.0] for _ in links_idx_local]])

        def get_links_vel(self, *, links_idx_local, envs_idx, relative):
            return torch.tensor([[[0.0, float(index), 0.0] for index in links_idx_local]])

        def get_links_ang(self, *, links_idx_local, envs_idx):
            return torch.tensor([[[0.0, 0.0, float(index)] for index in links_idx_local]])

    handler = GenesisHandler.__new__(GenesisHandler)
    handler.object_inst_dict = {"robot": FakeEntity()}
    body_names = handler._get_body_names("robot", sort=True)
    assert body_names == ["a_link", "z_link"]
    body_state = handler._get_body_state(handler.object_inst_dict["robot"], body_names, [0])
    assert tuple(body_state.shape) == (1, 2, 13)
    torch.testing.assert_close(body_state[0, :, 0], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(body_state[0, :, 3], torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(body_state[0, :, 8], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(body_state[0, :, 12], torch.tensor([1.0, 0.0]))
