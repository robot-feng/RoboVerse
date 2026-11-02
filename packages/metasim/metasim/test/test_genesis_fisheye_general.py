"""Genesis fisheye cube-face orientation checks."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("genesis")

from metasim.sim.genesis.genesis import _fisheye_face_mount, _fisheye_world_rotation, _validate_fisheye_support
from metasim.utils.fisheye import CUBEMAP_ROTATIONS, cubemap_lookup


@pytest.mark.parametrize("look_at", [(1, 0, 0), (0, 0, -1), (0, 0, 1), (1, 2, 3)])
def test_world_fixed_cube_faces_follow_camera_frame(look_at):
    camera = SimpleNamespace(name="test", pos=(0, 0, 0), look_at=look_at)
    rotation = _fisheye_world_rotation(camera)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(-rotation[:, 2], np.asarray(look_at) / np.linalg.norm(look_at))
    for face in CUBEMAP_ROTATIONS:
        np.testing.assert_allclose(np.linalg.det(rotation @ face), 1, atol=1e-12)
    # Face-center lookup must agree with the same six world-space orientations.
    rays = -CUBEMAP_ROTATIONS[:, :, 2].reshape(1, 6, 3)
    faces, u, v = cubemap_lookup(rays, np.ones((1, 6), dtype=bool), 64)
    np.testing.assert_array_equal(faces[0], np.arange(6))
    np.testing.assert_allclose(u, 31.5)
    np.testing.assert_allclose(v, 31.5)


def test_world_fixed_camera_rejects_degenerate_look_at():
    camera = SimpleNamespace(name="test", pos=(0, 0, 0), look_at=(0, 0, 0))
    with pytest.raises(ValueError, match="look_at"):
        _fisheye_world_rotation(camera)


def test_mounted_faces_share_origin_and_rotate_in_camera_coordinates():
    camera = SimpleNamespace(name="test", pos=(0, 0, 0), look_at=(1, 2, 3))
    transform = np.eye(4)
    transform[:3, :3] = _fisheye_world_rotation(camera)
    transform[:3, 3] = (0.1, -0.2, 0.3)
    for face_rotation in CUBEMAP_ROTATIONS:
        result = _fisheye_face_mount(transform, face_rotation)
        np.testing.assert_array_equal(result[:3, 3], transform[:3, 3])
        np.testing.assert_allclose(-result[:3, 2], transform[:3, :3] @ -face_rotation[:, 2])


@pytest.mark.parametrize(
    "data_types,num_envs,gs_scene,message",
    [(["rgb", "depth"], 1, None, "RGB only"), (["rgb"], 2, None, "one environment"),
     (["rgb"], 1, object(), "Gaussian-splat")],
)
def test_unsupported_fisheye_features_fail_explicitly(data_types, num_envs, gs_scene, message):
    camera = SimpleNamespace(data_types=data_types)
    scenario = SimpleNamespace(num_envs=num_envs, gs_scene=gs_scene)
    with pytest.raises(NotImplementedError, match=message):
        _validate_fisheye_support(camera, scenario)


def test_genesis_fisheye_rgb_scene_cpu(monkeypatch):
    import torch

    from metasim.scenario.cameras import FisheyeCameraCfg
    from metasim.scenario.objects import PrimitiveSphereCfg
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.sim.genesis.genesis import GenesisHandler

    monkeypatch.setenv("METASIM_GENESIS_BACKEND", "cpu")
    camera = FisheyeCameraCfg(
        name="fish", width=64, height=64, data_types=["rgb"], pos=(0, -2, 1), look_at=(0, 0, 1),
        calibration_size=(64, 64), optical_center=(32, 32), focal_length_px=(20, 20),
        coefficients=(0, 0, 0, 0), max_fov_rad=2.0,
    )
    scenario = ScenarioCfg(
        simulator="genesis", headless=True, cameras=[camera],
        objects=[
            PrimitiveSphereCfg(name="ball", radius=0.2, default_position=(0, 0, 1), color=(1, 0, 0)),
            PrimitiveSphereCfg(name="side", radius=0.2, default_position=(1, -1.3, 1), color=(0, 1, 0)),
        ],
    )
    handler = GenesisHandler(scenario)
    try:
        handler.launch()
        state = handler.get_states(mode="tensor").cameras["fish"]
        assert state.rgb.shape == (1, 64, 64, 3)
        assert state.valid_mask.dtype == torch.bool
        assert state.valid_mask[0, 32, 32]
        assert not state.valid_mask[0, 0, 0]
        assert state.depth is None
        # The camera looks directly at the sphere: its center pixel is red.
        center = state.rgb[0, 32, 32]
        assert center[0] > center[1] + 20 and center[0] > center[2] + 20
        # The second sphere is about 55 degrees to the right: it exercises a different cube face.
        side = state.rgb[0, 32, 51]
        assert side[1] > side[0] + 20 and side[1] > side[2] + 20
    finally:
        handler.close()
