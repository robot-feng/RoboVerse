"""Native Newton fisheye calibration and inverse-domain coverage."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("newton")

from metasim.sim.newton.newton import _fisheye_camera_rays
from metasim.utils.fisheye import fisheye_rays, invertible_fisheye_fov


def _camera(**overrides):
    values = dict(
        name="fish",
        width=64,
        height=64,
        projection_model="opencv_fisheye",
        calibration_size=(64, 64),
        optical_center=(32.0, 32.0),
        focal_length_px=(20.0, 20.0),
        coefficients=(0.0, 0.0, 0.0, 0.0),
        max_fov_rad=2.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_opencv_rays_follow_equidistant_projection_and_invalidate_beyond_fov():
    rays = _fisheye_camera_rays(_camera(), "cpu").numpy()[:, :, 0, 1, :]
    # Newton samples pixel centers and camera forward is -Z.
    x, y = (41.5 - 32) / 20, (31.5 - 32) / 20
    theta = np.hypot(x, y)
    expected = np.array([np.sin(theta) * x / theta, -np.sin(theta) * y / theta, -np.cos(theta)])
    np.testing.assert_allclose(rays[31, 41], expected, atol=1e-6)
    np.testing.assert_array_equal(rays[0, 0], np.zeros(3))


def test_ftheta_uses_radius_in_pixels_as_function_of_angle():
    camera = _camera(projection_model="ftheta", coefficients=(0.0, 20.0, 0.0, 0.0, 0.0))
    rays = _fisheye_camera_rays(camera, "cpu").numpy()[:, :, 0, 1, :]
    assert np.linalg.norm(rays[0, 0]) == 0
    np.testing.assert_allclose(rays[31, 41, 2], -np.cos(np.hypot(9.5, -0.5) / 20), atol=1e-6)


def test_nonmonotonic_source_requires_explicit_safe_maximum():
    coefficients = (0.0994866863, -0.0455949791, 0.01246827, -0.00149616995)
    safe = invertible_fisheye_fov("opencv_fisheye", coefficients, 2 * np.pi)
    assert 4.29 < safe < 4.30
    with pytest.raises(ValueError, match="exceeds the invertible"):
        _fisheye_camera_rays(_camera(coefficients=coefficients, max_fov_rad=2 * np.pi), "cpu")
    rays = _fisheye_camera_rays(_camera(coefficients=coefficients, max_fov_rad=4.2), "cpu")
    assert np.isfinite(rays.numpy()).all()


def test_newton_native_rays_match_shared_projection_for_g2_calibration():
    camera = _camera(
        width=96,
        height=80,
        calibration_size=(1920, 1536),
        optical_center=(963.237, 767.674),
        focal_length_px=(471.616, 471.361),
        coefficients=(0.099486686, -0.045594979, 0.01246827, -0.00149617),
        max_fov_rad=4.2,
    )
    actual = _fisheye_camera_rays(camera, "cpu").numpy()[:, :, 0, 1, :]
    expected, mask = fisheye_rays(camera)
    np.testing.assert_array_equal(np.any(actual != 0, axis=-1), mask)
    np.testing.assert_allclose(actual, expected, atol=3e-6)


def test_old_newton_rejects_fisheye_instead_of_rendering_pinhole(monkeypatch):
    from metasim.sim.newton import newton as backend

    monkeypatch.setattr(backend, "_SENSOR_CAMERA_API", False)
    with pytest.raises(RuntimeError, match=r"Newton >= 1\.7"):
        backend._fisheye_camera_rays(_camera(), "cpu")


def test_newton_camera_sensor_renders_distinct_fisheye_models_on_cpu():
    import newton
    import torch
    import warp as wp

    from metasim.scenario.cameras import FisheyeCameraCfg, PinholeCameraCfg
    from metasim.sim.newton.newton import NewtonHandler

    builder = newton.ModelBuilder()
    builder.add_shape_sphere(body=-1, xform=wp.transform((0, 0, -2), wp.quat_identity()), radius=0.5)
    model = builder.finalize(device="cpu")
    common = dict(width=32, height=32, pos=(0, 0, 0), look_at=(0, 0, -1))
    fisheye = dict(calibration_size=(32, 32), optical_center=(16, 16), max_fov_rad=1.8)
    cameras = [
        FisheyeCameraCfg(name="fish1", focal_length_px=(12, 12), coefficients=(0, 0, 0, 0), **fisheye, **common),
        FisheyeCameraCfg(name="fish2", focal_length_px=(20, 20), coefficients=(0, 0, 0, 0), **fisheye, **common),
        PinholeCameraCfg(name="pinhole", **common),
    ]
    handler = NewtonHandler.__new__(NewtonHandler)
    handler.scenario = SimpleNamespace(
        cameras=cameras,
        sim_params=SimpleNamespace(newton_use_scenario_lights=False, newton_camera_background_rgb=None),
    )
    handler._model = model
    handler._state_0 = model.state()
    handler._device = "cpu"
    handler._num_envs = 1
    handler._shape_color_overrides = {}
    handler._init_tiled_cameras()
    assert len(handler._camera_groups) == 3
    assert not np.array_equal(
        handler._camera_groups[0]["camera_rays"].numpy(), handler._camera_groups[1]["camera_rays"].numpy()
    )
    states = handler._render_tiled_cameras([0])
    assert set(states) == {"fish1", "fish2", "pinhole"}
    for name in ("fish1", "fish2"):
        assert states[name].rgb.shape == (1, 32, 32, 3)
        assert states[name].valid_mask.dtype == torch.bool
        assert states[name].valid_mask.shape == (1, 32, 32)
        assert states[name].valid_mask[0, 16, 16]
        assert states[name].intrinsics is None
        assert 1.4 < states[name].depth[0, 16, 16] < 1.7
    assert not states["fish1"].valid_mask[0, 0, 0]
    assert states["pinhole"].valid_mask is None
