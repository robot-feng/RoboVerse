"""Integration checks for MuJoCo's cube-map fisheye projection."""

import pytest

pytest.importorskip("mujoco")


def test_mujoco_fisheye_rgb_direction_and_validity():
    import torch

    from metasim.scenario.cameras import FisheyeCameraCfg
    from metasim.scenario.objects import PrimitiveSphereCfg
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.sim.mujoco.mujoco import MujocoHandler

    camera = FisheyeCameraCfg(
        name="fish", width=64, height=64, data_types=["rgb"], pos=(0, -2, 1), look_at=(0, 0, 1),
        calibration_size=(64, 64), optical_center=(32, 32), focal_length_px=(20, 20),
        coefficients=(0, 0, 0, 0), max_fov_rad=2.0,
    )
    scenario = ScenarioCfg(
        simulator="mujoco", headless=True, cameras=[camera],
        objects=[
            PrimitiveSphereCfg(name="ball", radius=0.2, default_position=(0, 0, 1), color=(1, 0, 0)),
            PrimitiveSphereCfg(name="side", radius=0.2, default_position=(1, -1.3, 1), color=(0, 1, 0)),
        ],
    )
    handler = MujocoHandler(scenario)
    try:
        handler.launch()
        state = handler.get_states(mode="tensor").cameras["fish"]
        assert state.rgb.shape == (1, 64, 64, 3)
        assert state.valid_mask.dtype == torch.bool
        assert state.valid_mask[0, 32, 32] and not state.valid_mask[0, 0, 0]
        assert state.depth is None and state.intrinsics is None
        center, side = state.rgb[0, 32, 32], state.rgb[0, 32, 51]
        assert int(center[0]) > int(center[1]) + 20
        assert int(side[1]) > int(side[0]) + 20
    finally:
        handler.close()
