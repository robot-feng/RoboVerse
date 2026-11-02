"""Lightweight provenance and behavior contracts for migrated SynthICL skills."""

from __future__ import annotations

import numpy as np
import pytest

from roboverse_learn.il.icl.waypoints import (
    pick_place_task_key,
    push_task_key,
    sample_push_gn_waypoints,
)


@pytest.mark.general
def test_pick_place_key_pins_upstream_and_all_acceptance_axes():
    key = pick_place_task_key(
        object_id="synthicl_tea_box_base2",
        target_object_id="synthicl_tea_box_base2",
        grasp_index=3,
        backend="mujoco",
        seed=11,
    )
    assert key == {
        "source": "https://github.com/sNiper-Qian/synthicl-pseudo-demo",
        "upstream_sha": "131d22f9a4a8f066aa2fd7bee8d88118dcc0766a",
        "task_id": "pick_and_place_gn",
        "level_or_variant": "graspnet_index_3",
        "scene_id": "two_pose_tabletop_v1/synthicl_tea_box_base2_on_synthicl_tea_box_base2",
        "embodiment": "franka",
        "backend": "mujoco",
        "seed": 11,
    }


@pytest.mark.general
@pytest.mark.parametrize(
    ("field", "value"),
    (("backend", "isaacsim"), ("seed", -1), ("grasp_index", -1), ("object_id", "../box")),
)
def test_pick_place_key_rejects_invalid_identity_values(field, value):
    arguments = {
        "object_id": "box",
        "target_object_id": "support",
        "grasp_index": 0,
        "backend": "mujoco",
        "seed": 0,
    }
    arguments[field] = value
    with pytest.raises(ValueError):
        pick_place_task_key(**arguments)


@pytest.mark.general
def test_object_on_top_relation_places_source_bottom_on_target_top_without_rescaling():
    from roboverse_learn.il.icl.waypoints import object_on_top_relation

    target_bounds = [[-0.04, -0.05, -0.01], [0.04, 0.05, 0.08]]
    source_bounds = [[-0.03, -0.02, -0.02], [0.03, 0.02, 0.06]]
    relation = object_on_top_relation(target_bounds_m=target_bounds, source_bounds_m=source_bounds)
    source_bottom = relation[2, 3] + source_bounds[0][2]
    target_top = target_bounds[1][2]
    assert source_bottom == pytest.approx(target_top)
    with pytest.raises(ValueError, match="scale"):
        pick_place_task_key(
            object_id="box",
            target_object_id="support",
            grasp_index=0,
            backend="mujoco",
            seed=0,
            source_scale=0.9,
        )


@pytest.mark.general
def test_push_key_pins_single_object_skill_and_rejects_rescaling():
    key = push_task_key(object_id="synthicl_tea_box_base2", grasp_index=7, backend="genesis", seed=21)
    assert key == {
        "source": "https://github.com/sNiper-Qian/synthicl-pseudo-demo",
        "upstream_sha": "131d22f9a4a8f066aa2fd7bee8d88118dcc0766a",
        "task_id": "push_gn",
        "level_or_variant": "graspnet_index_7",
        "scene_id": "single_object_ground_plane_v1/synthicl_tea_box_base2",
        "embodiment": "franka",
        "backend": "genesis",
        "seed": 21,
    }
    with pytest.raises(ValueError, match="scale"):
        push_task_key(object_id="box", grasp_index=0, backend="mujoco", seed=0, source_scale=0.95)


@pytest.mark.general
def test_push_sampler_preserves_five_source_waypoints_and_action_order():
    grasp = np.zeros((1, 17), dtype=np.float64)
    grasp[0, 3] = 0.02
    grasp[0, 4:13] = np.eye(3).reshape(-1)
    grasp[0, 13:16] = [0.01, 0.02, 0.03]
    sampled = sample_push_gn_waypoints(
        grasp_rows=grasp,
        bbox_half_extents_m=[0.02, 0.03, 0.04],
        grasp_angle_rad=1.57,
        seed=17,
        pregrasp_distance_m=(0.05, 0.10),
    )
    assert sampled.waypoints.shape == (5, 8)
    assert sampled.grasp_index == 0
    assert sampled.action_semantics == (
        "approach_pregrasp_open",
        "close_without_attachment",
        "approach_grasp_closed",
        "attach_object_to_hand",
        "approach_postpush_attached",
    )
    np.testing.assert_array_equal(sampled.waypoints[:, 0], np.zeros(5))
    np.testing.assert_array_equal(sampled.waypoints[1, 1:], np.full(7, -1.0))
    np.testing.assert_array_equal(sampled.waypoints[3, 1:], np.ones(7))
    np.testing.assert_allclose(sampled.waypoints[4, 1:4] - sampled.waypoints[2, 1:4], sampled.postpush_delta)
    np.testing.assert_allclose(sampled.waypoints[4, 4:], sampled.waypoints[2, 4:])
    # The source samples geometric push distance from object half-width plus 5–10 cm.
    contact_side = np.sign(sampled.geometric_contact[1])
    assert abs(sampled.geometric_contact[1]) == pytest.approx(0.03 + 0.01)
    assert np.sign(sampled.geometric_postpush[1]) == -contact_side
    assert 0.12 <= abs(sampled.postpush_delta[1]) <= 0.17
    # GraspNet +X is the source approach direction; the pregrasp is behind it.
    assert 0.05 <= sampled.waypoints[2, 1] - sampled.waypoints[0, 1] <= 0.10


@pytest.mark.general
def test_push_sampler_is_seeded_and_samples_source_grasp_index():
    grasps = np.zeros((3, 17), dtype=np.float64)
    grasps[:, 4:13] = np.eye(3).reshape(1, 9)
    grasps[:, 13] = np.arange(3) * 0.02
    arguments = {
        "grasp_rows": grasps,
        "bbox_half_extents_m": [0.02, 0.03, 0.04],
        "grasp_angle_rad": 1.57,
        "seed": 31,
    }
    first = sample_push_gn_waypoints(**arguments)
    second = sample_push_gn_waypoints(**arguments)
    assert first.grasp_index == second.grasp_index
    np.testing.assert_array_equal(first.waypoints, second.waypoints)
