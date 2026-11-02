"""Object-relative key poses for paired synthetic demonstrations.

GraspNet annotations use a grasp-frame rotation and approach depth. The frame
conversion here follows the configuration documented by SynthICL's generator:
https://github.com/sNiper-Qian/synthicl-pseudo-demo. No physical success is
implied by a kinematically generated pose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

SYNTHICL_PSEUDO_DEMO_SOURCE = "https://github.com/sNiper-Qian/synthicl-pseudo-demo"
SYNTHICL_PSEUDO_DEMO_SHA = "131d22f9a4a8f066aa2fd7bee8d88118dcc0766a"
SYNTHICL_PICK_PLACE_TASK_ID = "pick_and_place_gn"
SYNTHICL_PUSH_TASK_ID = "push_gn"


def pick_place_task_key(
    *, object_id: str, target_object_id: str, grasp_index: int, backend: str, seed: int, source_scale: float = 1.0
) -> dict[str, str | int]:
    """Return the pinned provenance key for one RoboVerse pick-and-place pair.

    This describes the source task and the local two-pose tabletop variant. It
    does not claim behavioral equivalence with the upstream Isaac Sim pipeline.
    """
    for name, value in (("object_id", object_id), ("target_object_id", target_object_id)):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
            raise ValueError(f"{name} must be a non-empty path-safe identifier")
    if isinstance(grasp_index, bool) or not isinstance(grasp_index, int) or grasp_index < 0:
        raise ValueError("grasp_index must be a nonnegative integer")
    if not isinstance(backend, str) or backend not in {"mujoco", "genesis", "newton"}:
        raise ValueError(f"Unsupported backend: {backend}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(source_scale, bool) or not isinstance(source_scale, (int, float)) or not np.isfinite(source_scale):
        raise ValueError("The GraspNet source object scale must be exactly 1.0")
    if not np.isclose(source_scale, 1.0, atol=0, rtol=0):
        raise ValueError("The GraspNet source object scale must remain exactly 1.0")
    return {
        "source": SYNTHICL_PSEUDO_DEMO_SOURCE,
        "upstream_sha": SYNTHICL_PSEUDO_DEMO_SHA,
        "task_id": SYNTHICL_PICK_PLACE_TASK_ID,
        "level_or_variant": f"graspnet_index_{grasp_index}",
        "scene_id": f"two_pose_tabletop_v1/{object_id}_on_{target_object_id}",
        "embodiment": "franka",
        "backend": backend,
        "seed": seed,
    }


def push_task_key(
    *, object_id: str, grasp_index: int, backend: str, seed: int, source_scale: float = 1.0
) -> dict[str, str | int]:
    """Return the pinned provenance key for the single-object push variant."""
    if not isinstance(object_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", object_id):
        raise ValueError("object_id must be a non-empty path-safe identifier")
    if isinstance(grasp_index, bool) or not isinstance(grasp_index, int) or grasp_index < 0:
        raise ValueError("grasp_index must be a nonnegative integer")
    if not isinstance(backend, str) or backend not in {"mujoco", "genesis", "newton"}:
        raise ValueError(f"Unsupported backend: {backend}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(source_scale, bool) or not isinstance(source_scale, (int, float)) or not np.isfinite(source_scale):
        raise ValueError("The GraspNet source object scale must be exactly 1.0")
    if not np.isclose(source_scale, 1.0, atol=0, rtol=0):
        raise ValueError("The GraspNet source object scale must remain exactly 1.0")
    return {
        "source": SYNTHICL_PSEUDO_DEMO_SOURCE,
        "upstream_sha": SYNTHICL_PSEUDO_DEMO_SHA,
        "task_id": SYNTHICL_PUSH_TASK_ID,
        "level_or_variant": f"graspnet_index_{grasp_index}",
        "scene_id": f"single_object_ground_plane_v1/{object_id}",
        "embodiment": "franka",
        "backend": backend,
        "seed": seed,
    }


def object_on_top_relation(*, target_bounds_m, source_bounds_m, clearance_m: float = 0.0) -> np.ndarray:
    """Return a target-local source pose that places the source on the target top.

    Bounds are expressed in each object's unchanged mesh frame. The small
    positive clearance is explicit because the current source meshes use
    convex collision wrappers and this plan is rendered kinematically.
    """
    target_bounds = np.asarray(target_bounds_m, dtype=np.float64)
    source_bounds = np.asarray(source_bounds_m, dtype=np.float64)
    for bounds, name in ((target_bounds, "target_bounds_m"), (source_bounds, "source_bounds_m")):
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
            raise ValueError(f"{name} must be finite, ordered bounds with shape (2, 3)")
    if (
        isinstance(clearance_m, bool)
        or not isinstance(clearance_m, (int, float))
        or not np.isfinite(clearance_m)
        or clearance_m < 0
    ):
        raise ValueError("clearance_m must be finite and nonnegative")
    relation = np.eye(4, dtype=np.float64)
    relation[2, 3] = target_bounds[1, 2] - source_bounds[0, 2] + clearance_m
    return relation


_GRASPNET_TO_TCP = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
_CLOSE_CODE = np.full(7, -1.0)
_GRASP_CODE = np.ones(7)
_PUSH_ACTION_SEMANTICS = (
    "approach_pregrasp_open",
    "close_without_attachment",
    "approach_grasp_closed",
    "attach_object_to_hand",
    "approach_postpush_attached",
)


@dataclass(frozen=True)
class PushGNSample:
    """One deterministic five-waypoint sample of the source push skill."""

    waypoints: np.ndarray
    grasp_index: int
    geometric_prepush: np.ndarray
    geometric_contact: np.ndarray
    geometric_postpush: np.ndarray
    postpush_delta: np.ndarray
    action_semantics: tuple[str, ...] = _PUSH_ACTION_SEMANTICS


def _sample_range(rng: np.random.RandomState, value, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive number or a two-value range")
    if isinstance(value, (int, float)):
        bounds = (float(value), float(value))
    else:
        try:
            bounds = tuple(float(item) for item in value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a positive number or a two-value range") from None
        if len(bounds) != 2:
            raise ValueError(f"{name} must be a positive number or a two-value range")
    if not np.isfinite(bounds).all() or bounds[0] <= 0 or bounds[0] > bounds[1]:
        raise ValueError(f"{name} must be positive and ordered")
    return float(rng.uniform(*bounds))


def sample_push_gn_waypoints(
    *,
    grasp_rows,
    bbox_half_extents_m,
    grasp_angle_rad: float,
    seed: int,
    grasp_index: int | None = None,
    pregrasp_distance_m=(0.05, 0.10),
    postpush_offset_m=None,
    base_orientation_z_negative: bool = False,
    reference_position_m=(0.0, 0.0, 0.0),
) -> PushGNSample:
    """Sample the public ``push_gn`` waypoint and command contract.

    This independently implements the pinned source configuration behavior. It
    deliberately preserves the source's grasp-and-attach command sequence; it
    is not a physical pushing controller or success predicate.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(base_orientation_z_negative, bool):
        raise ValueError("base_orientation_z_negative must be boolean")
    extents = np.asarray(bbox_half_extents_m, dtype=np.float64)
    if extents.shape != (3,) or not np.isfinite(extents).all() or np.any(extents <= 0):
        raise ValueError("bbox_half_extents_m must contain three positive finite half extents")
    if (
        isinstance(grasp_angle_rad, bool)
        or not isinstance(grasp_angle_rad, (int, float))
        or not np.isfinite(grasp_angle_rad)
    ):
        raise ValueError("grasp_angle_rad must be finite")
    poses = np.asarray(grasp_rows, dtype=np.float64)
    if poses.ndim == 1:
        poses = poses[None, :]
    if poses.ndim != 2 or poses.shape[1] != 17 or not len(poses) or not np.isfinite(poses).all():
        raise ValueError("grasp_rows must be a nonempty finite array with shape (N, 17)")
    if np.any(poses[:, 3] < 0):
        raise ValueError("GraspNet depth must be nonnegative")
    rotations = poses[:, 4:13].reshape(-1, 3, 3)
    if not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-4) or not np.allclose(
        np.linalg.det(rotations), 1.0, atol=1e-4
    ):
        raise ValueError("GraspNet rotations must be proper orthonormal matrices")
    reference = np.asarray(reference_position_m, dtype=np.float64)
    if reference.shape != (3,) or not np.isfinite(reference).all():
        raise ValueError("reference_position_m must contain three finite values")
    if grasp_index is not None and (
        isinstance(grasp_index, bool) or not isinstance(grasp_index, int) or not 0 <= grasp_index < len(poses)
    ):
        raise ValueError("grasp_index is outside the GraspNet array")
    if postpush_offset_m is not None and not isinstance(postpush_offset_m, dict):
        fixed = np.asarray(postpush_offset_m, dtype=np.float64)
        if fixed.shape != (3,) or not np.isfinite(fixed).all():
            raise ValueError("postpush_offset_m must be a finite xyz vector or axis-range mapping")
    elif isinstance(postpush_offset_m, dict) and set(postpush_offset_m) != {"x", "y", "z"}:
        raise ValueError("postpush_offset_m must define x, y and z")

    rng = np.random.RandomState(seed)
    side_y = float(rng.choice([-extents[1] - 0.01, extents[1] + 0.01]))
    base_yaw = -1.57 - float(grasp_angle_rad)
    yaw = float(rng.choice([base_yaw, base_yaw + 1.57]))
    # Match the source sampler's fixed-range draws and its [3.14, 0, yaw] Euler pose.
    contact_xyz = np.array(
        [rng.uniform(0.0, 0.0), rng.uniform(side_y, side_y), rng.uniform(0.0, 0.0)], dtype=np.float64
    )
    euler = [3.14 - rng.uniform(0.0, 0.0), rng.uniform(0.0, 0.0), rng.uniform(yaw, yaw)]
    contact_rotation = Rotation.from_euler("xyz", euler).as_matrix()
    contact_quaternion = Rotation.from_matrix(contact_rotation).as_quat(scalar_first=True)
    contact = np.r_[contact_xyz, contact_quaternion]
    direction = contact_xyz / np.linalg.norm(contact_xyz)
    side_sign = side_y / abs(side_y)
    pre_factor = rng.uniform(side_y + 0.05 * side_sign, side_y + 0.10 * side_sign)
    geometric_prepush = np.r_[pre_factor * direction * side_sign, contact_quaternion]
    post_factor = rng.uniform(side_y + 0.05 * side_sign, side_y + 0.10 * side_sign)
    geometric_postpush = np.r_[-post_factor * direction * side_sign, contact_quaternion]
    geometric_delta = geometric_postpush[:3] - contact[:3]

    selected_index = int(rng.randint(len(poses))) if grasp_index is None else grasp_index
    row = poses[selected_index]
    grasp_rotation = row[4:13].reshape(3, 3).copy()
    grasp_translation = row[13:16].copy() - reference
    if base_orientation_z_negative:
        correction = Rotation.from_euler("z", np.pi / 2.0).as_matrix()
        grasp_translation = correction @ grasp_translation
        grasp_rotation = correction @ grasp_rotation
    grasp_translation += grasp_rotation[:, 0] * row[3]
    grasp_position = grasp_translation
    grasp_quaternion = Rotation.from_matrix(grasp_rotation @ _GRASPNET_TO_TCP).as_quat(scalar_first=True)
    pregrasp_distance = _sample_range(rng, pregrasp_distance_m, name="pregrasp_distance_m")
    pregrasp = np.r_[grasp_position - grasp_rotation[:, 0] * pregrasp_distance, grasp_quaternion]
    grasp = np.r_[grasp_position, grasp_quaternion]

    if postpush_offset_m is None:
        postpush_delta = geometric_delta
    elif isinstance(postpush_offset_m, dict):
        bounds = []
        for axis in ("x", "y", "z"):
            value = postpush_offset_m[axis]
            pair = tuple(float(item) for item in value)
            if len(pair) != 2 or not np.isfinite(pair).all() or pair[0] > pair[1]:
                raise ValueError(f"postpush_offset_m[{axis!r}] must be an ordered finite range")
            bounds.append(pair)
        postpush_delta = np.array([rng.uniform(*pair) for pair in bounds])
    else:
        postpush_delta = np.asarray(postpush_offset_m, dtype=np.float64)
    postpush = np.r_[grasp[:3] + postpush_delta, grasp_quaternion]
    waypoints = np.stack((
        np.r_[0.0, pregrasp],
        np.r_[0.0, _CLOSE_CODE],
        np.r_[0.0, grasp],
        np.r_[0.0, _GRASP_CODE],
        np.r_[0.0, postpush],
    ))
    return PushGNSample(
        waypoints=waypoints,
        grasp_index=selected_index,
        geometric_prepush=geometric_prepush,
        geometric_contact=contact,
        geometric_postpush=geometric_postpush,
        postpush_delta=np.asarray(postpush_delta, dtype=np.float64),
    )


def _transform(value, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError(f"{name} must be a finite homogeneous 4x4 transform")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise ValueError(f"{name} must have a proper rotation")
    return matrix


def graspnet_to_tcp(row, *, approach_distance: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
    """Return pregrasp and grasp TCP transforms in the original object-mesh frame.

    The mesh and annotation must retain the same scale and orientation. A
    caller applying an object-frame correction must transform both together.
    """
    row = np.asarray(row, dtype=np.float64)
    if row.shape != (17,) or not np.isfinite(row).all():
        raise ValueError("GraspNet row must contain 17 finite values")
    if row[3] < 0 or not np.isfinite(approach_distance) or approach_distance <= 0:
        raise ValueError("Grasp depth must be nonnegative and approach distance positive")
    grasp_frame = np.eye(4)
    grasp_frame[:3, :3] = row[4:13].reshape(3, 3)
    grasp_frame[:3, 3] = row[13:16]
    _transform(grasp_frame, "GraspNet pose")
    grasp = grasp_frame.copy()
    grasp[:3, 3] += grasp_frame[:3, 0] * row[3]
    grasp[:3, :3] = grasp_frame[:3, :3] @ _GRASPNET_TO_TCP
    pregrasp = grasp.copy()
    pregrasp[:3, 3] -= grasp_frame[:3, 0] * approach_distance
    return pregrasp, grasp


def resolve_pick_place(
    *, world_from_source, world_from_destination, destination_from_placed_source, pregrasp, grasp, lift: float = 0.10
) -> dict[str, np.ndarray]:
    """Resolve one shared object-relative task in one world configuration.

    Both scenes in a pair should use the same source-relative grasp and the
    same destination-relative final object pose. The returned tool poses are
    kinematic targets, not evidence of collision-free or successful grasps.
    """
    source = _transform(world_from_source, "world_from_source")
    destination = _transform(world_from_destination, "world_from_destination")
    relation = _transform(destination_from_placed_source, "destination_from_placed_source")
    pre = _transform(pregrasp, "pregrasp")
    grip = _transform(grasp, "grasp")
    if not np.isfinite(lift) or lift <= 0:
        raise ValueError("lift must be positive and finite")
    grasp_world = source @ grip
    lift_world = grasp_world.copy()
    lift_world[2, 3] += lift
    place_world = destination @ relation @ grip
    preplace_world = place_world.copy()
    preplace_world[2, 3] += lift
    return {
        "pregrasp": source @ pre,
        "grasp": grasp_world,
        "lift": lift_world,
        "preplace": preplace_world,
        "place": place_world,
    }
