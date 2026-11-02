# Copyright (c) 2026 RoboVerse contributors.
# SPDX-License-Identifier: MPL-2.0
#
# Adapted from Genie Sim benchmark checker semantics
# (https://github.com/AgibotTech/genie_sim), licensed under MPL-2.0.
# Changes: backend-independent RoboVerse checker graph compiler and state evaluator.
# Full license: roboverse_pack/tasks/icl/LICENSE.geniesim-benchmark
"""Compile Genie Sim task checker graphs into backend-independent goal evaluators.

The source task definitions use small action graphs: sequential lists, wait-any/all
groups, step limits and geometric checkers. This module preserves that structure
and evaluates it against state snapshots supplied by a RoboVerse task or collector.
It deliberately keeps simulator state acquisition outside the checker so the same
goal definition can be used with MuJoCo, Genesis and Newton.
"""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

import numpy as np


class CheckerStatus(str, Enum):
    """Current status of one source task graph."""

    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True)
class ObjectObservation:
    """World pose and optional local geometry/state for one rigid or articulated object.

    ``pose`` is a homogeneous 4x4 matrix. ``bounds`` contains local-frame
    ``(minimum, maximum)`` corners and is required by checkers that query an
    object's actual bounds. Articulated joint positions are in source order.
    ``subframes`` maps source child paths to their rigid transform in this
    object's local frame; it does not describe moving articulation links.
    """

    pose: np.ndarray
    bounds: np.ndarray | None = None
    joint_positions: np.ndarray | None = None
    subframes: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        pose = np.asarray(self.pose, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("object pose must be a finite 4x4 homogeneous matrix")
        object.__setattr__(self, "pose", pose)
        if self.bounds is not None:
            bounds = np.asarray(self.bounds, dtype=np.float64)
            if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] < bounds[0]):
                raise ValueError("object bounds must be finite local-frame (2, 3) minimum/maximum coordinates")
            object.__setattr__(self, "bounds", bounds)
        if self.joint_positions is not None:
            joints = np.asarray(self.joint_positions, dtype=np.float64).reshape(-1)
            if not np.isfinite(joints).all():
                raise ValueError("object joint positions must be finite")
            object.__setattr__(self, "joint_positions", joints)
        frames = {}
        for name, value in self.subframes.items():
            transform = np.asarray(value, dtype=np.float64)
            if not name or transform.shape != (4, 4) or not np.isfinite(transform).all():
                raise ValueError("object subframes must be named finite 4x4 transforms")
            frames[name] = transform
        object.__setattr__(self, "subframes", frames)


@dataclass(frozen=True)
class CheckerFrame:
    """One simulator observation used to advance a compiled checker graph."""

    step: int
    objects: Mapping[str, ObjectObservation]
    grippers: Mapping[str, np.ndarray] = field(default_factory=dict)
    robot_pose: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    floor_height: float = 0.0
    delta_time: float = 1.0 / 30.0
    vlm_scores: Mapping[str, float] = field(default_factory=dict)
    placeholders: dict[str, str] = field(default_factory=dict, repr=False)
    followed_objects: set[str] = field(default_factory=set, repr=False)
    picked_objects: set[str] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        if (
            self.step < 0
            or not math.isfinite(self.floor_height)
            or not math.isfinite(self.delta_time)
            or self.delta_time < 0
        ):
            raise ValueError("checker step, floor height or delta time is invalid")
        robot_pose = np.asarray(self.robot_pose, dtype=np.float64)
        if robot_pose.shape != (4, 4) or not np.isfinite(robot_pose).all():
            raise ValueError("robot pose must be a finite 4x4 homogeneous matrix")
        object.__setattr__(self, "robot_pose", robot_pose)
        normalized_grippers: dict[str, np.ndarray] = {}
        for name, pose in self.grippers.items():
            value = np.asarray(pose, dtype=np.float64)
            if value.shape != (4, 4) or not np.isfinite(value).all():
                raise ValueError(f"gripper {name!r} pose must be a finite 4x4 homogeneous matrix")
            normalized_grippers[name] = value
        object.__setattr__(self, "grippers", normalized_grippers)


@dataclass(frozen=True)
class CheckerResult:
    """Result returned after evaluating one observation."""

    status: CheckerStatus
    problem_id: str | None = None
    instruction: str | None = None


@dataclass
class _Node:
    op: str
    value: Any = None
    children: list[_Node] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)


_COMPOSITES = {"ActionList": "sequence", "ActionSetWaitAny": "any", "ActionSetWaitAll": "all"}
_LEAVES = {
    "Follow",
    "InBBox",
    "Inside",
    "LiftUp",
    "MixedRules",
    "Onfloor",
    "PickUpOnGripper",
    "PushPull",
    "RelativePosition",
    "StableGrasp",
    "Stack",
    "StepOut",
    "Upright",
    "VLM",
    # These appear in mixed-rule recipes in the source checker implementation.
    "Cover",
    "Ontop",
    "Timeout",
    "ActionWaitForTime",
}


def _items(value: Any, *, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a list of checker nodes")
    return value


def _compile_node(spec: Any, *, path: str = "root") -> _Node:
    if not isinstance(spec, dict) or len(spec) != 1:
        raise ValueError(f"{path} must be a one-key checker node, got {spec!r}")
    name, value = next(iter(spec.items()))
    if not isinstance(name, str):
        raise ValueError(f"{path} checker name must be a string")
    if name in _COMPOSITES:
        return _Node(
            _COMPOSITES[name],
            children=[
                _compile_node(child, path=f"{path}.{name}[{i}]") for i, child in enumerate(_items(value, path=path))
            ],
        )
    if name.startswith("ActionSetWaitSome"):
        suffix = name.removeprefix("ActionSetWaitSome").lstrip("_")
        minimum = int(suffix) if suffix.isdigit() else 1
        children = [
            _compile_node(child, path=f"{path}.{name}[{i}]") for i, child in enumerate(_items(value, path=path))
        ]
        if not 1 <= minimum <= len(children):
            raise ValueError(f"{path} has invalid wait-some count {minimum} for {len(children)} children")
        return _Node("some", value=minimum, children=children)
    if name in _LEAVES:
        return _Node(name, value=value)
    raise ValueError(f"Unsupported Genie Sim checker {name!r} at {path}")


def compile_problem_graph(problem: Mapping[str, Any]) -> _Node:
    """Compile one imported ``problems.json`` entry, preserving source graph order."""
    acts = problem.get("Acts")
    if not isinstance(acts, list) or not acts or not isinstance(acts[0], dict):
        raise ValueError("Genie Sim problem must contain a non-empty Acts list")
    # Genie Sim's own action parser selects Acts[0]. Preserve that selection.
    return _compile_node(acts[0], path="Acts[0]")


def available_problem_ids(instance: Mapping[str, Any]) -> list[str]:
    """Return the numeric source problem keys in stable episode-selection order."""
    graphs = instance.get("checker_graphs", {})
    if not isinstance(graphs, dict):
        raise ValueError("Genie Sim instance checker_graphs must be a mapping")
    return sorted((key for key in graphs if re.fullmatch(r"problem\d+", key)), key=lambda key: int(key[7:]))


def _parse_params(value: Any, *, name: str, count: int | None = None) -> list[str]:
    if not isinstance(value, str):
        raise ValueError(f"{name} checker parameters must be a pipe-delimited string")
    values = [part.strip() for part in value.split("|")]
    if count is not None and len(values) < count:
        raise ValueError(f"{name} checker expects at least {count} parameters, got {len(values)}")
    return values


def _vector(value: str, *, name: str, size: int = 3) -> np.ndarray:
    try:
        result = np.asarray(ast.literal_eval(value), dtype=np.float64).reshape(-1)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Invalid {name} vector: {value!r}") from exc
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _expand_placeholder(frame: CheckerFrame, object_name: str) -> str:
    match = re.fullmatch(r"\{@([^}]+)\}", object_name.strip())
    if match is not None:
        placeholder = match.group(1)
        try:
            return frame.placeholders[placeholder]
        except KeyError as exc:
            raise KeyError(f"Checker references unresolved placeholder {object_name!r}") from exc
    return object_name.strip()


def _resolve_object_name(frame: CheckerFrame, object_name: str) -> str:
    object_name = _expand_placeholder(frame, object_name)
    if object_name in frame.objects:
        return object_name
    match = re.fullmatch(r"/(?:Workspace|World)/[Oo]bjects/([^/]+)(?:/(.+))?", object_name)
    if match and match.group(1) in frame.objects:
        return match.group(1)
    if "/" in object_name and object_name.rsplit("/", 1)[-1] in frame.objects:
        return object_name.rsplit("/", 1)[-1]
    return object_name


def _pose(frame: CheckerFrame, object_name: str) -> np.ndarray:
    source_name = _expand_placeholder(frame, object_name)
    object_name = _resolve_object_name(frame, source_name)
    try:
        obj = frame.objects[object_name]
    except KeyError as exc:
        raise KeyError(f"Checker references object {object_name!r}, which is absent from this scene") from exc
    match = re.fullmatch(r"/(?:Workspace|World)/[Oo]bjects/([^/]+)/(.+)", source_name)
    if match:
        child = match.group(2)
        try:
            return obj.pose @ obj.subframes[child]
        except KeyError as exc:
            raise KeyError(f"Checker object {object_name!r} has no verified source subframe {child!r}") from exc
    return obj.pose


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def _pose_from_state_row(row: np.ndarray) -> np.ndarray:
    """Convert MetaSim ``[xyz, quaternion_wxyz, ...]`` state into a matrix."""
    row = np.asarray(row, dtype=np.float64).reshape(-1)
    if row.size < 7 or not np.isfinite(row[:7]).all():
        raise ValueError("MetaSim root/body state must contain finite position and quaternion values")
    quaternion = row[3:7]
    norm = np.linalg.norm(quaternion)
    if norm < 1e-8:
        raise ValueError("MetaSim root/body state contains a zero quaternion")
    w, x, y, z = quaternion / norm
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = row[:3]
    return pose


def checker_frame_from_states(
    states: Any,
    *,
    robot_name: str,
    step: int,
    object_bounds: Mapping[str, np.ndarray] | None = None,
    env_index: int = 0,
    object_joint_indices: Mapping[str, list[int] | tuple[int, ...]] | None = None,
    object_subframes: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    static_objects: Mapping[str, ObjectObservation] | None = None,
    gripper_links: Mapping[str, str] | None = None,
    gripper_link_transforms: Mapping[str, np.ndarray] | None = None,
    floor_height: float = 0.0,
    delta_time: float = 1.0 / 30.0,
    vlm_scores: Mapping[str, float] | None = None,
) -> CheckerFrame:
    """Build one checker input from MetaSim's backend-neutral tensor state.

    Robot link states must be available through ``RobotState.body_names`` and
    ``RobotState.body_state``. The simulator backend must use the documented
    position/quaternion/linear/angular velocity ordering. Object local bounds
    are optional except for checkers such as ``Inside``, ``Ontop`` and ``Cover``.
    """
    if env_index < 0:
        raise ValueError("env_index must be non-negative")
    object_bounds = object_bounds or {}
    object_joint_indices = object_joint_indices or {}
    object_subframes = object_subframes or {}
    objects: dict[str, ObjectObservation] = {}
    for name, state in states.objects.items():
        root = _as_numpy(state.root_state)
        if root.ndim != 2 or env_index >= root.shape[0]:
            raise ValueError(f"No root state for object {name!r} at environment {env_index}")
        joint_positions = None
        if state.joint_pos is not None:
            joints = _as_numpy(state.joint_pos)
            if joints.ndim != 2 or env_index >= joints.shape[0]:
                raise ValueError(f"No joint state for object {name!r} at environment {env_index}")
            indices = object_joint_indices.get(name)
            if indices is None:
                joint_positions = joints[env_index]
            else:
                if any(index < 0 or index >= joints.shape[1] for index in indices):
                    raise ValueError(f"Invalid source joint index mapping for articulated object {name!r}")
                joint_positions = joints[env_index, list(indices)]
        objects[name] = ObjectObservation(
            pose=_pose_from_state_row(root[env_index]),
            bounds=object_bounds.get(name),
            joint_positions=joint_positions,
            subframes=object_subframes.get(name, {}),
        )

    for name, obj in (static_objects or {}).items():
        if name in objects or not isinstance(obj, ObjectObservation):
            raise ValueError(f"Duplicate or invalid static checker object {name!r}")
        objects[name] = obj

    try:
        robot_state = states.robots[robot_name]
    except (AttributeError, KeyError) as exc:
        raise KeyError(f"MetaSim state does not contain robot {robot_name!r}") from exc
    robot_root = _as_numpy(robot_state.root_state)
    if robot_root.ndim != 2 or env_index >= robot_root.shape[0]:
        raise ValueError(f"No root state for robot {robot_name!r} at environment {env_index}")
    if robot_state.body_state is None or not robot_state.body_names:
        raise RuntimeError(
            f"Backend did not expose per-link body_state for robot {robot_name!r}; "
            "implement body-state output before using gripper-based Genie Sim checkers"
        )
    body_states = _as_numpy(robot_state.body_state)
    if (
        body_states.ndim != 3
        or env_index >= body_states.shape[0]
        or body_states.shape[1] != len(robot_state.body_names)
    ):
        raise ValueError(f"Malformed per-link body state for robot {robot_name!r}")

    configured_links = gripper_links or {
        "left_gripper": "gripper_l_center_link",
        "right_gripper": "gripper_r_center_link",
    }
    configured_transforms = gripper_link_transforms or {}
    grippers: dict[str, np.ndarray] = {}
    for gripper_name, link_name in configured_links.items():
        matches = [
            index for index, body_name in enumerate(robot_state.body_names) if body_name.split("/")[-1] == link_name
        ]
        if len(matches) != 1:
            raise KeyError(f"Expected one robot link named {link_name!r} for {gripper_name}, found {len(matches)}")
        pose = _pose_from_state_row(body_states[env_index, matches[0]])
        if gripper_name in configured_transforms:
            offset = np.asarray(configured_transforms[gripper_name], dtype=np.float64)
            if offset.shape != (4, 4) or not np.isfinite(offset).all():
                raise ValueError(f"Invalid local gripper transform for {gripper_name!r}")
            pose = pose @ offset
        grippers[gripper_name] = pose

    return CheckerFrame(
        step=step,
        objects=objects,
        grippers=grippers,
        robot_pose=_pose_from_state_row(robot_root[env_index]),
        floor_height=floor_height,
        delta_time=delta_time,
        vlm_scores=vlm_scores or {},
    )


def _world_aabb(
    frame: CheckerFrame, object_name: str, *, local_size: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    source_name = object_name
    object_name = _resolve_object_name(frame, source_name)
    obj = frame.objects.get(object_name)
    if obj is None:
        raise KeyError(f"Checker references object {object_name!r}, which is absent from this scene")
    if local_size is None:
        if obj.bounds is None:
            raise ValueError(f"Checker needs local geometry bounds for object {object_name!r}")
        lower, upper = obj.bounds
    else:
        lower, upper = -local_size / 2, local_size / 2
    corners = np.array([
        [x, y, z] for x in (lower[0], upper[0]) for y in (lower[1], upper[1]) for z in (lower[2], upper[2])
    ])
    pose = _pose(frame, source_name) if local_size is not None else obj.pose
    world = corners @ pose[:3, :3].T + pose[:3, 3]
    return world.min(axis=0), world.max(axis=0)


def _inside(point: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> bool:
    return bool(np.all(point >= lower) and np.all(point <= upper))


def _consecutive(node: _Node, condition: bool, required: int = 2) -> bool:
    node.state["frames"] = node.state.get("frames", 0) + 1 if condition else 0
    return node.state["frames"] >= required


def _split_bbox(value: str, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    values = value.split(",")
    try:
        result = np.asarray([float(item.strip()) for item in values], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"Invalid {name} coordinates: {value!r}") from exc
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite values")
    return result, result


def _eval_leaf(node: _Node, frame: CheckerFrame) -> CheckerStatus:
    if "terminal" in node.state:
        return node.state["terminal"]
    name, value = node.op, node.value
    status = CheckerStatus.RUNNING
    if name in {"StepOut", "Timeout"}:
        if "start_step" not in node.state:
            node.state["start_step"] = frame.step
        if name == "StepOut":
            try:
                limit = int(value)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid StepOut step limit: {value!r}") from exc
            if limit < 0:
                raise ValueError("StepOut step limit cannot be negative")
            if frame.step - node.state["start_step"] > limit:
                status = CheckerStatus.FAILURE
        else:
            try:
                limit = float(value)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid Timeout duration: {value!r}") from exc
            node.state["elapsed"] = node.state.get("elapsed", 0.0) + frame.delta_time
            if node.state["elapsed"] >= limit:
                status = CheckerStatus.FAILURE
    elif name == "ActionWaitForTime":
        node.state["elapsed"] = node.state.get("elapsed", 0.0) + frame.delta_time
        if node.state["elapsed"] >= float(value):
            status = CheckerStatus.SUCCESS
    elif name == "Inside":
        active, passive, scale = _parse_params(value, name=name, count=3)[:3]
        scale_value = float(scale)
        if not math.isfinite(scale_value) or scale_value <= 0:
            raise ValueError("Inside scale must be positive")
        lower, upper = _world_aabb(frame, passive)
        center = (lower + upper) / 2
        half = (upper - lower) * scale_value / 2
        if _consecutive(node, _inside(_pose(frame, active)[:3, 3], center - half, center + half)):
            status = CheckerStatus.SUCCESS
    elif name == "InBBox":
        object_name, center_text, size_text = _parse_params(value, name=name, count=3)[:3]
        center, _ = _split_bbox(center_text, name="InBBox center")
        size, _ = _split_bbox(size_text, name="InBBox size")
        if np.any(size < 0):
            raise ValueError("InBBox size cannot be negative")
        pos = _pose(frame, object_name)[:3, 3]
        if _consecutive(node, _inside(pos, center - size / 2, center + size / 2)):
            status = CheckerStatus.SUCCESS
    elif name == "Follow":
        object_names_text, size_text, gripper_name = _parse_params(value, name=name, count=3)[:3]
        size = _vector(size_text, name="Follow box size")
        if np.any(size < 0):
            raise ValueError("Follow box size cannot be negative")
        try:
            gripper = frame.grippers[gripper_name]
        except KeyError as exc:
            raise KeyError(f"Checker references unavailable gripper {gripper_name!r}") from exc
        object_names = [_expand_placeholder(frame, item) for item in object_names_text.split(",") if item.strip()]
        candidates = [item for item in object_names if item not in frame.followed_objects]
        if not candidates and object_names:
            candidates = object_names[:1]
        matched = None
        for object_name in candidates:
            if _resolve_object_name(frame, object_name) not in frame.objects:
                continue
            lower, upper = _world_aabb(frame, object_name, local_size=size)
            if _inside(gripper[:3, 3], lower, upper):
                matched = object_name
                break
        if matched is not None:
            node.state["matched_object"] = matched
            frame.placeholders["placeholder_str1"] = matched
            frame.followed_objects.add(matched)
            status = CheckerStatus.SUCCESS
    elif name in {"PickUpOnGripper", "LiftUp"}:
        params = _parse_params(value, name=name, count=2)
        if name == "LiftUp":
            object_names = [_expand_placeholder(frame, params[0])]
            gripper_name = None
            lift = float(params[1])
        else:
            object_names = [_expand_placeholder(frame, item) for item in params[0].split(",") if item.strip()]
            gripper_name = params[1]
            lift = 0.02
            try:
                gripper_pos = frame.grippers[gripper_name][:3, 3]
            except KeyError as exc:
                raise KeyError(f"Checker references unavailable gripper {gripper_name!r}") from exc

        initial_z = node.state.setdefault("initial_z_by_object", {})
        candidates = object_names
        is_placeholder = bool(re.fullmatch(r"\{@[^}]+\}", params[0].strip()))
        if name == "PickUpOnGripper" and not is_placeholder:
            remaining = [item for item in candidates if item not in frame.picked_objects]
            candidates = remaining or candidates[:1]
        matched = None
        for object_name in candidates:
            if _resolve_object_name(frame, object_name) not in frame.objects:
                continue
            obj_pose = _pose(frame, object_name)
            start_z = initial_z.setdefault(object_name, float(obj_pose[2, 3]))
            lifted = (obj_pose[2, 3] - start_z >= lift) if name == "LiftUp" else (obj_pose[2, 3] - start_z > lift)
            if name == "LiftUp":
                passed = lifted
            else:
                passed = lifted and np.linalg.norm(gripper_pos - obj_pose[:3, 3]) < 0.2
            if passed:
                matched = object_name
                break
        if matched is not None:
            node.state["matched_object"] = matched
            if name == "PickUpOnGripper":
                frame.placeholders["placeholder_str1"] = matched
                frame.picked_objects.add(matched)
            status = CheckerStatus.SUCCESS
    elif name == "Onfloor":
        object_name, ref_height = _parse_params(value, name=name, count=2)[:2]
        # In Genie Sim this action is an episode failure trigger, not a success condition.
        if abs(_pose(frame, object_name)[2, 3] - float(ref_height)) < 0.3:
            status = CheckerStatus.FAILURE
    elif name == "Upright":
        params = _parse_params(value, name=name, count=2)
        object_name, threshold = params[:2]
        allow_flipped = len(params) > 2 and params[2].lower() == "true"
        world_y = _pose(frame, object_name)[:3, :3] @ np.array([0.0, 1.0, 0.0])
        dot = float(world_y[2])
        limit = math.cos(math.radians(float(threshold)))
        if dot >= limit or (allow_flipped and dot <= -limit):
            status = CheckerStatus.SUCCESS
    elif name == "RelativePosition":
        params = _parse_params(value, name=name, count=3)
        a, b, relation = params[:3]
        relation = relation.lower()
        # The source checker compares centers of the world AABBs, rather than
        # object-frame origins.  The authored asset pivot can be far away from
        # its visual geometry (for example a laptop or a mouse).
        lower_a, upper_a = _world_aabb(frame, a)
        lower_b, upper_b = _world_aabb(frame, b)
        delta = (lower_a + upper_a - lower_b - upper_b) / 2
        # GenieSim applies custom thresholds to aligned_* relations only; the
        # four directional relations always use a fixed five-centimeter margin.
        threshold = float(params[3]) if relation.startswith("aligned_") and len(params) > 3 else 0.05
        rotation = frame.robot_pose[:3, :3]
        projection = {
            "leftof": float(delta @ (rotation @ [0.0, 1.0, 0.0])),
            "rightof": float(delta @ (rotation @ [0.0, -1.0, 0.0])),
            "topof": float(delta @ (rotation @ [1.0, 0.0, 0.0])),
            "bottomof": float(delta @ (rotation @ [-1.0, 0.0, 0.0])),
            "aligned_x": abs(float(delta @ (rotation @ [1.0, 0.0, 0.0]))),
            "aligned_y": abs(float(delta @ (rotation @ [0.0, 1.0, 0.0]))),
            "aligned_z": abs(float(delta[2])),
        }.get(relation)
        if projection is None:
            raise ValueError(f"Unsupported RelativePosition relation: {relation!r}")
        passed = projection < threshold if relation.startswith("aligned_") else projection > threshold
        if _consecutive(node, passed):
            status = CheckerStatus.SUCCESS
    elif name == "Stack":
        objects_text, threshold_text = _parse_params(value, name=name, count=2)[:2]
        try:
            thresholds = np.asarray(ast.literal_eval(threshold_text), dtype=np.float64)
        except (ValueError, SyntaxError) as exc:
            raise ValueError("Invalid Stack XY tolerance") from exc
        objects = [item.strip() for item in objects_text.strip().strip("[]").split(",") if item.strip()]
        if not isinstance(objects, list) or len(objects) < 2 or thresholds.shape != (2,):
            raise ValueError("Stack requires at least two objects and an [x, y] tolerance")
        centers = np.asarray([_pose(frame, obj)[:2, 3] for obj in objects])
        passed = bool(np.all(np.abs(centers[1:] - centers[0]) <= thresholds))
        if _consecutive(node, passed):
            status = CheckerStatus.SUCCESS
    elif name == "PushPull":
        params = _parse_params(value, name=name, count=3)
        object_name, min_value, max_value = params[:3]
        joint_index = int(params[3]) if len(params) > 3 else 0
        obj = frame.objects.get(object_name)
        if obj is None and "/" in object_name:
            obj = frame.objects.get(object_name.rsplit("/", 1)[-1])
        if obj is None or obj.joint_positions is None:
            raise ValueError(f"PushPull requires articulated joint state for object {object_name!r}")
        if not 0 <= joint_index < len(obj.joint_positions):
            raise ValueError(f"PushPull joint index {joint_index} is unavailable on {object_name!r}")
        passed = float(min_value) <= obj.joint_positions[joint_index] <= float(max_value)
        # The source checker counts successful samples without resetting on misses.
        node.state["frames"] = node.state.get("frames", 0) + int(passed)
        if node.state["frames"] > 1:
            status = CheckerStatus.SUCCESS
    elif name in {"Ontop", "Cover"}:
        active, passive = _parse_params(value, name=name, count=2)[:2]
        lower_a, upper_a = _world_aabb(frame, active)
        lower_b, upper_b = _world_aabb(frame, passive)
        inter_x = max(0.0, min(upper_a[0], upper_b[0]) - max(lower_a[0], lower_b[0]))
        inter_y = max(0.0, min(upper_a[1], upper_b[1]) - max(lower_a[1], lower_b[1]))
        intersection = inter_x * inter_y
        dx, dy, dz = upper_a - lower_a
        area_a = max(dx * dy, dx * dz, dy * dz)
        passed = False
        if name == "Ontop":
            passed = abs(lower_a[2] - upper_b[2]) <= 0.02 and area_a > 0 and intersection / area_a >= 0.5
        else:
            lower_b2, upper_b2 = _world_aabb(frame, passive)
            dx_b, dy_b, dz_b = upper_b2 - lower_b2
            area_b = max(dx_b * dy_b, dx_b * dz_b, dy_b * dz_b)
            passed = (
                (lower_a[2] + upper_a[2]) / 2 - upper_b[2] <= 0.002
                and min(area_a, area_b) > 0
                and intersection / min(area_a, area_b) >= 0.5
            )
        if passed:
            status = CheckerStatus.SUCCESS
    elif name == "StableGrasp":
        params = _parse_params(value, name=name, count=2)
        object_name, gripper_name = params[:2]
        distance_limit = float(params[2]) if len(params) > 2 else 0.1
        position_limit = float(params[3]) if len(params) > 3 else 0.02
        rotation_limit = float(params[4]) if len(params) > 4 else 0.1
        try:
            gripper_pose = frame.grippers[gripper_name]
        except KeyError as exc:
            raise KeyError(f"Checker references unavailable gripper {gripper_name!r}") from exc
        obj_pose = _pose(frame, object_name)
        prev_gripper = node.state.get("previous_gripper")
        prev_object = node.state.get("previous_object")
        stable = False
        if prev_gripper is not None and prev_object is not None:
            dg = gripper_pose[:3, 3] - prev_gripper[:3, 3]
            do = obj_pose[:3, 3] - prev_object[:3, 3]
            rg = gripper_pose[:3, :3] @ prev_gripper[:3, :3].T
            ro = obj_pose[:3, :3] @ prev_object[:3, :3].T
            rot_delta = ro @ rg.T
            angle = math.acos(float(np.clip((np.trace(rot_delta) - 1) / 2, -1.0, 1.0)))
            stable = (
                np.linalg.norm(gripper_pose[:3, 3] - obj_pose[:3, 3]) < distance_limit
                and np.linalg.norm(dg) >= 0.05
                and np.linalg.norm(do) >= 0.05
                and np.linalg.norm(do - dg) < position_limit
                and angle < rotation_limit
            )
        node.state["previous_gripper"] = gripper_pose.copy()
        node.state["previous_object"] = obj_pose.copy()
        if _consecutive(node, stable, required=2):
            status = CheckerStatus.SUCCESS
    elif name == "VLM":
        task_id, interval = _parse_params(value, name=name, count=2)[:2]
        interval_value = int(interval)
        if interval_value <= 0:
            raise ValueError("VLM scoring interval must be positive")
        node.state["frames"] = node.state.get("frames", 0) + 1
        if node.state["frames"] % interval_value == 0:
            if task_id not in frame.vlm_scores:
                raise RuntimeError(f"VLM checker {task_id!r} needs a score in CheckerFrame.vlm_scores")
            if float(frame.vlm_scores[task_id]) >= 1.0:
                status = CheckerStatus.SUCCESS
    elif name == "MixedRules":
        return _eval_mixed(node, frame)
    else:
        raise ValueError(f"No evaluator implemented for Genie Sim checker {name!r}")
    if status != CheckerStatus.RUNNING:
        node.state["terminal"] = status
    return status


def _eval_mixed(node: _Node, frame: CheckerFrame) -> CheckerStatus:
    value = node.value
    if not isinstance(value, Mapping) or not isinstance(value.get("rules"), list):
        raise ValueError("MixedRules must contain a rules list")
    interval = int(value.get("check_interval", 1))
    if interval < 1:
        raise ValueError("MixedRules check_interval must be positive")
    node.state["ticks"] = node.state.get("ticks", 0) + 1
    mode = str(value.get("mode", "or")).lower()
    if mode not in {"and", "or"}:
        raise ValueError(f"Unsupported MixedRules mode: {mode!r}")
    if node.state["ticks"] % interval:
        return CheckerStatus.RUNNING
    if not node.children:
        node.children = [
            _compile_node({rule["name"]: rule["params"]}, path="MixedRules.rules") for rule in value["rules"]
        ]
    if mode == "and":
        for child in node.children:
            child.state.pop("terminal", None)
        results = [_eval_node(child, frame) for child in node.children]
    else:
        passed = node.state.setdefault("sub_results", [False] * len(node.children))
        for index, child in enumerate(node.children):
            if not passed[index]:
                result = _eval_node(child, frame)
                if result == CheckerStatus.FAILURE:
                    return result
                passed[index] = result == CheckerStatus.SUCCESS
        return CheckerStatus.SUCCESS if all(passed) else CheckerStatus.RUNNING
    if any(result == CheckerStatus.FAILURE for result in results):
        return CheckerStatus.FAILURE
    if mode == "and":
        passed = all(result == CheckerStatus.SUCCESS for result in results)
        if not passed:
            # The upstream AND mode reevaluates every rule and can revoke a prior pass.
            for child in node.children:
                child.state.pop("terminal", None)
        return CheckerStatus.SUCCESS if passed else CheckerStatus.RUNNING
    raise AssertionError("MixedRules mode dispatch failed")


def _eval_node(node: _Node, frame: CheckerFrame) -> CheckerStatus:
    if "terminal" in node.state:
        return node.state["terminal"]
    if node.op in _LEAVES:
        return _eval_leaf(node, frame)
    if node.op == "sequence":
        index = node.state.get("index", 0)
        while index < len(node.children):
            result = _eval_node(node.children[index], frame)
            if result == CheckerStatus.FAILURE:
                node.state["terminal"] = result
                return result
            if result == CheckerStatus.RUNNING:
                node.state["index"] = index
                return result
            index += 1
        node.state["index"] = index
        result = CheckerStatus.SUCCESS
    elif node.op in {"any", "all", "some"}:
        statuses = [_eval_node(child, frame) for child in node.children]
        if node.op == "any":
            result = (
                CheckerStatus.FAILURE
                if CheckerStatus.FAILURE in statuses
                else CheckerStatus.SUCCESS
                if CheckerStatus.SUCCESS in statuses
                else CheckerStatus.RUNNING
            )
        elif node.op == "all":
            result = (
                CheckerStatus.FAILURE
                if CheckerStatus.FAILURE in statuses
                else CheckerStatus.SUCCESS
                if all(item == CheckerStatus.SUCCESS for item in statuses)
                else CheckerStatus.RUNNING
            )
        else:
            result = (
                CheckerStatus.FAILURE
                if CheckerStatus.FAILURE in statuses
                else CheckerStatus.SUCCESS
                if statuses.count(CheckerStatus.SUCCESS) >= node.value
                else CheckerStatus.RUNNING
            )
    else:
        raise ValueError(f"Unsupported compiled checker node {node.op!r}")
    if result != CheckerStatus.RUNNING:
        node.state["terminal"] = result
    return result


class GenieSimChecker:
    """Stateful evaluator for one imported Genie Sim problem definition."""

    def __init__(self, graph: _Node, *, problem_id: str | None = None, instruction: str | None = None):
        self._graph = graph
        self.problem_id = problem_id
        self.instruction = instruction
        self._last_step = -1
        self._status = CheckerStatus.RUNNING
        self._placeholders: dict[str, str] = {}
        self._followed_objects: set[str] = set()
        self._picked_objects: set[str] = set()

    @classmethod
    def from_instance(cls, instance: Mapping[str, Any], *, problem_index: int = 0) -> GenieSimChecker:
        """Select the source problem paired with ``instructions[problem_index]`` and compile it."""
        if problem_index < 0:
            raise ValueError("problem_index must be non-negative")
        problem_ids = available_problem_ids(instance)
        if not problem_ids:
            raise ValueError(f"Genie Sim instance {instance.get('task')}/{instance.get('instance_id')} has no problems")
        if problem_index >= len(problem_ids):
            raise IndexError(f"problem_index {problem_index} exceeds available problems {problem_ids}")
        problem_id = problem_ids[problem_index]
        problem = instance["checker_graphs"][problem_id]
        instructions = instance.get("instructions", {}).get("instructions", [])
        instruction = None
        if isinstance(instructions, list) and problem_index < len(instructions):
            item = instructions[problem_index]
            if isinstance(item, Mapping):
                instruction = item.get("instruction")
        return cls(compile_problem_graph(problem), problem_id=problem_id, instruction=instruction)

    @classmethod
    def from_subtask_steps(cls, steps: list[Mapping[str, Any]]) -> GenieSimChecker:
        """Compile Genie Sim's human-language subtask sequence and per-step checkers."""
        if not steps:
            raise ValueError("subtask_steps must not be empty")
        stages = []
        for index, step in enumerate(steps):
            if not isinstance(step, Mapping):
                raise ValueError(f"subtask_steps[{index}] must be a mapping")
            checker = step.get("checker")
            if checker == "passthrough" or checker is None:
                raise ValueError(f"subtask_steps[{index}] has no executable checker")
            limit = step.get("step_out")
            stage = _compile_node(checker, path=f"subtask_steps[{index}].checker")
            if limit is not None:
                stage = _Node("any", children=[stage, _compile_node({"StepOut": limit})])
            stages.append(stage)
        return cls(_Node("sequence", children=stages))

    def reset(self) -> None:
        """Clear all temporal state and return the checker to its initial state."""
        self._graph = _reset_node(self._graph)
        self._last_step = -1
        self._status = CheckerStatus.RUNNING
        self._placeholders.clear()
        self._followed_objects.clear()
        self._picked_objects.clear()

    def step(self, frame: CheckerFrame) -> CheckerResult:
        """Advance the checker with one monotonically increasing simulator snapshot."""
        if frame.step <= self._last_step:
            raise ValueError(f"Checker frames must increase monotonically; got {frame.step} after {self._last_step}")
        self._last_step = frame.step
        if self._status == CheckerStatus.RUNNING:
            runtime_frame = replace(
                frame,
                placeholders=dict(self._placeholders),
                followed_objects=set(self._followed_objects),
                picked_objects=set(self._picked_objects),
            )
            self._status = _eval_node(self._graph, runtime_frame)
            self._placeholders = runtime_frame.placeholders
            self._followed_objects = runtime_frame.followed_objects
            self._picked_objects = runtime_frame.picked_objects
        return CheckerResult(self._status, self.problem_id, self.instruction)


def _reset_node(node: _Node) -> _Node:
    node.state.clear()
    node.children = [_reset_node(child) for child in node.children]
    return node


def checker_graph_vocabulary(graph: Mapping[str, Any]) -> set[str]:
    """Return all checker/operator names in an imported graph for migration audits."""
    names: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "name" and isinstance(child, str) and child in _LEAVES:
                    names.add(child)
                elif key.startswith("ActionSet") or key in {*_LEAVES, "ActionList"}:
                    names.add(key)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(graph)
    return names
