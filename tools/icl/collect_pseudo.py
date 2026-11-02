"""Render small paired kinematic SynthICL task demonstrations."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from metasim.constants import PhysicStateType
from roboverse_learn.il.icl.waypoints import (
    graspnet_to_tcp,
    object_on_top_relation,
    pick_place_task_key,
    push_task_key,
    resolve_pick_place,
    sample_push_gn_waypoints,
)
from roboverse_pack.tasks.icl.control import FrankaCartesianController
from roboverse_pack.tasks.icl.push_to_region import ASSET_ROOT, PushToRegionTask, build_scenario
from tools.icl.collect import CAMERA_ROLES, CAMERAS
from tools.icl.prepare_assets import _sha256


def _pose(position, rotation=None) -> np.ndarray:
    value = np.eye(4)
    value[:3, 3] = position
    if rotation is not None:
        value[:3, :3] = rotation.as_matrix()
    return value


def _paired_scenario(simulator: str, object_id: str, target_object_id: str, size: int, source_xy, asset_root):
    """Build a two-object scene for one kinematic pick-and-place trajectory."""
    scenario = build_scenario(
        simulator=simulator,
        object_id=object_id,
        image_size=size,
        object_xy=tuple(source_xy),
        asset_root=asset_root,
    )
    target_scenario = build_scenario(
        simulator=simulator,
        object_id=target_object_id,
        image_size=size,
        asset_root=asset_root,
    )
    target = copy.deepcopy(target_scenario.objects[0])
    target.name = "object2"
    target.physics = PhysicStateType.GEOM
    target.fix_base_link = True
    target.default_position = (0.62, 0.12, target.default_position[2])
    scenario.objects = [scenario.objects[0], target]
    return scenario


def _trajectory(
    simulator: str, object_id: str, target_object_id: str, size: int, xy, row: np.ndarray, asset_root
) -> tuple[dict, dict]:
    """Render one object-on-object trajectory by setting exact states; no physics is simulated."""
    scenario = _paired_scenario(simulator, object_id, target_object_id, size, xy, asset_root)
    env = PushToRegionTask(scenario, goal="left")
    try:
        obs, _ = env.reset()
        names = env.handler.get_joint_names("franka", sort=True)
        bare_names = [name.split("/")[-1] for name in names]
        controller = FrankaCartesianController(scenario.robots[0], names)
        q = obs.robots["franka"].joint_pos[0].detach().cpu().numpy().copy()
        object_root = obs.objects["object"].root_state[0].detach().cpu().numpy()
        source = _pose(object_root[:3], Rotation.from_quat(object_root[3:7], scalar_first=True))
        target_root = obs.objects["object2"].root_state[0].detach().cpu().numpy()
        destination = _pose(target_root[:3], Rotation.from_quat(target_root[3:7], scalar_first=True))
        asset_root = Path(scenario.objects[0].urdf_path).parent
        target_asset_root = Path(scenario.objects[1].urdf_path).parent
        source_info = json.loads((asset_root / "provenance.json").read_text(encoding="utf-8"))
        target_info = json.loads((target_asset_root / "provenance.json").read_text(encoding="utf-8"))
        relation = object_on_top_relation(
            target_bounds_m=target_info["bounds_m"],
            source_bounds_m=source_info["bounds_m"],
        )
        pregrasp, grasp = graspnet_to_tcp(row, approach_distance=0.05)
        plan = resolve_pick_place(
            world_from_source=source,
            world_from_destination=destination,
            destination_from_placed_source=relation,
            pregrasp=pregrasp,
            grasp=grasp,
            lift=0.08,
        )
        initial = env._get_initial_states()
        images = {key: [] for key in CAMERAS}
        states, object_poses, target_poses, actions, assigned, waypoints = [], [], [], [], [], [0]
        source_to_tcp_actual = None
        object_pose = source.copy()

        def capture():
            position, orientation = controller.tcp_pose(q)
            finger = [bare_names.index(name) for name in scenario.robots[0].ee_joint_names]
            openness = float(np.clip(np.mean(q[finger]) / 0.04, 0, 1))
            states.append(np.r_[position, orientation.as_rotvec(), openness].astype(np.float32))
            object_poses.append(obs.objects["object"].root_state[0, :7].detach().cpu().numpy().copy())
            target_poses.append(obs.objects["object2"].root_state[0, :7].detach().cpu().numpy().copy())
            for key, camera in CAMERAS.items():
                frame = obs.cameras[camera].rgb[0].detach().cpu().numpy()
                if frame.shape != (size, size, 3) or frame.dtype != np.uint8:
                    raise ValueError(f"Invalid camera frame for {camera}: {frame.shape} {frame.dtype}")
                images[key].append(frame.copy())

        def step(action, phase):
            nonlocal q, obs, object_pose
            joints = controller.joint_targets(q, action)
            q_expected = np.array([joints[name] for name in bare_names])
            if source_to_tcp_actual is not None:
                position, orientation = controller.tcp_pose(q_expected)
                object_pose = _pose(position, orientation) @ np.linalg.inv(source_to_tcp_actual)
            initial[0]["objects"]["object"]["pos"] = torch.tensor(object_pose[:3, 3], dtype=torch.float32)
            initial[0]["objects"]["object"]["rot"] = torch.tensor(
                Rotation.from_matrix(object_pose[:3, :3]).as_quat(scalar_first=True), dtype=torch.float32
            )
            initial[0]["robots"]["franka"]["dof_pos"] = joints
            env.handler.set_states(initial)
            env.handler.refresh_render()
            obs = env.handler.get_states(mode="tensor")
            q = obs.robots["franka"].joint_pos[0].detach().cpu().numpy().copy()
            actual_object = obs.objects["object"].root_state[0].detach().cpu().numpy()
            if not np.allclose(q, q_expected, atol=0.001, rtol=0) or not np.allclose(
                actual_object[:3], object_pose[:3, 3], atol=0.001, rtol=0
            ):
                raise RuntimeError("Backend did not apply the requested kinematic state")
            actions.append(np.asarray(action, dtype=np.float32))
            assigned.append(phase)
            capture()

        def move(target, gripper: float, phase: int):
            target_rotation = Rotation.from_matrix(target[:3, :3])
            for _ in range(100):
                position, orientation = controller.tcp_pose(q)
                delta = target[:3, 3] - position
                rotation = (target_rotation * orientation.inv()).as_rotvec()
                distance, angle = np.linalg.norm(delta), np.linalg.norm(rotation)
                if distance < 0.0015 and angle < 0.025:
                    return
                command = np.r_[
                    delta * min(1.0, 0.018 / max(distance, 1e-12)),
                    rotation * min(1.0, 0.18 / max(angle, 1e-12)),
                    gripper,
                ]
                step(command, phase)
            raise RuntimeError(f"Could not reach kinematic waypoint for phase {phase}")

        capture()
        move(plan["pregrasp"], 1.0, 1)
        waypoints.append(len(actions))
        move(plan["grasp"], 1.0, 2)
        step(np.r_[np.zeros(6), 0.0], 2)
        waypoints.append(len(actions))
        position, orientation = controller.tcp_pose(q)
        source_to_tcp_actual = np.linalg.inv(source) @ _pose(position, orientation)
        attached_plan = resolve_pick_place(
            world_from_source=source,
            world_from_destination=destination,
            destination_from_placed_source=relation,
            pregrasp=pregrasp,
            grasp=source_to_tcp_actual,
            lift=0.08,
        )
        move(attached_plan["lift"], 0.0, 3)
        waypoints.append(len(actions))
        move(attached_plan["preplace"], 0.0, 4)
        waypoints.append(len(actions))
        move(attached_plan["place"], 0.0, 5)
        source_to_tcp_actual = None
        step(np.r_[np.zeros(6), 1.0], 5)
        waypoints.append(len(actions))
        expected_object = destination @ relation
        position_error = float(np.linalg.norm(object_pose[:3, 3] - expected_object[:3, 3]))
        orientation_error = float(
            np.linalg.norm(Rotation.from_matrix(object_pose[:3, :3] @ expected_object[:3, :3].T).as_rotvec())
        )
        if position_error > 0.004 or orientation_error > 0.04:
            raise RuntimeError("Final kinematic object pose does not match the shared target relation")
        actions.append(np.r_[np.zeros(6), 1.0].astype(np.float32))
        assigned.append(5)
        if len(waypoints) != 6 or np.any(np.diff(waypoints) <= 0):
            raise RuntimeError("Waypoints are not six distinct ordered frames")
        data = {
            **{key: np.stack(value) for key, value in images.items()},
            "robot_states": np.stack(states),
            "actions": np.stack(actions),
            "object_poses": np.stack(object_poses),
            "target_object_poses": np.stack(target_poses),
            "waypoints_idx": np.array(waypoints, dtype=np.int64),
            "assigned_wps": np.array(assigned, dtype=np.int64),
        }
        return data, {
            "source_pose": source.tolist(),
            "destination_pose": destination.tolist(),
            "destination_from_placed_source": relation.tolist(),
            "position_error_m": position_error,
            "orientation_error_rad": orientation_error,
        }
    finally:
        env.close()


def _single_object_scenario(simulator: str, object_id: str, size: int, xy, asset_root):
    """Build a single-object scene while reusing RoboVerse's ICL task setup."""
    scenario = build_scenario(
        simulator=simulator,
        object_id=object_id,
        image_size=size,
        object_xy=tuple(xy),
        asset_root=asset_root,
    )
    # push_gn has exactly one sampled object. Goal markers from the task's
    # physical probe are not part of this source skill's scene specification.
    scenario.objects = scenario.objects[:1]
    return scenario


def _push_trajectory(simulator: str, object_id: str, size: int, xy, sample, asset_root) -> tuple[dict, dict]:
    """Render the source push waypoint sequence with explicit object attachment."""
    scenario = _single_object_scenario(simulator, object_id, size, xy, asset_root)
    env = PushToRegionTask(scenario, goal="left")
    try:
        obs, _ = env.reset()
        names = env.handler.get_joint_names("franka", sort=True)
        bare_names = [name.split("/")[-1] for name in names]
        controller = FrankaCartesianController(scenario.robots[0], names)
        q = obs.robots["franka"].joint_pos[0].detach().cpu().numpy().copy()
        root = obs.objects["object"].root_state[0].detach().cpu().numpy()
        source = _pose(root[:3], Rotation.from_quat(root[3:7], scalar_first=True))
        initial = env._get_initial_states()
        images = {key: [] for key in CAMERAS}
        states, object_poses, actions, assigned, waypoints = [], [], [], [], [0]
        object_pose = source.copy()
        source_to_tcp_actual = None

        def capture():
            position, orientation = controller.tcp_pose(q)
            fingers = [bare_names.index(name) for name in scenario.robots[0].ee_joint_names]
            openness = float(np.clip(np.mean(q[fingers]) / 0.04, 0.0, 1.0))
            states.append(np.r_[position, orientation.as_rotvec(), openness].astype(np.float32))
            object_poses.append(obs.objects["object"].root_state[0, :7].detach().cpu().numpy().copy())
            for key, camera in CAMERAS.items():
                frame = obs.cameras[camera].rgb[0].detach().cpu().numpy()
                if frame.shape != (size, size, 3) or frame.dtype != np.uint8:
                    raise ValueError(f"Invalid camera frame for {camera}: {frame.shape} {frame.dtype}")
                images[key].append(frame.copy())

        def step(action, phase):
            nonlocal q, obs, object_pose
            joints = controller.joint_targets(q, action)
            q_expected = np.array([joints[name] for name in bare_names])
            if source_to_tcp_actual is not None:
                position, orientation = controller.tcp_pose(q_expected)
                object_pose = _pose(position, orientation) @ np.linalg.inv(source_to_tcp_actual)
            initial[0]["objects"]["object"]["pos"] = torch.tensor(object_pose[:3, 3], dtype=torch.float32)
            initial[0]["objects"]["object"]["rot"] = torch.tensor(
                Rotation.from_matrix(object_pose[:3, :3]).as_quat(scalar_first=True), dtype=torch.float32
            )
            initial[0]["robots"]["franka"]["dof_pos"] = joints
            env.handler.set_states(initial)
            env.handler.refresh_render()
            obs = env.handler.get_states(mode="tensor")
            q = obs.robots["franka"].joint_pos[0].detach().cpu().numpy().copy()
            actual_object = obs.objects["object"].root_state[0].detach().cpu().numpy()
            if not np.allclose(q, q_expected, atol=0.001, rtol=0) or not np.allclose(
                actual_object[:3], object_pose[:3, 3], atol=0.001, rtol=0
            ):
                raise RuntimeError("Backend did not apply the requested kinematic state")
            actions.append(np.asarray(action, dtype=np.float32))
            assigned.append(phase)
            capture()

        def move(target, gripper: float, phase: int):
            start_count = len(actions)
            target_rotation = Rotation.from_matrix(target[:3, :3])
            for _ in range(100):
                position, orientation = controller.tcp_pose(q)
                delta = target[:3, 3] - position
                rotation = (target_rotation * orientation.inv()).as_rotvec()
                distance, angle = np.linalg.norm(delta), np.linalg.norm(rotation)
                if distance < 0.0015 and angle < 0.025:
                    break
                command = np.r_[
                    delta * min(1.0, 0.018 / max(distance, 1e-12)),
                    rotation * min(1.0, 0.18 / max(angle, 1e-12)),
                    gripper,
                ]
                step(command, phase)
            else:
                raise RuntimeError(f"Could not reach kinematic push waypoint for phase {phase}")
            if len(actions) == start_count:
                step(np.r_[np.zeros(6), gripper], phase)

        def waypoint_pose(index: int) -> np.ndarray:
            pose = sample.waypoints[index, 1:]
            return source @ _pose(pose[:3], Rotation.from_quat(pose[3:], scalar_first=True))

        finger_ids = [bare_names.index(name) for name in scenario.robots[0].ee_joint_names]
        initial_openness = float(np.clip(np.mean(q[finger_ids]) / 0.04, 0.0, 1.0))
        actions.append(np.r_[np.zeros(6), initial_openness].astype(np.float32))
        assigned.append(0)
        capture()

        move(waypoint_pose(0), 1.0, 1)
        waypoints.append(len(states) - 1)
        step(np.r_[np.zeros(6), 0.0], 2)  # CLOSE_CODE: close without attaching.
        waypoints.append(len(states) - 1)
        move(waypoint_pose(2), 0.0, 3)
        waypoints.append(len(states) - 1)
        position, orientation = controller.tcp_pose(q)
        source_to_tcp_actual = np.linalg.inv(source) @ _pose(position, orientation)
        step(np.r_[np.zeros(6), 0.0], 4)  # GRASP_CODE: attach object to the hand.
        waypoints.append(len(states) - 1)
        move(waypoint_pose(4), 0.0, 5)

        expected_object = source.copy()
        expected_object[:3, 3] += source[:3, :3] @ sample.postpush_delta
        position_error = float(np.linalg.norm(object_pose[:3, 3] - expected_object[:3, 3]))
        orientation_error = float(
            np.linalg.norm(Rotation.from_matrix(object_pose[:3, :3] @ expected_object[:3, :3].T).as_rotvec())
        )
        if position_error > 0.004 or orientation_error > 0.04:
            raise RuntimeError("Final kinematic object pose does not match the sampled push displacement")
        actions.append(np.r_[np.zeros(6), actions[-1][6]].astype(np.float32))
        assigned.append(5)
        capture()
        waypoints.append(len(states) - 1)
        if len(waypoints) != 6 or np.any(np.diff(waypoints) <= 0):
            raise RuntimeError("Push waypoints are not six distinct ordered frames")
        data = {
            **{key: np.stack(value) for key, value in images.items()},
            "robot_states": np.stack(states),
            "actions": np.stack(actions),
            "object_poses": np.stack(object_poses),
            "waypoints_idx": np.asarray(waypoints, dtype=np.int64),
            "assigned_wps": np.asarray(assigned, dtype=np.int64),
        }
        return data, {
            "source_pose": source.tolist(),
            "geometric_prepush": sample.geometric_prepush.tolist(),
            "geometric_contact": sample.geometric_contact.tolist(),
            "geometric_postpush": sample.geometric_postpush.tolist(),
            "postpush_delta_m": sample.postpush_delta.tolist(),
            "final_position_error_m": position_error,
            "final_orientation_error_rad": orientation_error,
        }
    finally:
        env.close()


def collect(
    simulator: str,
    object_id: str,
    image_size: int,
    grasp_index: int | None,
    output: Path,
    seed: int = 0,
    target_object_id: str | None = None,
    task: str = "pick_and_place_gn",
    asset_root: Path | None = None,
) -> None:
    """Create one reviewable two-trajectory prototype pair, atomically."""
    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size < 16 or output.suffix != ".h5":
        raise ValueError("Use an image size of at least 16 and a .h5 output")
    if output.exists():
        raise FileExistsError(output)
    if task not in {"pick_and_place_gn", "push_gn"}:
        raise ValueError(f"Unsupported SynthICL task: {task}")
    asset_root = Path(asset_root) if asset_root is not None else ASSET_ROOT
    if task == "push_gn" and target_object_id is not None:
        raise ValueError("push_gn is a single-object task and does not accept target_object_id")
    if task == "pick_and_place_gn":
        target_object_id = target_object_id or object_id
        selected_grasp_index = 0 if grasp_index is None else grasp_index
        task_key = pick_place_task_key(
            object_id=object_id,
            target_object_id=target_object_id,
            grasp_index=selected_grasp_index,
            backend=simulator,
            seed=seed,
        )
    else:
        selected_grasp_index = grasp_index
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    scenario = build_scenario(simulator=simulator, object_id=object_id, image_size=image_size, asset_root=asset_root)
    asset = Path(scenario.objects[0].urdf_path).parent
    info = json.loads((asset / "provenance.json").read_text(encoding="utf-8"))
    if info.get("source_revision") != "5f3124eff350ea0e1014253c700842aa94345607":
        raise ValueError("Expected a prepared pinned SynthICL source asset")
    if info.get("original_scale") != 1.0:
        raise ValueError("SynthICL object assets must retain their original scale")
    for name in ("textured.obj", "collision.ply", "material_0.png"):
        if _sha256(asset / name) != info["source_files_sha256"][name]:
            raise ValueError(f"Prepared asset was modified: {name}")
    grasp_file = asset / info["graspnet_path"]
    origin_grasps = [digest for name, digest in info["origin_files_sha256"].items() if name.endswith("_grasps.npy")]
    if len(origin_grasps) != 1 or _sha256(grasp_file) != origin_grasps[0]:
        raise ValueError("Prepared GraspNet annotations differ from the verified source")
    grasps = np.load(grasp_file, allow_pickle=False)
    if task == "push_gn":
        if not all(key in info for key in ("bbox_half_extents_m", "grasp_angle_rad", "source_base_orientation")):
            raise ValueError("Re-prepare this asset with the current tool to include source push sampler metadata")
        base_orientation = info["source_base_orientation"]
        if not isinstance(base_orientation, dict) or not isinstance(base_orientation.get("z_negative"), bool):
            raise ValueError("Prepared source asset is missing the boolean base_orientation.z_negative value")
        if selected_grasp_index is not None and (
            isinstance(selected_grasp_index, bool)
            or not isinstance(selected_grasp_index, int)
            or not 0 <= selected_grasp_index < len(grasps)
        ):
            raise ValueError("Grasp index is outside the source annotation array")
        sample = sample_push_gn_waypoints(
            grasp_rows=grasps,
            bbox_half_extents_m=info["bbox_half_extents_m"],
            grasp_angle_rad=info["grasp_angle_rad"],
            seed=seed,
            grasp_index=selected_grasp_index,
            pregrasp_distance_m=(0.05, 0.10),
            postpush_offset_m=None,
            base_orientation_z_negative=base_orientation["z_negative"],
        )
        selected_grasp_index = sample.grasp_index
        task_key = push_task_key(
            object_id=object_id,
            grasp_index=selected_grasp_index,
            backend=simulator,
            seed=seed,
        )
        traces = [
            _push_trajectory(simulator, object_id, image_size, xy, sample, asset_root)
            for xy in ((0.48, -0.02), (0.52, 0.02))
        ]
        target_info = None
        target_asset = None
    else:
        if grasps.ndim != 2 or grasps.shape[1] != 17 or not 0 <= selected_grasp_index < len(grasps):
            raise ValueError("Grasp index is outside the source annotation array")
        grasp_row = grasps[selected_grasp_index]
        target_scenario = build_scenario(
            simulator=simulator,
            object_id=target_object_id,
            image_size=image_size,
            asset_root=asset_root,
        )
        target_asset = Path(target_scenario.objects[0].urdf_path).parent
        target_info = json.loads((target_asset / "provenance.json").read_text(encoding="utf-8"))
        if target_info.get("source_revision") != "5f3124eff350ea0e1014253c700842aa94345607":
            raise ValueError("Expected a prepared pinned SynthICL target asset")
        if target_info.get("original_scale") != 1.0:
            raise ValueError("SynthICL target assets must retain their original scale")
        for name in ("textured.obj", "collision.ply", "material_0.png"):
            if _sha256(target_asset / name) != target_info["source_files_sha256"][name]:
                raise ValueError(f"Prepared target asset was modified: {name}")
        traces = [
            _trajectory(simulator, object_id, target_object_id, image_size, xy, grasp_row, asset_root)
            for xy in ((0.48, -0.02), (0.52, 0.02))
        ]
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    temporary = Path(path)
    try:
        with h5py.File(temporary, "w") as h5:
            metadata = {
                "format": "roboverse.icl.pseudo_prototype.v1",
                "demonstration_kind": "pseudo_kinematic",
                "task_key": task_key,
                "task_semantics": (
                    "GraspNet approach, close without attachment, contact, attach object to hand, then move to the sampled postpush pose"
                    if task == "push_gn"
                    else "GraspNet pick and place source object on target object"
                ),
                "simulator": simulator,
                "task_id": task,
                "object_id": object_id,
                "source_revision": info["source_revision"],
                "asset_source_repository": info["source_repository"],
                "asset_origin_files_sha256": info["origin_files_sha256"],
                "grasp_index": selected_grasp_index,
                "source_object_scale": 1.0,
                "grasp_sha256": _sha256(grasp_file),
                "camera_roles": CAMERA_ROLES,
                "asset_files_sha256": {
                    name: info["source_files_sha256"][name]
                    for name in ("textured.obj", "collision.ply", "material_0.png")
                },
                "camera_configs": {
                    camera.name: {
                        "pos": camera.pos,
                        "look_at": camera.look_at,
                        "width": camera.width,
                        "height": camera.height,
                        "horizontal_fov": camera.horizontal_fov,
                    }
                    for camera in scenario.cameras
                },
                "image_processing": "unsegmented_rgb",
                "frame_period_s": 1 / 30,
                "physics_simulated": False,
                "trajectories": [item[1] for item in traces],
            }
            if target_info is not None:
                metadata["target_object_id"] = target_object_id
                metadata["target_source_revision"] = target_info["source_revision"]
                metadata["target_object_scale"] = 1.0
                metadata["target_asset_files_sha256"] = {
                    name: target_info["source_files_sha256"][name]
                    for name in ("textured.obj", "collision.ply", "material_0.png")
                }
                metadata["placement_semantics"] = (
                    "source object bottom aligned with target object top; no physical contacts simulated"
                )
            else:
                metadata["action_semantics"] = list(sample.action_semantics)
                metadata["push_sampler"] = {
                    "bbox_half_extents_m": info["bbox_half_extents_m"],
                    "grasp_angle_rad": info["grasp_angle_rad"],
                    "base_orientation_z_negative": base_orientation["z_negative"],
                    "pregrasp_distance_range_m": [0.05, 0.10],
                    "postpush_offset": "source_geometric_default",
                    "waypoints_object_local": sample.waypoints.tolist(),
                }
                metadata["waypoint_names"] = [
                    "start",
                    "pregrasp",
                    "close_without_attachment",
                    "contact",
                    "attach_object_to_hand",
                    "postpush",
                ]
                metadata["push_semantics"] = (
                    "The object is kinematically attached to the hand after contact; no physical sliding or contact success is asserted."
                )
            h5.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True, allow_nan=False)
            for index, (data, _) in enumerate(traces):
                group = h5.create_group(f"trajectory_{index}")
                for key, values in data.items():
                    group.create_dataset(key, data=values, compression="gzip", compression_opts=1)
        temporary.rename(output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", choices=("mujoco", "genesis", "newton"), required=True)
    parser.add_argument("--object-id", default="synthicl_tea_box_base2")
    parser.add_argument("--target-object-id")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--task", choices=("pick_and_place_gn", "push_gn"), default="pick_and_place_gn")
    parser.add_argument("--grasp-index", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(
        args.sim,
        args.object_id,
        args.image_size,
        args.grasp_index,
        args.output,
        seed=args.seed,
        target_object_id=args.target_object_id,
        task=args.task,
        asset_root=args.asset_root,
    )
    print(f"Saved kinematic pseudo-demonstration pair: {args.output}")


if __name__ == "__main__":
    main()
