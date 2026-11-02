"""Contracts for reference-conditioned tasks, data preparation and collection."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from roboverse_pack.tasks.icl.push_to_region import (
    GOAL_CENTERS,
    PickObjectTask,
    PickPlaceRegionTask,
    PushToRegionTask,
    _scenario,
    build_object_choice_scenario,
    build_scenario,
)


@pytest.mark.general
def test_object_choice_changes_success_without_changing_the_scene():
    scene = build_object_choice_scenario(simulator="mujoco", layout=0, image_size=64)
    swapped = build_object_choice_scenario(simulator="mujoco", layout=1, image_size=64)
    assert [obj.name for obj in scene.objects] == ["red", "blue", "destination"]
    assert scene.objects[0].default_position[1] == swapped.objects[1].default_position[1]
    root_red = torch.zeros(1, 13)
    root_blue = torch.zeros(1, 13)
    root_red[0, :3] = torch.tensor([0.62, 0.0, 0.025])
    root_blue[0, :3] = torch.tensor([0.48, 0.12, 0.025])
    states = SimpleNamespace(
        objects={"red": SimpleNamespace(root_state=root_red), "blue": SimpleNamespace(root_state=root_blue)},
        robots={
            "franka": SimpleNamespace(joint_pos=torch.full((1, 2), 0.04), joint_pos_target=torch.full((1, 2), 0.04))
        },
    )
    task = PickObjectTask.__new__(PickObjectTask)
    task._lifted = torch.ones(1, dtype=torch.bool)
    task._finger_indices = [0, 1]
    task._target = "red"
    assert task._terminated(states).item()
    task._target = "blue"
    assert not task._terminated(states).item()
    task._target = "red"
    root_blue[0, :3] = root_red[0, :3]
    assert not task._terminated(states).item()


@pytest.mark.general
def test_goal_switch_changes_success_for_the_same_state():
    roots = torch.zeros(4, 13)
    roots[:, 3] = 1.0
    roots[0, :3] = torch.tensor([*GOAL_CENTERS["left"], 0.02])
    roots[1, :3] = torch.tensor([*GOAL_CENTERS["right"], 0.02])
    roots[2, :3] = roots[0, :3]
    roots[2, 7] = 0.2
    roots[3, :3] = roots[0, :3]
    roots[3, 10] = 0.5
    states = SimpleNamespace(objects={"object": SimpleNamespace(root_state=roots)})
    task = PushToRegionTask.__new__(PushToRegionTask)
    task._goal_xy = GOAL_CENTERS["left"]
    assert task._terminated(states).tolist() == [True, False, False, False]
    task._goal_xy = GOAL_CENTERS["right"]
    assert task._terminated(states).tolist() == [False, True, False, False]


@pytest.mark.general
def test_pick_place_requires_lift_and_release_and_clears_history_on_reset():
    task = PickPlaceRegionTask.__new__(PickPlaceRegionTask)
    task._goal_xy = GOAL_CENTERS["left"]
    task._lifted = torch.zeros(1, dtype=torch.bool)
    task._finger_indices = [0, 1]
    root = torch.zeros(1, 13)
    root[0, :3] = torch.tensor([*GOAL_CENTERS["left"], 0.02])
    states = SimpleNamespace(
        objects={"object": SimpleNamespace(root_state=root)},
        robots={
            "franka": SimpleNamespace(
                joint_pos=torch.tensor([[0.04, 0.04]]), joint_pos_target=torch.tensor([[0.04, 0.04]])
            )
        },
    )
    assert not task._terminated(states)[0]
    root[0, 2] = 0.15
    task._observe_lift(states)
    assert not task._lifted[0]
    states.robots["franka"].joint_pos_target[:] = 0
    task._observe_lift(states)
    root[0, 2] = 0.02
    states.robots["franka"].joint_pos[:] = 0
    assert not task._terminated(states)[0]
    states.robots["franka"].joint_pos[:] = 0.04
    states.robots["franka"].joint_pos_target[:] = 0.04
    assert task._terminated(states)[0]
    task._reset_lift([0])
    assert not task._terminated(states)[0]


@pytest.mark.general
def test_context_comparison_requires_identical_query():
    from tools.icl.compare_context import compare

    common = {
        "task": "icl.push_to_region",
        "simulator": "mujoco",
        "object_id": "brick",
        "asset_sha256": "a" * 64,
        "query_seed": 7,
        "position_jitter": 0.003,
        "checkpoint_sha256": "b" * 64,
        "initial_object_state": [0.5, 0.0, 0.02],
        "initial_robot_joints": [0.0, 1.0],
        "steps": 10,
        "failure": "",
    }
    left = {**common, "goal": "left", "reference_origin": "left_demo", "success": True}
    right = {**common, "goal": "right", "reference_origin": "right_demo", "success": False}
    assert compare(left, right)["both_succeeded"] is False
    with pytest.raises(ValueError, match="initial state"):
        compare(left, {**right, "initial_object_state": [0.6, 0.0, 0.02]})


@pytest.mark.general
def test_native_source_episode_rejects_wrong_timebase_or_action_shape():
    pytest.importorskip("pyarrow")
    pytest.importorskip("av")
    from tools.icl.inspect_dataset import validate_episode

    episode = {"episode_index": 4, "length": 3, "dataset_from_index": 12, "dataset_to_index": 15}
    frames = [
        {
            "episode_index": 4,
            "frame_index": i,
            "index": 12 + i,
            "timestamp": i / 30,
            "action.joints": [0, 1],
            "observation.joints": [0, 1, 2],
            "observation.scalar": 1,
        }
        for i in range(3)
    ]
    validate_episode(episode=episode, frames=frames, fps=30, action_key="action.joints", action_dim=2)
    # Human video can provide observations without inventing robot action labels.
    validate_episode(episode=episode, frames=frames, fps=30, action_key=None, action_dim=None)
    validate_episode(
        episode=episode,
        frames=frames,
        fps=30,
        action_key=None,
        action_dim=None,
        observation_dims={"observation.scalar": 1},
    )
    with pytest.raises(ValueError, match="both"):
        validate_episode(episode=episode, frames=frames, fps=30, action_key=None, action_dim=2)
    with pytest.raises(ValueError, match="timestamps"):
        validate_episode(
            episode=episode,
            frames=[*frames[:2], {**frames[2], "timestamp": 2}],
            fps=30,
            action_key="action.joints",
            action_dim=2,
        )
    with pytest.raises(ValueError, match="action"):
        validate_episode(episode=episode, frames=frames, fps=30, action_key="action.joints", action_dim=7)
    with pytest.raises(ValueError, match="observation"):
        validate_episode(
            episode=episode,
            frames=frames,
            fps=30,
            action_key="action.joints",
            action_dim=2,
            observation_dims={"observation.joints": 2},
        )


@pytest.mark.general
def test_batch_rejects_asset_split_leak_before_starting_a_simulator(tmp_path, monkeypatch):
    from tools.icl import collect

    asset = tmp_path / "object"
    asset.mkdir()
    (asset / "provenance.json").write_text(
        json.dumps({"source_files_sha256": {"textured.obj": "a", "collision.ply": "b"}})
    )
    monkeypatch.setattr(
        collect,
        "build_scenario",
        lambda **kwargs: SimpleNamespace(objects=[SimpleNamespace(urdf_path=asset / "object.urdf")]),
    )
    jobs = [
        {"sim": "mujoco", "goal": "left", "split": split, "object_id": "box", "seed": seed}
        for split, seed in (("train", 0), ("test", 1))
    ]
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(jobs))
    with pytest.raises(ValueError, match="multiple splits"):
        collect.batch(SimpleNamespace(plan=plan, output=tmp_path / "dataset", image_size=64))
    assert not (tmp_path / "dataset/logs").exists()


@pytest.mark.general
def test_batch_retries_startup_errors_without_overwriting_logs(tmp_path, monkeypatch):
    from tools.icl import collect

    asset = tmp_path / "object"
    asset.mkdir()
    (asset / "provenance.json").write_text(
        json.dumps({"source_files_sha256": {"textured.obj": "a", "collision.ply": "b"}})
    )
    monkeypatch.setattr(
        collect,
        "build_scenario",
        lambda **kwargs: SimpleNamespace(objects=[SimpleNamespace(urdf_path=asset / "object.urdf")]),
    )
    monkeypatch.setattr(collect.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps([{"sim": "mujoco", "goal": "left", "split": "train", "object_id": "box", "seed": 0}]))
    args = SimpleNamespace(plan=plan, output=tmp_path / "dataset", image_size=64)
    first = collect.batch(args)
    second = collect.batch(args)
    assert first["jobs"][0]["status"] == second["jobs"][0]["status"] == "execution_error"
    assert first["jobs"][0]["log"] != second["jobs"][0]["log"]
    assert len(list((args.output / "logs").glob("*.log"))) == 2


@pytest.mark.general
def test_reset_contains_all_visible_objects():
    scenario = _scenario(asset_dir=Path("unused"), z=0.03, simulator="mujoco", headless=True, image_size=32)
    task = PushToRegionTask.__new__(PushToRegionTask)
    task.scenario = scenario
    state = task._get_initial_states()[0]
    assert set(state["objects"]) == {"object", "region_left", "region_right"}
    assert set(state["robots"]) == {"franka"}
    assert scenario.sim_params.njmax == 512


@pytest.mark.general
def test_build_scenario_uses_explicit_asset_root_and_bounds(tmp_path):
    asset = tmp_path / "my_box"
    asset.mkdir()
    for name in ("object.urdf", "object.xml"):
        (asset / name).write_text("")
    (asset / "provenance.json").write_text(json.dumps({"bounds_m": [[-0.02, -0.02, -0.04], [0.02, 0.02, 0.04]]}))
    scenario = build_scenario(simulator="mujoco", object_id="my_box", asset_root=tmp_path, image_size=32)
    assert Path(scenario.objects[0].urdf_path) == asset / "object.urdf"
    assert scenario.objects[0].default_position[2] == pytest.approx(0.043)


@pytest.mark.general
def test_cartesian_controller_reaches_a_small_delta():
    pytest.importorskip("mujoco")
    from roboverse_pack.robots.franka_cfg import FrankaCfg
    from roboverse_pack.tasks.icl.control import FrankaCartesianController

    robot = FrankaCfg()
    if not Path(robot.mjcf_path).is_file():
        pytest.skip("local Franka MJCF asset is not installed")
    names = sorted(robot.default_joint_positions)
    q = np.array([robot.default_joint_positions[name] for name in names])
    controller = FrankaCartesianController(robot, names)
    start, rotation = controller.tcp_pose(q)
    result = controller.joint_targets(q, [0.01, 0, 0, 0, 0, 0, 1])
    end, end_rotation = controller.tcp_pose(np.array([result[name] for name in names]))
    assert np.linalg.norm(end - start - [0.01, 0, 0]) < 0.001
    assert np.linalg.norm((end_rotation * rotation.inv()).as_rotvec()) < 0.02
    assert result["panda_finger_joint1"] == pytest.approx(0.04)
    with pytest.raises(ValueError, match="exceeds"):
        controller.joint_targets(q, [1, 0, 0, 0, 0, 0, 1])


@pytest.mark.general
def test_dtd_extraction_rejects_path_traversal_and_does_not_overwrite(tmp_path, monkeypatch):
    from tools.icl import download_dtd

    monkeypatch.setattr(download_dtd, "verify", lambda _: None)
    monkeypatch.setattr(download_dtd, "IMAGE_COUNT", 1)
    monkeypatch.setattr(download_dtd, "CATEGORY_COUNT", 1)

    def archive_with(name, path):
        with tarfile.open(path, "w:gz") as tar:
            member = tarfile.TarInfo(name)
            content = b"texture"
            member.size = len(content)
            tar.addfile(member, io.BytesIO(content))

    bad = tmp_path / "bad.tar.gz"
    archive_with("dtd/../../outside.jpg", bad)
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError, match="Unexpected archive member"):
        download_dtd.extract(bad, output)
    assert not (tmp_path / "outside.jpg").exists() and not (output / "extracted").exists()
    good = tmp_path / "good.tar.gz"
    archive_with("dtd/images/striped/example.jpg", good)
    images = download_dtd.extract(good, output)
    assert (images / "striped/example.jpg").read_bytes() == b"texture"
    with pytest.raises(FileExistsError):
        download_dtd.extract(good, output)


@pytest.mark.general
def test_synthicl_grasp_and_place_keep_object_relative_semantics_across_scenes():
    from roboverse_learn.il.icl.waypoints import graspnet_to_tcp, resolve_pick_place

    row = np.zeros(17)
    row[3] = 0.02
    row[4:13] = np.eye(3).reshape(-1)
    row[13:16] = [0.01, 0.02, 0.03]
    pre, grasp = graspnet_to_tcp(row, approach_distance=0.08)
    np.testing.assert_allclose(grasp[:3, 3], [0.03, 0.02, 0.03])
    np.testing.assert_allclose(pre[:3, 3], [-0.05, 0.02, 0.03])
    relation = np.eye(4)
    relation[:3, 3] = [0.01, -0.02, 0.07]
    sources = [np.eye(4), np.eye(4)]
    destinations = [np.eye(4), np.eye(4)]
    sources[0][:3, 3], sources[1][:3, 3] = [0.4, -0.1, 0.05], [0.6, 0.1, 0.05]
    destinations[0][:3, 3], destinations[1][:3, 3] = [0.7, 0.1, 0], [0.45, -0.2, 0]
    destinations[1][:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    plans = [
        resolve_pick_place(
            world_from_source=s,
            world_from_destination=d,
            destination_from_placed_source=relation,
            pregrasp=pre,
            grasp=grasp,
        )
        for s, d in zip(sources, destinations, strict=True)
    ]
    for source, destination, plan in zip(sources, destinations, plans, strict=True):
        np.testing.assert_allclose(np.linalg.inv(source) @ plan["grasp"], grasp, atol=1e-10)
        np.testing.assert_allclose(
            np.linalg.inv(destination) @ plan["place"] @ np.linalg.inv(grasp), relation, atol=1e-10
        )
    row[4:13] = (-np.eye(3)).reshape(-1)
    with pytest.raises(ValueError, match="proper rotation"):
        graspnet_to_tcp(row)


@pytest.mark.general
def test_dtd_range_download_resumes_and_rejects_wrong_range(tmp_path, monkeypatch):
    from tools.icl import download_dtd

    payload = b"0123456789"
    monkeypatch.setattr(download_dtd, "SIZE", len(payload))
    chunk = tmp_path / "chunk-0000.part"
    chunk.write_bytes(payload[:3])

    class Response:
        status_code = 206
        headers = {"Content-Range": "bytes 3-9/10"}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def iter_content(self, _):
            yield payload[3:]

    def get(url, *, headers, stream, timeout):
        assert headers["Range"] == "bytes=3-9"
        return Response()

    monkeypatch.setattr(download_dtd.requests, "get", get)
    assert download_dtd.download_chunk(tmp_path, 0, 0, 9).read_bytes() == payload
    chunk.write_bytes(payload[:3])
    Response.headers = {"Content-Range": "bytes 0-9/10"}
    with pytest.raises(ValueError, match="requested range"):
        download_dtd.download_chunk(tmp_path, 0, 0, 9)
    assert chunk.read_bytes() == payload[:3]


@pytest.mark.general
def test_synthicl_asset_preparation_preserves_grasps_and_rejects_empty_sources(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    yaml = pytest.importorskip("yaml")
    Image = pytest.importorskip("PIL.Image")
    from tools.icl.prepare_synthicl_assets import prepare, verify_source

    source = tmp_path / "source"
    obj = source / "objects" / "box"
    obj.mkdir(parents=True)
    mesh = trimesh.creation.box(extents=[0.04, 0.06, 0.08])
    (obj / "box.obj").write_text("mtllib material.mtl\n" + mesh.export(file_type="obj"), encoding="utf-8")
    (obj / "material.mtl").write_text("newmtl material_0\nmap_Kd material_0.png\n", encoding="utf-8")
    Image.new("RGB", (2, 2), (255, 0, 0)).save(obj / "material_0.png")
    grasps = np.zeros((1, 17), dtype=np.float32)
    grasps[0, 4:13] = np.eye(3).reshape(-1)
    np.save(obj / "grasps.npy", grasps)
    entry = {
        "name": "box",
        "path": "objects/box/box.obj",
        "graspnet_pose_path": "objects/box/grasps.npy",
        "bbox_size": [0.02, 0.03, 0.04],
        "grasp_angle": 1.57,
        "base_orientation": {"z_negative": False},
    }
    (source / "graspable_configuration.yaml").write_text(yaml.safe_dump({"objects": [entry]}), encoding="utf-8")

    def write_source_manifest():
        files = {}
        for path in [source / "graspable_configuration.yaml", *obj.iterdir()]:
            content = path.read_bytes()
            digest = hashlib.sha1(f"blob {len(content)}\0".encode() + content, usedforsecurity=False).hexdigest()
            files[path.relative_to(source).as_posix()] = {"size": len(content), "git_blob_sha1": digest}
        (source / "roboverse_source_manifest.json").write_text(
            json.dumps({
                "repository": "sniper-qian/SynthICL-PseudoDemo-Objects",
                "revision": "5f3124eff350ea0e1014253c700842aa94345607",
                "files": files,
            }),
            encoding="utf-8",
        )

    write_source_manifest()
    assert not verify_source(source)["missing"] and not verify_source(source)["invalid"]
    output = tmp_path / "prepared"
    prepare(source, output, ["box"])
    prepared = output / "synthicl_box"
    info = json.loads((prepared / "provenance.json").read_text(encoding="utf-8"))
    assert info["original_scale"] == 1.0 and info["graspnet_count"] == 1
    assert info["source_object"] == "box"
    assert info["bbox_half_extents_m"] == [0.02, 0.03, 0.04]
    assert info["grasp_angle_rad"] == 1.57
    assert info["source_base_orientation"] == {"z_negative": False}
    np.testing.assert_array_equal(np.load(prepared / "grasps.npy", allow_pickle=False), grasps)
    assert trimesh.load(prepared / "collision.obj", force="mesh").is_watertight
    dtd_root = tmp_path / "dtd"
    dtd_images = dtd_root / "extracted/dtd/images/banded"
    dtd_images.mkdir(parents=True)
    image = dtd_images / "sample.jpg"
    Image.new("RGB", (4, 4), (20, 200, 100)).save(image)
    (dtd_root / "roboverse_source_manifest.json").write_text(
        json.dumps({"archive_md5": "fff73e5086ae6bdbea199a49dfb8a4c1"}), encoding="utf-8"
    )
    prepare(source, output, ["box"], dtd_root=dtd_root, texture=image)
    variant = output / f"synthicl_box_dtd_{hashlib.sha256(image.read_bytes()).hexdigest()[:10]}"
    info = json.loads((variant / "provenance.json").read_text(encoding="utf-8"))
    assert info["appearance_texture"]["path"] == "banded/sample.jpg"
    with Image.open(image) as source_image, Image.open(variant / "material_0.png") as adapted_image:
        np.testing.assert_array_equal(np.asarray(source_image.convert("RGB")), np.asarray(adapted_image.convert("RGB")))
    with pytest.raises(FileExistsError):
        prepare(source, output, ["box"])
    texture_path = obj / "material_0.png"
    original_texture = texture_path.read_bytes()
    texture_path.write_bytes(b"x" + original_texture[1:])
    assert verify_source(source)["invalid"] == ["objects/box/material_0.png"]
    with pytest.raises(ValueError, match="checksum"):
        prepare(source, tmp_path / "corrupt", ["box"])
    texture_path.write_bytes(original_texture)
    np.save(obj / "grasps.npy", np.empty((0, 17)))
    write_source_manifest()
    with pytest.raises(ValueError, match="nonempty"):
        prepare(source, tmp_path / "empty", ["box"])


@pytest.mark.general
def test_prepared_asset_keeps_provenance_and_refuses_overwrite(tmp_path, monkeypatch):
    trimesh = pytest.importorskip("trimesh")
    zstandard = pytest.importorskip("zstandard")
    Image = pytest.importorskip("PIL.Image")
    from tools.icl import prepare_assets

    mesh = trimesh.creation.box(extents=[0.08, 0.06, 0.04])
    image = io.BytesIO()
    Image.new("RGB", (2, 2), (255, 0, 0)).save(image, format="PNG")
    files = {
        "collision.ply": mesh.export(file_type="ply"),
        "textured.obj": mesh.export(file_type="obj").encode(),
        "material_0.mtl": b"newmtl material_0\nmap_Kd material_0.png\n",
        "material_0.png": image.getvalue(),
    }
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w") as tar:
        for name, data in files.items():
            member = tarfile.TarInfo(f"assets/mani_skill2_ycb/models/test_box/{name}")
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    archive = tmp_path / "assets.tar.zst"
    archive.write_bytes(zstandard.ZstdCompressor().compress(tar_bytes.getvalue()))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    monkeypatch.setattr(prepare_assets, "_ARCHIVE_SHA256", digest)
    output = tmp_path / "objects"
    prepare_assets.prepare_assets(archive=archive, output=output, objects=["test_box"])
    asset = output / "test_box"
    provenance = json.loads((asset / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["source_archive_sha256"] == digest
    assert provenance["mass_kg"] == pytest.approx(0.1)
    assert (asset / "object.xml").is_file() and (asset / "object.urdf").is_file()
    assert trimesh.load(asset / "collision.obj", force="mesh").is_watertight
    with pytest.raises(FileExistsError):
        prepare_assets.prepare_assets(archive=archive, output=output, objects=["test_box"])


@pytest.mark.general
def test_trajectory_pair_uses_independent_sources_and_blocks_split_leaks(tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    from roboverse_learn.il.icl.data import ACTION_SPEC, IMAGE_KEYS, validate_splits, write_pair, write_trajectory

    def make(name, *, origin, goal="left", split="train", asset="a"):
        count = 6
        actions = np.zeros((count, 7), dtype=np.float32)
        actions[:, 6] = 1
        actions[0, 0] = 0.01
        metadata = {
            "origin_id": origin,
            "task": "icl.push_to_region",
            "goal": goal,
            "split": split,
            "object_id": "box",
            "asset_sha256": asset * 64,
            "simulator": "mujoco",
            "control_period": 1 / 30,
            "camera_roles": {
                "images_front": "external_front",
                "images_side": "external_side",
                "images_wrist": "external_overhead",
            },
            "success": True,
            "physical_record": f"{origin}.npz",
            "action_spec": ACTION_SPEC,
            "phase_schema": ["start", "approach", "lower", "push", "lift", "finish"],
        }
        images = {key: np.full((count, 8, 8, 3), index, dtype=np.uint8) for index, key in enumerate(IMAGE_KEYS)}
        return write_trajectory(
            tmp_path / "raw" / f"{name}.hdf5",
            metadata=metadata,
            images=images,
            robot_states=np.zeros((count, 7)),
            actions=actions,
            waypoints_idx=np.arange(count),
            assigned_wps=np.arange(count),
        )

    reference = make("ref", origin="episode_a")
    query = make("query", origin="episode_b")
    duplicate = make("duplicate", origin="episode_a")
    different_goal = make("other_goal", origin="episode_c", goal="right")
    test_asset_leak = make("test", origin="episode_d", split="test")
    pair = write_pair(tmp_path / "pairs" / "episode_0.h5", reference=reference, query=query)
    with h5py.File(pair, "r") as file:
        assert file["trajectory_0/actions"].shape == (6, 7)
        assert file["trajectory_1/images_front"].shape == (6, 8, 8, 3)
    with pytest.raises(ValueError, match="independent"):
        write_pair(tmp_path / "pairs" / "duplicate.h5", reference=reference, query=duplicate)
    with pytest.raises(ValueError, match="goal"):
        write_pair(tmp_path / "pairs" / "mismatch.h5", reference=reference, query=different_goal)
    with pytest.raises(ValueError, match="asset"):
        validate_splits([reference, test_asset_leak])
    test_asset_leak.unlink()
    write_pair(tmp_path / "pairs" / "episode_1.h5", reference=reference, query=query)
    from tools.icl import train

    monkeypatch.setattr(
        train,
        "read_header",
        lambda path: {
            "num_steps": 5,
            "info": {"success": True, "task": "icl.push_to_region", "goal": "left"},
            "provenance": {"simulator": "mujoco", "env_step_s": 1 / 30},
        },
    )
    with pytest.raises(ValueError, match="Origin reused"):
        train.validate_training_data(tmp_path / "pairs", tmp_path / "raw")


@pytest.mark.general
@pytest.mark.parametrize("mode", ["stretch", "letterbox", "center_crop"])
def test_icl_camera_resize_plan_maps_pixel_centers_and_preserves_intrinsics(mode):
    from roboverse_learn.il.icl.data import _plan_calibration, image_preprocessing_plan

    plan = image_preprocessing_plan((400, 640), (256, 256), mode=mode)
    k = np.array([[320.0, 0, 319.5], [0, 318.0, 199.5], [0, 0, 1]])
    calibration = {
        "intrinsic": k.tolist(),
        "intrinsic_provenance": "source_nominal",
        "projection_model": "fisheye_kannala_brandt",
        "distortion_coefficients": [0.1, -0.02, 0.003, 0.0],
    }
    _plan_calibration(plan, calibration)
    affine = np.asarray(plan["source_to_output_pixel_affine"])
    np.testing.assert_allclose(plan["calibration"]["output_intrinsic"], affine @ k)
    assert plan["calibration"]["input"] == calibration
    np.testing.assert_allclose(affine @ [319.5, 199.5, 1], [127.5, 127.5, 1])
    assert "no undistortion" in plan["lens_distortion"]
    if mode == "letterbox":
        assert plan["resized_hw"] == [160, 256] and plan["padding_ltrb"] == [0, 48, 0, 48]
    elif mode == "center_crop":
        assert plan["resized_hw"] == [256, 410] and plan["crop_ltrb"] == [77, 0, 77, 0]
    else:
        assert plan["resized_hw"] == [256, 256]
    with pytest.raises(ValueError, match="provenance"):
        _plan_calibration(plan, {"intrinsic": k.tolist()})
    with pytest.raises(ValueError, match="not physical G1 calibration"):
        _plan_calibration(plan, {"intrinsic": k.tolist(), "intrinsic_provenance": "physical_calibration",
                                 "calibration_source": "borrowed_g2"})


@pytest.mark.general
def test_icl_letterbox_and_center_crop_change_only_expected_pixels():
    from roboverse_learn.il.icl.data import image_preprocessing_plan, preprocess_rgb_images

    image = np.arange(1, 9, dtype=np.uint8).reshape(1, 2, 4, 1).repeat(3, axis=-1)
    letterboxed = preprocess_rgb_images(image, image_preprocessing_plan((2, 4), (4, 4), mode="letterbox"))
    np.testing.assert_array_equal(letterboxed[:, 0], 0)
    np.testing.assert_array_equal(letterboxed[:, 1:3], image)
    np.testing.assert_array_equal(letterboxed[:, 3], 0)
    cropped = preprocess_rgb_images(image, image_preprocessing_plan((2, 4), (2, 2), mode="center_crop"))
    np.testing.assert_array_equal(cropped, image[:, :, 1:3])


@pytest.mark.general
def test_icl_native_mixed_resolution_and_traceable_synthicl_preprocessing(tmp_path, monkeypatch):
    import hashlib

    h5py = pytest.importorskip("h5py")
    from roboverse_learn.il.icl.data import (
        ACTION_SPEC,
        HEAD_CAMERA_ROLES,
        IMAGE_KEYS,
        MASK_KEYS,
        preprocess_trajectory,
        read_metadata,
        resolve_physical_record,
        write_pair,
        write_trajectory,
    )

    source_dir = tmp_path / "library" / "native"
    transformed_dir = tmp_path / "library" / "training"
    camera_hw = {"images_front": (400, 640), "images_wrist": (403, 640), "images_side": (512, 640)}
    calibrations = {
        "images_front": {"intrinsic": [[300, 0, 319.5], [0, 300, 199.5], [0, 0, 1]],
                         "intrinsic_provenance": "renderer_reported", "projection_model": "pinhole"},
        "images_wrist": {"intrinsic": [[275, 0, 319.5], [0, 275, 201], [0, 0, 1]],
                         "intrinsic_provenance": "source_nominal", "projection_model": "fisheye_kb4",
                         "calibration_source": "borrowed_g2",
                         "distortion_coefficients": [0.11, 0.02, -0.03, 0.004]},
        "images_side": {"projection_model": "pinhole_approximation", "intrinsic_provenance": "not_available"},
    }

    def record(origin):
        from types import SimpleNamespace

        import torch

        from metasim.types import CameraState
        from tools.icl.collect import _capture_valid_masks

        count = 3
        actions = np.zeros((count, 7), dtype=np.float32)
        actions[:, 6] = 1
        meta = {"origin_id": origin, "task": "icl.push_to_region", "goal": "left", "split": "train",
                "object_id": "box", "asset_sha256": "a" * 64, "simulator": "mujoco", "control_period": 1 / 30,
                "camera_roles": HEAD_CAMERA_ROLES, "camera_calibrations": calibrations, "success": True,
                "camera_valid_masks": {"images_wrist": {"source": "borrowed_g2_fisheye_projection"}},
                "physical_record": f"{origin}.npz", "action_spec": ACTION_SPEC, "phase_schema": ["start", "finish"]}
        images = {key: np.full((count, *camera_hw[key], 3), [50, 100, 150], dtype=np.uint8) for key in IMAGE_KEYS}
        mask = np.ones(camera_hw["images_wrist"], dtype=np.bool_)
        mask[:, :100] = False
        state = SimpleNamespace(cameras={
            key: CameraState(rgb=torch.from_numpy(images[key][:1]), depth=None,
                             valid_mask=torch.from_numpy(mask[None]) if key == "images_wrist" else None)
            for key in IMAGE_KEYS
        })
        captured = {}
        _capture_valid_masks(state, {key: key for key in IMAGE_KEYS}, captured)
        _capture_valid_masks(state, {key: key for key in IMAGE_KEYS}, captured)
        assert captured["images_front"] is None
        state.cameras["images_wrist"].valid_mask[0, 0, 0] = True
        with pytest.raises(ValueError, match="changed during recording"):
            _capture_valid_masks(state, {key: key for key in IMAGE_KEYS}, captured)
        return write_trajectory(source_dir / f"{origin}.hdf5", metadata=meta, images=images,
                                robot_states=np.zeros((count, 7)), actions=actions,
                                valid_masks={key: value for key, value in captured.items() if value is not None},
                                waypoints_idx=np.array([0, 2]), assigned_wps=np.array([0, 0, 1]))

    raw1, raw2 = record("a"), record("b")
    before = hashlib.sha256(raw1.read_bytes()).hexdigest()
    with h5py.File(raw1, "r") as h5:
        assert {key: tuple(h5["trajectory"][key].shape[1:3]) for key in IMAGE_KEYS} == camera_hw
    assert read_metadata(raw1)["camera_native_hw"] == {key: list(value) for key, value in camera_hw.items()}
    processed1 = preprocess_trajectory(raw1, transformed_dir / "a.hdf5", mode="letterbox", batch_size=1)
    processed2 = preprocess_trajectory(raw2, transformed_dir / "b.hdf5", mode="letterbox", batch_size=2)
    assert hashlib.sha256(raw1.read_bytes()).hexdigest() == before
    import subprocess
    import sys

    project = Path(__file__).resolve().parents[1]
    cli_output = transformed_dir / "cli_stretch.hdf5"
    cli = subprocess.run(
        [sys.executable, "tools/icl/collect.py", "preprocess", "--source", str(raw1), "--output", str(cli_output),
         "--mode", "stretch", "--height", "16", "--width", "16"],
        cwd=project, capture_output=True, text=True, timeout=60, check=False,
    )
    assert cli.returncode == 0, cli.stderr
    with h5py.File(cli_output, "r") as h5:
        assert h5["trajectory/images_front"].shape == (3, 16, 16, 3)
    meta = read_metadata(processed1)
    assert meta["image_preprocessing"]["raw_source"]["sha256"] == before
    assert meta["image_preprocessing"]["cameras"]["images_wrist"]["calibration"]["input"] == calibrations["images_wrist"]
    assert meta["image_preprocessing"]["cameras"]["images_wrist"]["valid_mask"]["source"] == "borrowed_g2_fisheye_projection"
    assert meta["image_preprocessing"]["cameras"]["images_side"]["calibration"]["output_intrinsic"] is None
    assert resolve_physical_record(processed1, meta, library=tmp_path / "library") == source_dir / "a.npz"
    with h5py.File(processed1, "r") as h5:
        for key in IMAGE_KEYS:
            frames = h5["trajectory"][key][:]
            assert frames.shape == (3, 256, 256, 3)
            np.testing.assert_array_equal(frames[0, 128, 128], [50, 100, 150])
            np.testing.assert_array_equal(frames[0, 0, 0], [0, 0, 0])
            mask = h5["trajectory"][MASK_KEYS[key]][:]
            assert mask.dtype == np.bool_ and mask.shape == (256, 256)
            assert not mask[0, 0] and mask[128, 128]
        assert not h5["trajectory"][MASK_KEYS["images_wrist"]][128, 10]

    pair = write_pair(tmp_path / "pairs" / "episode_0.h5", reference=processed1, query=processed2)
    with h5py.File(pair, "r") as h5:
        assert h5["trajectory_0/images_front"].shape == (3, 256, 256, 3)
    with pytest.raises(ValueError, match="image preprocessing"):
        write_pair(tmp_path / "pairs" / "mixed.h5", reference=raw1, query=processed2)

    synthicl = Path(__file__).resolve().parents[2] / "icl" / "synthicl"
    if (synthicl / "icil/data/pseudo_demo_dataset.py").is_file():
        monkeypatch.syspath_prepend(str(synthicl))
        from icil.config.dataset_config import DatasetConfig
        from icil.config.shared_config import SharedConfig
        from icil.data.pseudo_demo_dataset import PseudoDemoDataset

        from tools.icl.evaluate import image_tensor

        dataset = PseudoDemoDataset.__new__(PseudoDemoDataset)
        dataset.image_size = (256, 256)
        for intensity in (1, 128):
            probe = np.full((1, 4, 6, 3), intensity, dtype=np.uint8)
            np.testing.assert_allclose(image_tensor(probe, (256, 256), "cpu").numpy(), dataset.resize_images(probe))
        from roboverse_learn.il.icl.data import image_preprocessing_plan, preprocess_rgb_images

        textured = np.random.default_rng(42).integers(0, 256, (1, 7, 11, 3), dtype=np.uint8)
        dataset.image_size = (6, 6)
        stretched = preprocess_rgb_images(textured, image_preprocessing_plan((7, 11), (6, 6), mode="stretch"))
        np.testing.assert_allclose(dataset.resize_images(stretched), dataset.resize_images(textured), atol=0.5 / 255 + 1e-6)
        dataset.image_size = (256, 256)
        with h5py.File(raw1, "r") as raw:
            for key in IMAGE_KEYS:
                assert dataset.resize_images(raw["trajectory"][key][:]).shape == (3, 256, 256, 3)
        with h5py.File(pair, "r") as h5:
            for key in IMAGE_KEYS:
                array = h5[f"trajectory_0/{key}"][:]
                np.testing.assert_allclose(dataset.resize_images(array), array.astype(np.float32) / 255, atol=1e-7)
        raw_pair = write_pair(tmp_path / "raw_pairs" / "episode_0.h5", reference=raw1, query=raw2)
        for dataset_dir in (pair.parent, raw_pair.parent):
            loader = PseudoDemoDataset(
                DatasetConfig(hdf5_path=str(dataset_dir), train_val_split=1.0, num_prompt_traj=1, max_n_prompts=1,
                              label_keys=[], waypoint_key="waypoints_idx"),
                SharedConfig(image_keys=list(IMAGE_KEYS), image_size=(256, 256), num_traj_per_task=2,
                             prompt_length=2, task_length=3, n_hist_steps=1, n_pred_steps=1,
                             auxiliary_key="assigned_wps", obs_keys=[], device="cpu"),
                split="train",
            )
            sample = loader[0]
            assert sample is not None
            for key in IMAGE_KEYS:
                assert tuple(sample[key].shape) == (3, 1, 256, 256, 3)
                assert tuple(sample[f"prompt_{key}"].shape) == (2, 256, 256, 3)

    # Unknown fields are not silently discarded, because they may be image masks.
    with h5py.File(raw2, "r+") as h5:
        h5["trajectory"].create_dataset("unknown_mask", data=np.zeros((3, 2, 2), dtype=np.uint8))
    rejected = transformed_dir / "unsupported.hdf5"
    with pytest.raises(ValueError, match="unrecognized trajectory fields"):
        preprocess_trajectory(raw2, rejected)
    assert not rejected.exists()
    assert not list(transformed_dir.glob(f".{rejected.name}.*.tmp"))

    # Tampering with the raw file is detected before its physical record is used.
    with raw1.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        resolve_physical_record(processed1, meta, library=tmp_path / "library")


@pytest.mark.general
def test_synthicl_training_validation_uses_original_record_and_one_image_policy(tmp_path, monkeypatch):
    from roboverse_learn.il.icl.data import (
        ACTION_SPEC,
        HEAD_CAMERA_ROLES,
        IMAGE_KEYS,
        preprocess_trajectory,
        write_pair,
        write_trajectory,
    )
    from tools.icl import train

    library, pairs = tmp_path / "library", tmp_path / "pairs"
    raw, prepared = library / "native", library / "prepared"

    def trajectory(origin):
        actions = np.zeros((6, 7), dtype=np.float32)
        actions[:, 6] = 1
        meta = {"origin_id": origin, "task": "icl.push_to_region", "goal": "left", "split": "train",
                "object_id": "box", "asset_sha256": "a" * 64, "simulator": "mujoco", "control_period": 1 / 30,
                "camera_roles": HEAD_CAMERA_ROLES, "success": True, "physical_record": f"{origin}.npz",
                "action_spec": ACTION_SPEC, "phase_schema": ["a", "b", "c", "d", "e", "f"]}
        images = {key: np.full((6, 8 + index, 12, 3), 64, dtype=np.uint8) for index, key in enumerate(IMAGE_KEYS)}
        return write_trajectory(raw / f"{origin}.hdf5", metadata=meta, images=images,
                                robot_states=np.zeros((6, 7)), actions=actions,
                                waypoints_idx=np.arange(6), assigned_wps=np.arange(6))

    processed = []
    for name in ("a", "b", "c", "d"):
        source = trajectory(name)
        processed.append(preprocess_trajectory(source, prepared / f"{name}.hdf5"))
    write_pair(pairs / "episode_0.h5", reference=processed[0], query=processed[1])
    write_pair(pairs / "episode_1.h5", reference=processed[2], query=processed[3])
    observed_records = []

    def read_header(path):
        observed_records.append(path)
        return {"num_steps": 5, "info": {"success": True, "task": "icl.push_to_region", "goal": "left"},
                "provenance": {"simulator": "mujoco", "env_step_s": 1 / 30}}

    monkeypatch.setattr(train, "read_header", read_header)
    assert train.validate_training_data(pairs, library) == 2
    assert all(path.parent == raw for path in observed_records)
    raw_e, raw_f = trajectory("e"), trajectory("f")
    write_pair(pairs / "episode_2.h5", reference=raw_e, query=raw_f)
    with pytest.raises(ValueError, match="different image preprocessing policies"):
        train.validate_training_data(pairs, library)
