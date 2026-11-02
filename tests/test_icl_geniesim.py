"""Checks for the Genie Sim asset conversion prototype."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

_GENIESIM_CATALOG_PATH = Path(__file__).resolve().parents[1] / "roboverse_pack/tasks/icl/benchmark_catalog.json"
GENIESIM_G2_RUN_IDS = tuple(
    run["id"]
    for run in json.loads(_GENIESIM_CATALOG_PATH.read_text(encoding="utf-8"))["runs"]
    if run["robot"] == "g2_op"
)


def test_geniesim_g1_source_usd_drive_parameters_convert_angular_units(tmp_path):
    pytest.importorskip("pxr")
    from tools.icl.prepare_geniesim_g1 import _source_drive_parameters

    usd = tmp_path / "robot.usda"
    usd.write_text(
        "#usda 1.0\n"
        'def Xform "G01" {\n'
        '  over "joints" {\n'
        '    over "idx01_body_joint1" {\n'
        "      float drive:linear:physics:damping = 1000\n"
        "      float drive:linear:physics:maxForce = 100\n"
        "      float drive:linear:physics:stiffness = 10000\n"
        "    }\n"
        '    over "idx61_arm_r_joint1" {\n'
        "      float drive:angular:physics:damping = 500\n"
        "      float drive:angular:physics:stiffness = 10000\n"
        "    }\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    drives = _source_drive_parameters(usd)
    assert drives["idx01_body_joint1"] == {
        "stiffness": pytest.approx(10000.0),
        "damping": pytest.approx(1000.0),
        "effort_limit_sim": pytest.approx(100.0),
    }
    assert drives["idx61_arm_r_joint1"]["stiffness"] == pytest.approx(10000 * 180 / np.pi)
    assert drives["idx61_arm_r_joint1"]["damping"] == pytest.approx(500 * 180 / np.pi)
    assert drives["idx61_arm_r_joint1"]["effort_limit_sim"] is None


def test_geniesim_g1_mjcf_actuators_preserve_source_drive_parameters(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    from tools.icl.prepare_geniesim_g1 import _replace_mjcf_actuators

    mjcf = tmp_path / "robot.xml"
    mjcf.write_text(
        '<mujoco><worldbody><body name="base">'
        '<joint name="idx01_body_joint1" type="slide" axis="0 0 1" range="0 0.5"/>'
        '<geom type="box" size="0.1 0.1 0.1" mass="1"/>'
        '</body></worldbody><actuator><position name="old" joint="idx01_body_joint1" kp="1"/></actuator></mujoco>',
        encoding="utf-8",
    )
    _replace_mjcf_actuators(
        mjcf,
        {"idx01_body_joint1": {"stiffness": 10000.0, "damping": 1000.0, "effort_limit_sim": 100.0}},
    )
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    assert model.nu == 1
    assert model.actuator_gainprm[0, 0] == pytest.approx(10000.0)
    assert model.actuator_biasprm[0, 2] == pytest.approx(-1000.0)
    assert list(model.actuator_forcerange[0]) == pytest.approx([-100.0, 100.0])


@pytest.mark.general
def test_geniesim_g1_passive_joints_keep_authored_effort_limits():
    import xml.etree.ElementTree as ET

    from roboverse_pack.tasks.icl.fruit_sorting import CONTROL_JOINTS, ROBOT_ROOT, build_geniesim_g1_robot

    if not (ROBOT_ROOT / "provenance.json").is_file():
        pytest.skip("Prepare the G1 Omnipicker before checking its passive joint limits")
    robot = build_geniesim_g1_robot(ROBOT_ROOT)
    urdf = ET.parse(ROBOT_ROOT / "G1_omnipicker.urdf").getroot()
    passive_joints = [
        joint
        for joint in urdf.findall("joint")
        if joint.get("type") in {"revolute", "prismatic"} and joint.get("name") not in CONTROL_JOINTS
    ]
    assert passive_joints
    for joint in passive_joints:
        name = joint.get("name")
        limit = joint.find("limit")
        assert name in robot.actuators
        assert robot.actuators[name].stiffness == 0.0
        assert robot.actuators[name].damping == 0.0
        authored_effort = limit.get("effort")
        expected_effort = float(authored_effort) if authored_effort is not None else None
        assert robot.actuators[name].effort_limit_sim == expected_effort


@pytest.mark.general
def test_geniesim_g1_base_visual_attachment_places_chassis_on_scene_ground():
    import xml.etree.ElementTree as ET

    from roboverse_pack.tasks.icl.fruit_sorting import APPLE_DELIVERY_SCENE_ROOT, ROBOT_ROOT, build_geniesim_g1_robot

    provenance_path = ROBOT_ROOT / "provenance.json"
    if not provenance_path.is_file():
        pytest.skip("Prepare the G1 Omnipicker v4 asset before checking its ground alignment")
    build_geniesim_g1_robot(ROBOT_ROOT)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected = provenance["visual_attachment_transforms"]["links"]["base_link"]["xyz"]
    assert expected == pytest.approx([0.0, 0.0, 0.21113469724309647])

    mjcf = ET.parse(ROBOT_ROOT / "G1_omnipicker.xml").getroot()
    base_mesh_geoms = [
        geom
        for geom in mjcf.findall(".//body[@name='base_link']/geom")
        if geom.get("mesh", "").startswith("base_link_visual_")
    ]
    assert len(base_mesh_geoms) == 3
    for geom in base_mesh_geoms:
        assert np.fromstring(geom.get("pos", "0 0 0"), sep=" ")[2] == pytest.approx(expected[2], abs=1e-6)

    scene = json.loads((APPLE_DELIVERY_SCENE_ROOT / "scene.json").read_text(encoding="utf-8"))
    root_z = float(scene["robot_initial_state"]["position"][2])
    robot = ET.parse(ROBOT_ROOT / "G1_omnipicker.urdf").getroot()
    trimesh = pytest.importorskip("trimesh")
    lowest_world_z = []
    for visual in robot.findall("link[@name='base_link']/visual"):
        mesh = visual.find("geometry/mesh")
        if mesh is None:
            continue
        origin = visual.find("origin")
        offset = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
        mesh_path = ROBOT_ROOT / mesh.get("filename")
        local_bounds = trimesh.load(mesh_path, force="mesh").bounds
        lowest_world_z.append(float(local_bounds[0, 2] + offset[2] + root_z))
    assert lowest_world_z
    assert min(lowest_world_z) == pytest.approx(0.0, abs=1e-3)

    correction = provenance["collision_ground_alignment"]
    assert correction["method"] == "shift source collision proxies to match the lowest composed chassis visual"
    assert correction["vertical_shift_m"] == pytest.approx(0.002582078783, abs=1e-8)
    collision_bottoms = []
    for collision in robot.findall("link[@name='base_link']/collision"):
        origin = collision.find("origin")
        offset = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
        box_size = np.fromstring(collision.find("geometry/box").get("size"), sep=" ")
        collision_bottoms.append(float(offset[2] - box_size[2] / 2 + root_z))
    assert collision_bottoms
    assert min(collision_bottoms) == pytest.approx(min(lowest_world_z), abs=1e-6)

    corrected_geoms = [geom for geom in mjcf.findall(".//body[@name='base_link']/geom") if geom.get("group") == "3"]
    mjcf_bottoms = [
        float(np.fromstring(geom.get("pos", "0 0 0"), sep=" ")[2] - float(geom.get("size").split()[2]) + root_z)
        for geom in corrected_geoms
    ]
    assert mjcf_bottoms
    assert min(mjcf_bottoms) == pytest.approx(min(lowest_world_z), abs=1e-6)


@pytest.mark.general
def test_geniesim_g1_mujoco_position_drive_is_stable_with_effort_limits():
    from roboverse_pack.tasks.icl.geniesim_benchmark import ROBOT_ROOT, SPEC_PATH, GenieSimBenchmarkTask

    if not SPEC_PATH.is_file() or not (ROBOT_ROOT / "provenance.json").is_file():
        pytest.skip("Prepare the source-scene benchmark snapshot and G1 robot first")
    task = GenieSimBenchmarkTask(
        run_id="g1op_s2r_select_color",
        simulator="mujoco",
        action_mode="tcp_delta",
        tcp_arm="right",
        instance_index=0,
        problem_index=0,
        image_size=64,
        headless=True,
        cameras=(),
        reset_settle_seconds=0.0,
        max_episode_steps=3,
    )
    try:
        assert task.scenario.sim_params.mujoco_actuator_limits_at_joint is True
        observation, _ = task.reset(seed=0)
        robot_state = observation.robots[task.robot_name]
        before = robot_state.joint_pos[0, task._control_indices].detach().cpu().numpy()
        tcp_before, _ = task._cartesian_controller.tcp_pose(before)

        observation, _, _, _, _ = task.step(np.asarray([0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
        robot_state = observation.robots[task.robot_name]
        after = robot_state.joint_pos[0, task._control_indices].detach().cpu().numpy()
        wrist_index = task.control_joint_names.index("idx67_arm_r_joint7")
        tcp_after, _ = task._cartesian_controller.tcp_pose(after)

        assert abs(float(after[wrist_index] - before[wrist_index])) < 0.05
        assert np.linalg.norm(tcp_after - tcp_before) < 0.01
    finally:
        task.close()


def _write_fake_initial_pose_sources(source: Path, task_names: tuple[str, ...], monkeypatch) -> None:
    """Create minimal literal upstream reset tables for importer tests."""
    import hashlib

    from tools.icl import import_geniesim_specs as importer

    (source / "benchmark/config").mkdir(parents=True, exist_ok=True)
    (source / "benchmark/envs").mkdir(parents=True, exist_ok=True)
    (source / "utils").mkdir(parents=True, exist_ok=True)
    g1 = {"body_state": [0.1, 0.2, 0.3, 0.4], "init_arm": [0.5, 0.6], "init_hand": [0.0, 0.0]}
    g2 = {"body_state": [0.3], "head_state": [0.2], "init_arm": [0.5, 0.6], "init_hand": [0.0, 0.0]}
    tasks = {name: {"G1_omnipicker": g1, "G2_omnipicker": g2} for name in task_names}
    (source / "benchmark/config/robot_init_states.py").write_text(f"TASK_INFO_DICT = {tasks!r}\n", encoding="utf-8")
    configs = {
        "G1_omnipicker": {
            "arm_joints": ["left", "right"],
            "head_joints": ["head1", "head2"],
            "waist_joints": ["waist1", "waist2"],
            "gripper_joints": ["grip1", "grip2"],
            "init_gripper_open": [1.0, 1.0],
        },
        "G2_omnipicker": {
            "arm_joints": ["left", "right"],
            "head_joints": ["head1"],
            "waist_joints": ["waist1"],
            "gripper_joints": ["grip1", "grip2"],
            "init_gripper_open": [0.785, 0.785],
        },
    }
    (source / "utils/name_utils.py").write_text(f"ROBOT_CONFIGS = {configs!r}\n", encoding="utf-8")
    for name in ("utils/infer_pre_process.py", "benchmark/envs/dummy_env.py", "benchmark/envs/vec_env.py"):
        (source / name).write_text("# source reset semantics\n", encoding="utf-8")
    monkeypatch.setattr(
        importer,
        "_RESET_SEMANTICS_SHA256",
        {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in importer._RESET_SEMANTICS_SHA256},
    )


def test_g1_head_camera_converts_usd_row_transform():
    from tools.icl.probe_geniesim_g1 import head_rgb_camera

    rotation = Rotation.from_euler("xyz", [0.3, -0.2, 0.5])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation.as_matrix().T
    matrix[3, :3] = [0.08, -0.03, 0.02]
    cfg = head_rgb_camera(
        {
            "head_link_from_camera_usd_row_matrix": matrix.tolist(),
            "verticalAperture": 2.0,
            "horizontalAperture": 4.0,
            "focalLength": 2.0,
        },
        "agibot_g1",
        512,
    )
    assert cfg.mount_to == "agibot_g1" and cfg.mount_link == "head_link2"
    assert (cfg.width, cfg.height) == (512, 256)
    np.testing.assert_allclose(cfg.mount_pos, [0.08, -0.03, 0.02])
    np.testing.assert_allclose(Rotation.from_quat(cfg.mount_quat, scalar_first=True).as_matrix(), rotation.as_matrix())


def test_geniesim_waic_bottle_compatibility_alias_checks_provenance_and_dimensions(tmp_path):
    from roboverse_pack.tasks.icl.geniesim_scene import _resolve_scene_asset

    candidate = tmp_path / "genie_beverage_bottle_009"
    candidate.mkdir()
    (candidate / "provenance.json").write_text(
        json.dumps({
            "format": "roboverse.icl.geniesim_rigid_object.v3",
            "asset_id": "genie_beverage_bottle_009",
            "source_revision": "a0813aad7c16165daffbe6c2737754e0344809ee",
            "source_files_sha256": {"Aligned.usd": "260317e37c02b49254fc9b18392436be22f1c4d4e7088d0fd47e5fa6eb5ced0c"},
            "bounds_m": [
                [-0.024104660376906395, -0.02992143958132074, -0.0851838291789882],
                [0.024613840505480766, 0.02992026031459619, 0.08173619955460529],
            ],
        }),
        encoding="utf-8",
    )
    entry = {"description": {"dimensions": [0.04871850088238716, 0.16691981256008148, 0.05984169989824295]}}
    resolved, alias = _resolve_scene_asset(tmp_path, "waic_beverage_bottle_009", entry)
    assert resolved == "genie_beverage_bottle_009"
    assert alias["identity_verified"] is False
    assert alias["dimension_match_m"] == entry["description"]["dimensions"]

    entry["description"]["dimensions"][0] += 0.01
    with pytest.raises(ValueError, match="dimensions do not match"):
        _resolve_scene_asset(tmp_path, "waic_beverage_bottle_009", entry)
    entry["description"]["dimensions"][0] -= 0.01

    from roboverse_pack.tasks.icl.geniesim_scene import geniesim_object_bounds

    instance = {"scene_info": {"layout": {"bottle": {"usd": "waic_beverage_bottle_009", **entry}}}}
    bounds = geniesim_object_bounds(instance, tmp_path)["bottle"]
    assert bounds.shape == (2, 3)
    assert bounds[1, 2] - bounds[0, 2] == pytest.approx(0.1669198, abs=2e-6)


def test_benchmark_catalog_covers_all_pinned_run_configurations():
    root = Path(__file__).resolve().parents[1]
    catalog = json.loads((root / "roboverse_pack/tasks/icl/benchmark_catalog.json").read_text(encoding="utf-8"))
    runs = catalog["runs"]
    assert catalog["source"]["revision"] == "cd46c52d89e2ad40a62e1af7aa05b7ab1c67f7b7"
    assert catalog["source"]["revision_scope"].startswith("Task configs, evaluation configs, scene inputs")
    assert catalog["source"]["content_sha256"] == "f0cd8166a1e0fba4739f8c15500262718e55ac3cdadc83b4ec0e47e41cddb8a4"
    assert catalog["scope"]["configuration_count"] == 86
    assert catalog["scope"]["semantic_task_count"] == 36
    assert catalog["scope"]["scene_instance_count"] == 759
    assert catalog["scope"]["robot_configuration_counts"] == {"g1_op": 8, "g2_op": 78}
    assert len({entry["id"] for entry in runs}) == 86
    assert sum(entry["family"].startswith("robust_") for entry in runs) == 50
    assert all(entry["source_config_sha256"] and entry["scene_family"] and entry["subtask"] for entry in runs)
    assert all(entry["migration_status"] == "implemented_unverified" for entry in runs)
    alias_runs = [entry for entry in runs if entry["subtask"] == "pick_object_type"]
    assert len(alias_runs) == 6
    assert all(set(entry["backend_checks"].values()) == {"not_run"} for entry in runs)
    g1_tasks = [entry for entry in catalog["semantic_tasks"] if entry["robot"] == "g1_op"]
    assert len(g1_tasks) == 8
    assert all(entry["roboverse_task"] == "icl.geniesim_benchmark" for entry in g1_tasks)
    assert all(entry["migration_status"] == "implemented_unverified" for entry in g1_tasks)
    assert all(
        entry["backend_checks"][sim] == "not_run" for entry in g1_tasks for sim in ("mujoco", "genesis", "newton")
    )
    assert catalog["scope"]["behavioral_parity_claimed"] is False
    runtime = catalog["scope"]["shared_task_runtime"]
    assert runtime["roboverse_task"] == "icl.geniesim_benchmark"
    assert runtime["source_robots"] == ["g1_op", "g2_op"]
    assert runtime["joint_position_action_dimensions"] == {"g1_op": 20, "g2_op": 24}
    assert runtime["synthicl_action_dimensions"] == 7
    assert runtime["synthicl_action_spec"] == "tcp_world_delta_xyz_rotvec_gripper_open_v1"
    assert runtime["bimanual_tcp_action_dimensions"] == 14
    assert runtime["action_modes"] == ["joint_position", "tcp_delta", "bimanual_tcp_delta"]
    assert runtime["cartesian_controller_arm"] == "source_or_explicit_default_right"
    assert runtime["configured_control_rate_hz"] == 30
    assert runtime["control_and_timing_parity"] == "not_verified"
    assert runtime["supported_simulators"] == ["mujoco", "genesis", "newton"]
    assert runtime["default_cameras"] == ["head_rgb", "head_fisheye_left", "head_fisheye_right"]
    assert runtime["behavioral_parity"] == "not_verified"
    assert runtime["validation_source_revision"] == "unknown"
    assert runtime["validation_source_content_sha256"] != catalog["source"]["content_sha256"]
    assert runtime["active_import_payload_sha256"] == "2054e9269643c3240f09116b423f92d71aa687d88847bfe0f6768c406422aa35"
    assert runtime["supported_run_count"] == 86
    assert runtime["smoke_tested_run_count"] == 0
    assert runtime["source_asset_blocked_run_count"] == 0
    assert runtime["smoke_tested_with_asset_alias_run_count"] == 0
    assert runtime["historical_smoke_tested_run_count"] == 80
    assert runtime["historical_smoke_with_asset_alias_run_count"] == 6
    assert runtime["status"] == "implemented_unverified"
    assert len(catalog["source_checker_vocabulary"]) == 14
    assert catalog["checker_status"]["prototype_predicates"] == ["Inside"]
    implemented = set(catalog["checker_status"]["implemented_checkers"])
    external = set(catalog["checker_status"]["external_scoring_checkers"])
    assert set(catalog["source_checker_vocabulary"]) <= implemented | external
    assert external == {"VLM"}
    assert catalog["checker_status"]["graph_operators"] == [
        "ActionList",
        "ActionSetWaitAny",
        "ActionSetWaitAll",
        "ActionSetWaitSome",
    ]
    assert catalog["checker_status"]["parity_verified_predicates"] == []
    g2_tasks = [entry for entry in catalog["semantic_tasks"] if entry["robot"] == "g2_op"]
    assert all(entry["roboverse_task"] == "icl.geniesim_benchmark" for entry in g2_tasks)
    assert all(entry["migration_status"] == "implemented_unverified" for entry in g2_tasks)


def test_benchmark_extensions_catalog_records_user_supplied_g1_apple_scene():
    root = Path(__file__).resolve().parents[1]
    catalog = json.loads((root / "roboverse_pack/tasks/icl/benchmark_extensions.json").read_text(encoding="utf-8"))
    assert catalog["format"] == "roboverse.icl.benchmark_extensions.v1"
    extension = catalog["extensions"][0]
    assert extension["id"] == "g1_apple_scene_delivery_20261031_v4"
    assert extension["roboverse_task"] == "icl.geniesim_g1_apple_delivery"
    assert extension["source"]["archive_sha256"] == "548c5bd4760ad01ffd8f2f8c40bf32a46a4732f052610f11f4cc368ec42aadf8"
    assert extension["simulators"] == ["mujoco", "genesis", "newton"]
    assert extension["behavioral_parity"] == "not_verified"
    assert extension["migration_status"] == "three_backend_scene_smoke_tested"
    assert set(extension["backend_checks"].values()) == {"scene_smoke_tested"}


def test_icl_video_writer_pads_odd_calibrated_camera_frames_for_h264(tmp_path):
    import av

    from tools.icl.video_utils import RGBVideoWriter

    path = tmp_path / "odd_camera.mp4"
    writer = RGBVideoWriter(path, width=7, height=5, fps=30)
    writer.append(np.full((5, 7, 3), 128, dtype=np.uint8))
    writer.close()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height) == (8, 6)


@pytest.mark.examples
@pytest.mark.parametrize("simulator", ["mujoco", "genesis", "newton"])
def test_geniesim_g1_apple_delivery_builds_renders_and_holds_start_pose(simulator):
    from xml.etree import ElementTree as ET

    requested = set(os.environ.get("ROBOVERSE_ICL_TEST_BACKENDS", "").split(","))
    if simulator != "mujoco" and simulator not in requested:
        pytest.skip(f"Set ROBOVERSE_ICL_TEST_BACKENDS to include {simulator}")
    if importlib.util.find_spec(simulator) is None:
        pytest.skip(f"{simulator} is not installed")
    if simulator == "newton":
        import torch

        if not torch.cuda.is_available():
            pytest.skip("Newton integration test requires a CUDA GPU")

    from metasim.task.registry import get_task_class
    from roboverse_pack.tasks.icl import fruit_sorting

    root = Path(__file__).resolve().parents[1]
    scene_root = root / "roboverse_data/icl/scenes/g1_apple_scene_delivery_20261031_v4_r4"
    robot_root = fruit_sorting.ROBOT_ROOT
    if not (scene_root / "scene.json").is_file() or not (robot_root / "provenance.json").is_file():
        pytest.skip("Prepare the delivered G1 apple scene and pinned G1 Omnipicker model first")
    scene_metadata = json.loads((scene_root / "scene.json").read_text(encoding="utf-8"))
    assert scene_metadata["ground_rendering"].startswith("MetaSim backend ground")
    assert {item["name"] for item in scene_metadata["primitives"]} == {"table_top", "table_pedestal"}
    assert scene_metadata["omitted_source_primitives"] == ["back_wall", "ground"]
    for object_name in ("apple", "pink_plate"):
        object_root = scene_root / object_name
        urdf_root = ET.parse(object_root / "object.urdf").getroot()
        assert [link.get("name") for link in urdf_root.findall("link")] == [object_name]
        mjcf_root = ET.parse(object_root / "object.xml").getroot()
        assert mjcf_root.find("worldbody/body").get("name") == object_name
        collision_geoms = [
            geom for geom in mjcf_root.findall("worldbody/body/geom") if geom.get("name", "").startswith("collision")
        ]
        assert collision_geoms and all(geom.get("group") == "3" for geom in collision_geoms)
    assert scene_metadata["objects"]["pink_plate"]["newton_urdf"] == "object_newton.urdf"
    source_plate = ET.parse(scene_root / "pink_plate/object.urdf").getroot()
    newton_plate = ET.parse(scene_root / "pink_plate/object_newton.urdf").getroot()
    assert ET.tostring(source_plate.find("link/inertial")) == ET.tostring(newton_plate.find("link/inertial"))
    assert ET.tostring(source_plate.find("link/collision")) == ET.tostring(newton_plate.find("link/collision"))
    assert len(source_plate.findall("link/visual")) == 1
    assert len(newton_plate.findall("link/visual")) == 2
    assert scene_metadata["source"]["archive_sha256"] == (
        "548c5bd4760ad01ffd8f2f8c40bf32a46a4732f052610f11f4cc368ec42aadf8"
    )

    task = get_task_class("icl.geniesim_g1_apple_delivery")(
        simulator=simulator,
        image_size=128,
        headless=True,
    )
    try:
        assert task.scenario.add_default_ground is True
        scenario_object_names = {obj.name for obj in task.scenario.objects}
        assert scenario_object_names >= {"table_top", "table_pedestal", "apple", "pink_plate"}
        assert "back_wall" not in scenario_object_names
        observation, _ = task.reset()
        assert set(observation.cameras) == {
            "head_rgb",
            "head_fisheye_left",
            "head_fisheye_right",
            "overview",
        }
        for camera in observation.cameras.values():
            frame = camera.rgb[0].detach().cpu().numpy()
            assert frame.dtype == np.uint8 and frame.shape[-1] == 3 and frame.std() > 1.0
        initial_apple = observation.objects["apple"].root_state[0, :3].detach().cpu().numpy().copy()
        command = task.control_state(observation)[0].detach().cpu().numpy().astype(np.float32)
        for _ in range(3):
            observation, _, terminated, timeout, info = task.step(command)
            assert not bool(terminated[0])
            assert not bool(timeout[0])
            assert not bool(info["task_failure"][0])
        final_apple = observation.objects["apple"].root_state[0, :3].detach().cpu().numpy()
        assert np.isfinite(final_apple).all()
        np.testing.assert_allclose(final_apple, initial_apple, atol=0.03, rtol=0.0)
    finally:
        task.close()


def test_g1_apple_newton_plate_removes_overlapping_surface_without_changing_collision():
    """Newton receives one flat tray bottom while the source mesh and collider remain available."""
    import xml.etree.ElementTree as ET

    trimesh = pytest.importorskip("trimesh")
    from scipy.spatial import ConvexHull

    scene_root = (
        Path(__file__).resolve().parents[1] / "roboverse_data/icl/scenes/g1_apple_scene_delivery_20261031_v4_r4"
    )
    plate = scene_root / "pink_plate"
    if not (plate / "provenance.json").is_file():
        pytest.skip("Prepare the delivered G1 apple scene first")
    provenance = json.loads((plate / "provenance.json").read_text(encoding="utf-8"))["newton_visual"]
    original = trimesh.load(plate / provenance["source_visual"], force="mesh", process=False)
    remainder = trimesh.load(plate / provenance["visual_meshes"][0], force="mesh", process=False)
    surface = trimesh.load(plate / provenance["visual_meshes"][1], force="mesh", process=False)
    source_top = np.all(np.isclose(original.triangles[:, :, 2], provenance["surface_height_m"], atol=1e-7), axis=1)
    source_top &= original.face_normals[:, 2] > 0.9
    source_xy = original.triangles[source_top, :, :2].reshape(-1, 2)
    hull = ConvexHull(source_xy)
    assert original.area_faces[source_top].sum() > hull.volume * 1.5
    assert len(remainder.faces) == len(original.faces) - source_top.sum()
    np.testing.assert_allclose(remainder.bounds, original.bounds, atol=1e-7)
    np.testing.assert_allclose(
        surface.bounds[:, :2], np.stack((source_xy.min(axis=0), source_xy.max(axis=0))), atol=1e-7
    )
    assert np.allclose(surface.vertices[:, 2], provenance["surface_height_m"], atol=1e-7)
    assert np.all(surface.face_normals[:, 2] > 0.999)
    assert surface.area == pytest.approx(hull.volume, rel=1e-6)
    original_urdf = ET.parse(plate / "object.urdf").getroot()
    newton_urdf = ET.parse(plate / provenance["urdf"]).getroot()
    assert ET.tostring(original_urdf.find("link/collision")) == ET.tostring(newton_urdf.find("link/collision"))
    assert ET.tostring(original_urdf.find("link/inertial")) == ET.tostring(newton_urdf.find("link/inertial"))


def test_geniesim_spec_import_preserves_empty_scenes_and_groups_runs(tmp_path, monkeypatch):
    import yaml

    from tools.icl.import_geniesim_specs import (
        correct_source_provenance,
        import_specs,
        imported_payload_sha256,
        source_tree_content_sha256,
    )

    source = tmp_path / "source"
    _write_fake_initial_pose_sources(source, ("pick_item",), monkeypatch)
    (source / "config").mkdir(parents=True)
    (source / "benchmark/config/eval_tasks").mkdir(parents=True)
    (source / "app/robot_cfg").mkdir(parents=True)
    task_root = source / "benchmark/config/llm_task/pick_item"
    (task_root / "0").mkdir(parents=True)
    (task_root / "1").mkdir(parents=True)
    eval_config = source / "benchmark/config/eval_tasks/table.json"
    eval_config.write_text(json.dumps({"cameras": [], "robot": {"robot_cfg": "G1_omnipicker.json"}}), encoding="utf-8")
    (source / "app/robot_cfg/G1_omnipicker.json").write_text(
        json.dumps({"robot": {"init_joint_position": [0.4, None]}}), encoding="utf-8"
    )
    for run_id in ("g1op_s2r_pick_item", "g1op_robust_camposgen_pick_item"):
        (source / "config" / f"{run_id}.yaml").write_text(
            yaml.safe_dump(
                {"benchmark": {"task_name": "table", "platform": "g1_op", "sub_task_name": "pick_item"}},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
    (task_root / "0/scene_info.json").write_text("{}", encoding="utf-8")
    (task_root / "1/scene_info.json").write_text(
        json.dumps({"layout": {"apple": {"usd": "apple_001"}}, "relations": {}, "seed": 7}),
        encoding="utf-8",
    )
    for instance in ("0", "1"):
        root = task_root / instance
        (root / "problems.json").write_text('{"problem1": {"Acts": []}}', encoding="utf-8")
        (root / "instructions.json").write_text('{"instructions": []}', encoding="utf-8")
        (root / "scene.usda").write_text("#usda 1.0\n", encoding="utf-8")

    pack = import_specs(source)
    assert pack["source"]["revision"] == "unknown"
    assert pack["source"]["content_sha256"] == source_tree_content_sha256(source)
    assert pack["source"]["imported_payload_sha256"] == imported_payload_sha256(pack)
    assert pack["runs"][0]["initial_pose_robot"] == "G1_omnipicker"
    named = pack["initial_pose_source"]["joint_positions"]["pick_item"]["G1_omnipicker"]
    assert named["head2"] == 0.2 and named["waist1"] == 0.3 and named["grip1"] == 1.0
    assert pack["source"]["run_configuration_count"] == 2
    assert pack["source"]["semantic_task_count"] == 1
    assert pack["source"]["scene_assignment_count"] == 2
    assert pack["source"]["empty_scene_info_count"] == 1
    assert pack["asset_ids"] == ["apple_001"]
    assert len(pack["semantic_tasks"][0]["run_ids"]) == 2
    assert pack["instances"]["pick_item/0"]["checker_graphs"] == {"problem1": {"Acts": []}}
    assert pack["robot_configs"]["G1_omnipicker.json"]["config"]["robot"]["init_joint_position"] == [0.4, None]
    output = tmp_path / "imported"
    output.mkdir()
    pack["source"]["revision"] = "incorrect-parent-commit"
    (output / "benchmark.json").write_text(json.dumps(pack), encoding="utf-8")
    corrected = correct_source_provenance(source_root=source, output=output)
    assert corrected["source"]["revision"] == "unknown"
    assert corrected["source"]["revision_correction"]["previous_revision"] == "incorrect-parent-commit"
    corrected["instances"]["pick_item/0"]["checker_graphs"]["problem1"]["Acts"] = ["tampered"]
    (output / "benchmark.json").write_text(json.dumps(corrected), encoding="utf-8")
    with pytest.raises(ValueError, match="payload does not match"):
        correct_source_provenance(source_root=source, output=output)


def test_geniesim_importer_recovers_layout_from_unloaded_usd_payloads(tmp_path):
    pytest.importorskip("pxr")
    from tools.icl.import_geniesim_specs import _scene_layout_from_usd

    scene_path = tmp_path / "scene.usda"
    scene_path.write_text(
        """#usda 1.0
(
    metersPerUnit = 0.1
    upAxis = "Z"
)
def "World" {
    def Xform "Objects" {
        double3 xformOp:translate = (10, 20, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def "table_001" (
            prepend payload = @/geniesim_assets/objects/benchmark/table/benchmark_table_019/Aligned.usda@
        ) {
            double3 xformOp:translate = (1, 2, 3)
            quatf xformOp:orient = (0.70710678, 0, 0, 0.70710678)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
        }
    }
}
""",
        encoding="utf-8",
    )

    layout = _scene_layout_from_usd(scene_path)
    assert set(layout) == {"table_001"}
    assert layout["table_001"]["usd"] == "benchmark_table_019"
    np.testing.assert_allclose(layout["table_001"]["xyz"], [1.1, 2.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(layout["table_001"]["xyzw"], [0, 0, 0.70710678, 0.70710678], atol=1e-6)


def test_geniesim_importer_derives_only_empty_scene_layouts(tmp_path, monkeypatch):
    pytest.importorskip("pxr")
    import yaml

    from tools.icl.import_geniesim_specs import import_specs

    source = tmp_path / "source"
    _write_fake_initial_pose_sources(source, ("chip_handover",), monkeypatch)
    (source / "config").mkdir(parents=True)
    (source / "benchmark/config/eval_tasks").mkdir(parents=True)
    (source / "app/robot_cfg").mkdir(parents=True)
    task_root = source / "benchmark/config/llm_task/chip_handover"
    task_root.mkdir(parents=True)
    (source / "benchmark/config/eval_tasks/table.json").write_text(
        json.dumps({"robot": {"robot_cfg": "G1_omnipicker.json"}}), encoding="utf-8"
    )
    (source / "app/robot_cfg/G1_omnipicker.json").write_text('{"robot": {}}', encoding="utf-8")
    (source / "config/g1op_s2r_chip_handover.yaml").write_text(
        yaml.safe_dump({"benchmark": {"task_name": "table", "platform": "g1_op", "sub_task_name": "chip_handover"}}),
        encoding="utf-8",
    )
    instance_root = task_root / "0"
    instance_root.mkdir()
    (instance_root / "scene_info.json").write_text("{}", encoding="utf-8")
    (instance_root / "problems.json").write_text("{}", encoding="utf-8")
    (instance_root / "instructions.json").write_text("{}", encoding="utf-8")
    (instance_root / "scene.usda").write_text(
        """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def "World" {
    def "Objects" {
        def "chip_a" (
            prepend payload = @/geniesim_assets/objects/genie/chips/Aligned.usda@
        ) {
            double3 xformOp:translate = (0.1, 0.2, 0.3)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
    }
}
""",
        encoding="utf-8",
    )

    pack = import_specs(source)
    instance = pack["instances"]["chip_handover/0"]
    assert pack["source"]["empty_scene_info_count"] == 1
    assert pack["source"]["recovered_scene_layout_count"] == 1
    assert instance["scene_info_was_empty"] is True
    assert instance["scene_layout_source"] == "scene.usda_payloads"
    assert instance["scene_info"]["layout"]["chip_a"]["usd"] == "chips"
    np.testing.assert_allclose(instance["scene_info"]["layout"]["chip_a"]["xyz"], [0.1, 0.2, 0.3])
    assert pack["asset_ids"] == ["chips"]


def _write_geniesim_rigid_fixture(root: Path, asset_id: str = "benchmark_apple_001") -> Path:
    asset = root / asset_id
    asset.mkdir(parents=True)
    (asset / "visual.obj").write_text("o visual\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    (asset / "object.xml").write_text(
        '<mujoco model="source"><asset><mesh name="visual" file="visual.obj"/></asset>'
        '<worldbody><body name="source"><freejoint/><geom type="mesh" mesh="visual"/></body>'
        "</worldbody></mujoco>",
        encoding="utf-8",
    )
    (asset / "object.urdf").write_text(
        '<robot name="source"><link name="source"><visual><geometry>'
        '<mesh filename="visual.obj"/></geometry></visual></link></robot>',
        encoding="utf-8",
    )
    (asset / "provenance.json").write_text(
        json.dumps({
            "format": "roboverse.icl.geniesim_rigid_object.v3",
            "asset_id": asset_id,
            "bounds_m": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        }),
        encoding="utf-8",
    )
    return asset


def test_geniesim_scene_adapter_builds_instance_wrappers_and_source_pose(tmp_path):
    import xml.etree.ElementTree as ET

    from metasim.scenario.robot import RobotCfg
    from roboverse_pack.tasks.icl.geniesim_scene import build_scene_scenario

    assets = tmp_path / "assets"
    _write_geniesim_rigid_fixture(assets)
    robot_file = tmp_path / "dummy.urdf"
    robot_file.write_text('<robot name="dummy"/>', encoding="utf-8")
    xyzw = (0.0, 0.0, 2.0, 2.0)
    instance = {
        "task": "pick_item",
        "instance_id": "7",
        "scene_info": {
            "layout": {
                "apple": {"usd": "benchmark_apple_001", "xyz": [0.3, -0.1, 0.7], "xyzw": xyzw},
                "table": {"usd": "benchmark_apple_001", "xyz": [0.0, 0.0, 0.4], "xyzw": [0.0, 0.0, 0.0, 1.0]},
            }
        },
    }
    robot = RobotCfg(name="agibot_g1", urdf_path=str(robot_file), mjcf_path=str(robot_file), num_joints=0)
    scenario = build_scene_scenario(
        instance=instance,
        simulator="newton",
        robot=robot,
        assets_root=assets,
        cache_root=tmp_path / "cache",
        fixed_object_names=("table",),
        robot_position=(-0.8, 0.0, -0.01),
        robot_orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )

    floor, backdrop, apple, table = scenario.objects
    assert floor.name == "geniesim_studio_floor"
    assert backdrop.name == "geniesim_studio_backdrop"
    assert backdrop.fix_base_link is True
    assert floor.fix_base_link is True and scenario.add_default_ground is False
    assert scenario.sim_params.mujoco_use_scenario_lights is True
    assert apple.name == "apple" and apple.default_position == (0.3, -0.1, 0.7)
    np.testing.assert_allclose(apple.default_orientation, (np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)))
    assert apple.file_type["mujoco"] == apple.file_type["genesis"] == "mjcf"
    assert apple.file_type["newton"] == "urdf"
    assert table.fix_base_link is True
    assert scenario.robots[0].default_position == (-0.8, 0.0, -0.01)
    apple_wrapper = Path(apple.mjcf_path)
    assert apple_wrapper.parent.name.startswith("000_apple_")
    mjcf = ET.parse(apple_wrapper).getroot()
    assert mjcf.get("model") == "apple"
    assert mjcf.find("worldbody/body").get("name") == "apple"
    assert Path(mjcf.find("asset/mesh").get("file")).is_absolute()
    assert ET.parse(apple.urdf_path).getroot().find("link").get("name") == "apple"


def test_geniesim_billiards_wrapper_uses_spherical_collision_and_scaled_inertia(tmp_path):
    import xml.etree.ElementTree as ET

    from roboverse_pack.tasks.icl.geniesim_scene import _instance_wrapper, _sphere_collision_proxy

    asset_id = "benchmark_billiards_007"
    asset = tmp_path / "assets" / asset_id
    asset.mkdir(parents=True)
    (asset / "visual.obj").write_text("o ball\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    (asset / "collision.obj").write_text("o box\nv -1 -1 -1\nv 1 1 1\nf 1 2 3\n", encoding="utf-8")
    (asset / "object.xml").write_text(
        '<mujoco model="ball"><asset><mesh name="visual" file="visual.obj"/>'
        '<mesh name="collision" file="collision.obj"/></asset><worldbody><body name="ball">'
        '<freejoint/><inertial pos="0 0 0" mass="0.01" diaginertia="0.000005453 0.000005453 0.000005453"/>'
        '<geom name="visual" type="mesh" mesh="visual" contype="0" conaffinity="0"/>'
        '<geom name="collision" type="mesh" mesh="collision"/></body></worldbody></mujoco>',
        encoding="utf-8",
    )
    (asset / "object.urdf").write_text(
        '<robot name="ball"><link name="ball"><inertial><origin xyz="0 0 0"/>'
        '<mass value="0.01"/><inertia ixx="0.000005453" ixy="0" ixz="0" '
        'iyy="0.000005453" iyz="0" izz="0.000005453"/></inertial>'
        '<visual><geometry><mesh filename="visual.obj"/></geometry></visual>'
        '<collision><geometry><mesh filename="collision.obj"/></geometry></collision></link></robot>',
        encoding="utf-8",
    )
    provenance = {
        "format": "roboverse.icl.geniesim_rigid_object.v3",
        "asset_id": asset_id,
        "bounds_m": [[-0.0286, -0.0286, -0.0286], [0.0286, 0.0286, 0.0286]],
        "mass_kg": 0.01,
    }
    (asset / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")

    proxy = _sphere_collision_proxy(asset_id, provenance)
    assert proxy is not None and proxy["radius_m"] == pytest.approx(0.0286)
    expected_inertia = 0.4 * 0.01 * 0.0286**2
    assert proxy["inertia_diagonal_kg_m2"] == pytest.approx([expected_inertia] * 3)
    scale = 1.25
    wrapper = _instance_wrapper(
        asset.parent,
        asset_id,
        "ball_instance",
        tmp_path / "cache" / "ball",
        source_scene_payload={"uniform_scale": scale},
        collision_proxy=proxy,
    )
    mjcf = ET.parse(wrapper / "object.xml").getroot()
    collision = mjcf.find("worldbody/body/geom[@name='collision']")
    assert collision.get("type") == "sphere"
    assert float(collision.get("size")) == pytest.approx(0.0286 * scale)
    assert mjcf.find("asset/mesh[@name='collision']") is None
    mjcf_inertia = [float(value) for value in mjcf.find("worldbody/body/inertial").get("diaginertia").split()]
    assert mjcf_inertia == pytest.approx([expected_inertia * scale**2] * 3)
    urdf = ET.parse(wrapper / "object.urdf").getroot()
    assert float(urdf.find(".//collision/geometry/sphere").get("radius")) == pytest.approx(0.0286 * scale)
    urdf_inertia = urdf.find(".//inertia")
    assert float(urdf_inertia.get("ixx")) == pytest.approx(expected_inertia * scale**2)
    manifest = json.loads((wrapper / "instance.json").read_text(encoding="utf-8"))
    assert manifest["collision_proxy"] == proxy
    assert manifest["uniform_scale"] == scale


def test_geniesim_fixed_object_detection_recognizes_source_asset_ids():
    from roboverse_pack.tasks.icl.geniesim_scene import _source_marks_fixed

    assert _source_marks_fixed("table_39103499", "benchmark_table_019", {}, set())
    assert _source_marks_fixed("iros_drawer_cabinet_3eb4e98c", "iros_drawer_cabinet_000", {}, set())


def test_geniesim_static_articulation_wrapper_keeps_source_part_colliders(tmp_path):
    import xml.etree.ElementTree as ET

    from tools.icl.prepare_geniesim_objects import _write_wrappers

    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
    _write_wrappers(
        tmp_path,
        "static_cabinet",
        1.0,
        np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.float64),
        [],
        [(vertices, faces), (vertices + 1, faces)],
    )
    mjcf = ET.parse(tmp_path / "object.xml").getroot()
    assert [mesh.get("name") for mesh in mjcf.findall("asset/mesh")] == ["collision_0", "collision_1"]
    assert [geom.get("mesh") for geom in mjcf.findall("worldbody/body/geom")] == ["collision_0", "collision_1"]
    urdf = ET.parse(tmp_path / "object.urdf").getroot()
    assert [node.get("filename") for node in urdf.findall("link/collision/geometry/mesh")] == [
        "collision_0.obj",
        "collision_1.obj",
    ]


def test_geniesim_articulation_converter_preserves_laptop_and_door_joint_order(tmp_path):
    pytest.importorskip("pxr")
    mujoco = pytest.importorskip("mujoco")

    root = Path(__file__).resolve().parents[1]
    source = root / "datasets/icl/sources/geniesim-assets"
    if not source.joinpath("roboverse_source_manifest.json").is_file():
        pytest.skip("Prepare the pinned GenieSimAssets source snapshot first")
    expected = {
        "benchmark_laptop_000": ["RevoluteJoint"],
        "benchmark_door_000": ["RevoluteJoint_00", "RevoluteJoint_05"],
    }
    output = tmp_path / "prepared"
    command = [
        sys.executable,
        str(root / "tools/icl/prepare_geniesim_articulations.py"),
        "--source",
        str(source),
        "--output",
        str(output),
    ]
    for asset_id in expected:
        command.extend(["--asset-id", asset_id])
    subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
    for asset_id, source_joint_order in expected.items():
        asset = output / asset_id
        provenance = json.loads((asset / "provenance.json").read_text(encoding="utf-8"))
        assert provenance["joint_order"] == source_joint_order
        assert provenance["asset_license"] == "CC-BY-NC-SA-4.0; retain source terms"
        model = mujoco.MjModel.from_xml_path(str((asset / "object.xml").resolve()))
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) for index in range(model.njnt)]
        assert names == source_joint_order


def test_geniesim_source_hdr_resolver_and_panorama_backdrop(tmp_path):
    import xml.etree.ElementTree as ET

    cv2 = pytest.importorskip("cv2")
    import numpy as np

    from roboverse_pack.tasks.icl.geniesim_scene import (
        _studio_backdrop_assets,
        resolve_geniesim_environment_hdr,
    )

    source_root = tmp_path / "source"
    background_dir = source_root / "background/laboratory/laboratory_3"
    hdr_dir = source_root / "background/common/laboratory_3/splat_7"
    background_dir.mkdir(parents=True)
    hdr_dir.mkdir(parents=True)
    hdr_path = hdr_dir / "studio.hdr"
    hdr_pixels = np.zeros((16, 32, 3), dtype=np.float32)
    hdr_pixels[:, :16] = (1.0, 0.4, 0.1)
    hdr_pixels[:, 16:] = (0.1, 0.4, 1.0)
    assert cv2.imwrite(str(hdr_path), hdr_pixels)
    scene_path = background_dir / "background.usda"
    scene_path.write_text(
        'def DomeLight "DomeLight" { asset inputs:texture:file = @../../common/laboratory_3/splat_7/studio.hdr@ }\n',
        encoding="utf-8",
    )

    resolved = resolve_geniesim_environment_hdr("background/laboratory/laboratory_3/background.usda", source_root)
    assert resolved == hdr_path.resolve()
    urdf_path, mjcf_path, resources = _studio_backdrop_assets(tmp_path / "cache", panorama_path=resolved)
    assert all(Path(resource).is_file() for resource in resources)
    urdf = ET.parse(urdf_path).getroot()
    assert Path(urdf.find(".//visual/geometry/mesh").get("filename")).is_file()
    assert Path(urdf.find(".//visual/material/texture").get("filename")).is_file()
    mjcf = ET.parse(mjcf_path).getroot()
    assert mjcf.find("asset/texture") is not None
    assert mjcf.find("worldbody/geom[@name='studio_panorama']") is not None
    obj_lines = (urdf_path.parent / "panorama.obj").read_text(encoding="utf-8").splitlines()
    assert sum(line.startswith("vt ") for line in obj_lines) > 1000
    assert sum(line.startswith("f ") for line in obj_lines) > 1000
    first_vertex = np.fromstring(next(line[2:] for line in obj_lines if line.startswith("v ")), sep=" ")
    assert first_vertex[1] < 0 and np.linalg.norm(first_vertex) == pytest.approx(12.0)


def test_geniesim_source_environment_meshes_keep_transforms_and_textures(tmp_path):
    import xml.etree.ElementTree as ET

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    from roboverse_pack.tasks.icl.geniesim_scene import _source_environment_geometry_assets

    source_root = tmp_path / "source"
    scene_path = source_root / "background/laboratory/test/background.usda"
    scene_path.parent.mkdir(parents=True)
    texture_path = scene_path.parent / "color.png"
    Image.new("RGB", (4, 4), (180, 120, 40)).save(texture_path)
    stage = Usd.Stage.CreateNew(str(scene_path))
    UsdGeom.Xform.Define(stage, "/World")
    mesh = UsdGeom.Mesh.Define(stage, "/World/background/wall")
    mesh.CreatePointsAttr([Gf.Vec3f(0, 0, 0), Gf.Vec3f(0, 0, 2), Gf.Vec3f(2, 0, 2), Gf.Vec3f(2, 0, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.AddTranslateOp().Set(Gf.Vec3d(0, 3, 0))
    uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    uv.Set([Gf.Vec2f(0, 0), Gf.Vec2f(0, 1), Gf.Vec2f(1, 1), Gf.Vec2f(1, 0)])
    material = UsdShade.Material.Define(stage, "/World/Looks/wall_material")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/wall_material/PBR")
    shader.CreateIdAttr("OmniPBR")
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 1, 1))
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture_path)))
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)
    collision = UsdGeom.Mesh.Define(stage, "/World/background/CollisionMesh")
    collision.CreatePointsAttr([Gf.Vec3f(0), Gf.Vec3f(1), Gf.Vec3f(2)])
    collision.CreateFaceVertexCountsAttr([3])
    collision.CreateFaceVertexIndicesAttr([0, 1, 2])
    hidden_parent = UsdGeom.Xform.Define(stage, "/World/background/hidden_parent")
    hidden_parent.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    hidden = UsdGeom.Mesh.Define(stage, "/World/background/hidden_parent/collider_body_00")
    hidden.CreatePointsAttr([Gf.Vec3f(0), Gf.Vec3f(1), Gf.Vec3f(2)])
    hidden.CreateFaceVertexCountsAttr([3])
    hidden.CreateFaceVertexIndicesAttr([0, 1, 2])
    stage.GetRootLayer().Save()

    result = _source_environment_geometry_assets(
        tmp_path / "cache",
        "background/laboratory/test/background.usda",
        source_root,
    )
    assert result is not None
    urdf_path, mjcf_path, resources = result
    assert len(resources) == 3 and all(Path(resource).is_file() for resource in resources)
    urdf = ET.parse(urdf_path).getroot()
    assert len(urdf.findall(".//link/visual")) == 1
    assert Path(urdf.find(".//visual/material/texture").get("filename")).is_file()
    obj_path = next(Path(resource) for resource in resources if Path(resource).suffix == ".obj")
    vertices = [
        np.fromstring(line[2:], sep=" ")
        for line in obj_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("v ")
    ]
    np.testing.assert_allclose([vertex[1] for vertex in vertices], 0.03)
    mjcf = ET.parse(mjcf_path).getroot()
    assert len(mjcf.findall("./worldbody/geom")) == 1
    assert mjcf.find("./asset/texture") is not None
    assert mjcf.find("./asset/material") is not None
    obj_text = obj_path.read_text(encoding="utf-8")
    assert "vt 0 0" in obj_text and "f 1/1 2/2 3/3" in obj_text
    assert "map_Kd" in Path(urdf_path).with_name("environment.mtl").read_text(encoding="utf-8")
    assert not any("CollisionMesh" in path for path in resources)


def test_geniesim_environment_textures_are_capped_for_offscreen_renderers(tmp_path):
    from PIL import Image

    from roboverse_pack.tasks.icl.geniesim_scene import (
        _SOURCE_ENVIRONMENT_MAX_TEXTURE_SIZE,
        _copy_environment_texture,
    )

    source = tmp_path / "source.png"
    destination = tmp_path / "texture.png"
    Image.new("RGB", (4096, 2048), (180, 120, 40)).save(source)
    _copy_environment_texture(source, destination)
    with Image.open(destination) as texture:
        assert max(texture.size) == _SOURCE_ENVIRONMENT_MAX_TEXTURE_SIZE == 1024


def test_geniesim_panorama_resolution_is_capped(tmp_path):
    import cv2
    from PIL import Image

    from roboverse_pack.tasks.icl.geniesim_scene import (
        _STUDIO_PANORAMA_MAX_WIDTH,
        _studio_panorama_assets,
    )

    panorama = tmp_path / "source.hdr"
    hdr = np.full((512, 2048, 3), 0.25, dtype=np.float32)
    assert cv2.imwrite(str(panorama), hdr)
    _, _, resources = _studio_panorama_assets(tmp_path / "cache", panorama)
    with Image.open(resources[-1]) as texture:
        assert texture.size == (_STUDIO_PANORAMA_MAX_WIDTH, 256)


def test_geniesim_static_environment_simplification_preserves_uv_mapping():
    pytest.importorskip("fast_simplification")
    from roboverse_pack.tasks.icl.geniesim_scene import (
        _SOURCE_ENVIRONMENT_MAX_FACES,
        _simplify_environment_mesh,
    )

    rows, columns = 101, 203
    xx, yy = np.meshgrid(np.arange(columns, dtype=np.float64), np.arange(rows, dtype=np.float64))
    vertices = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    triangles = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            lower_left = row * columns + column
            lower_right = lower_left + 1
            upper_left = lower_left + columns
            upper_right = upper_left + 1
            triangles.extend(((lower_left, lower_right, upper_right), (lower_left, upper_right, upper_left)))
    triangles = np.asarray(triangles, dtype=np.int64)
    uv_values = vertices[:, :2].copy()

    simplified_vertices, simplified_faces, simplified_uv, uv_faces, changed = _simplify_environment_mesh(
        vertices, triangles, uv_values, triangles
    )

    assert changed
    assert 0 < len(simplified_faces) <= _SOURCE_ENVIRONMENT_MAX_FACES
    assert simplified_uv.shape == (len(simplified_vertices), 2)
    np.testing.assert_array_equal(uv_faces, simplified_faces)
    assert np.isfinite(simplified_vertices).all() and np.isfinite(simplified_uv).all()


def test_geniesim_object_bounds_load_from_validated_asset_provenance(tmp_path):
    from roboverse_pack.tasks.icl.geniesim_scene import geniesim_object_bounds

    assets = tmp_path / "assets"
    _write_geniesim_rigid_fixture(assets)
    instance = {
        "scene_info": {
            "layout": {"fruit": {"usd": "benchmark_apple_001"}},
        }
    }
    bounds = geniesim_object_bounds(instance, assets)
    np.testing.assert_allclose(bounds["fruit"], [[0, 0, 0], [1, 1, 1]])


def test_geniesim_run_adapter_selects_instance_and_source_spawn_pose(tmp_path):
    from metasim.scenario.robot import RobotCfg
    from roboverse_pack.tasks.icl.geniesim_scene import build_run_scenario

    assets = tmp_path / "assets"
    _write_geniesim_rigid_fixture(assets)
    robot_file = tmp_path / "dummy.urdf"
    robot_file.write_text('<robot name="dummy"/>', encoding="utf-8")
    instance = {
        "task": "pick_item",
        "instance_id": "2",
        "scene_info": {
            "layout": {
                "apple": {"usd": "benchmark_apple_001", "xyz": [0.1, 0.2, 0.3], "xyzw": [0.0, 0.0, 0.0, 1.0]},
                "table": {
                    "usd": "benchmark_apple_001",
                    "xyz": [0.0, 0.0, 0.4],
                    "xyzw": [0.0, 0.0, 0.0, 1.0],
                    "description": {"object_category": ["furniture", "table"]},
                },
            }
        },
    }
    spec = {
        "format": "roboverse.icl.geniesim_benchmark_specs.v1",
        "runs": [
            {
                "id": "g1op_s2r_pick_item",
                "subtask": "pick_item",
                "initial_pose_robot": "G1_omnipicker",
                "instances": ["pick_item/2"],
                "evaluation_config": "table",
            }
        ],
        "initial_pose_source": {
            "joint_positions": {
                "pick_item": {"G1_omnipicker": {"joint_a": 0.4}, "G2_omnipicker": {"joint_a": 0.7}},
                "pick_block_color": {"G2_omnipicker": {"joint_a": 0.3}},
            }
        },
        "instances": {"pick_item/2": instance},
        "evaluation_configs": {
            "table": {
                "config": {
                    "robot": {
                        "robot_id": "G1",
                        "robot_cfg": "G1_omnipicker.json",
                        "robot_init_pose": {
                            "workspace_00": {"position": [-0.7, 0.0, -0.01], "quaternion": [0.0, 0.0, 0.0, 2.0]}
                        },
                    },
                    "scene": {"function_space_objects": {"workspace_00": {"position": [-4.03, -0.032, 0.80621]}}},
                    "objects": {"fix_objects": ["apple"]},
                }
            }
        },
        "robot_configs": {"G1_omnipicker.json": {"config": {"robot": {"init_joint_position": [0.4, None]}}}},
    }
    spec_path = tmp_path / "benchmark.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    robot = RobotCfg(
        name="agibot_g1",
        urdf_path=str(robot_file),
        mjcf_path=str(robot_file),
        num_joints=2,
        default_joint_positions={"joint_a": 0.0, "joint_b": 0.0},
        joint_limits={"joint_a": (-1.0, 1.0), "joint_b": (-1.0, 1.0)},
    )

    scenario, run, selected, evaluation = build_run_scenario(
        spec_path=spec_path,
        run_id="g1op_s2r_pick_item",
        simulator="mujoco",
        robot=robot,
        assets_root=assets,
        cache_root=tmp_path / "cache",
    )
    assert run["id"] == "g1op_s2r_pick_item" and selected["instance_id"] == "2"
    assert evaluation["objects"]["fix_objects"] == ["apple"]
    assert scenario.sim_params.dt == pytest.approx(1 / 600)
    assert scenario.sim_params.dt * scenario.decimation == pytest.approx(1 / 30)
    np.testing.assert_allclose(scenario.robots[0].default_position, (-4.73, -0.032, -0.01))
    assert scenario.robots[0].default_orientation == (0.0, 0.0, 0.0, 1.0)
    assert scenario.robots[0].default_joint_positions == {"joint_a": 0.4, "joint_b": 0.0}
    np.testing.assert_allclose(scenario.objects[2].default_position, (-3.93, 0.168, 0.3))
    np.testing.assert_allclose(scenario.objects[3].default_position, (-4.03, -0.032, 0.4))
    assert scenario.objects[1].fix_base_link is True
    assert scenario.objects[2].fix_base_link is True

    from copy import deepcopy

    g2_robot = deepcopy(robot)
    g2_robot.name = "agibot_g2"
    with pytest.raises(ValueError, match="requires robot"):
        build_run_scenario(
            spec_path=spec_path,
            run_id="g1op_s2r_pick_item",
            simulator="mujoco",
            robot=g2_robot,
            assets_root=assets,
            cache_root=tmp_path / "cache",
        )
    adapted, adapted_run, _, _ = build_run_scenario(
        spec_path=spec_path,
        run_id="g1op_s2r_pick_item",
        simulator="mujoco",
        robot=g2_robot,
        assets_root=assets,
        cache_root=tmp_path / "cache",
        allow_robot_substitution=True,
    )
    assert adapted.robots[0].default_joint_positions["joint_a"] == pytest.approx(0.7)
    assert adapted_run["robot_adaptation"]["runtime_robot"] == "G2"
    assert adapted_run["robot_adaptation"]["joint_pose_source_task"] == "pick_item"
    assert adapted_run["robot_adaptation"]["cross_robot_pose_and_reach_validated"] is False
    del spec["initial_pose_source"]["joint_positions"]["pick_item"]["G2_omnipicker"]
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    fallback, fallback_run, _, _ = build_run_scenario(
        spec_path=spec_path,
        run_id="g1op_s2r_pick_item",
        simulator="mujoco",
        robot=g2_robot,
        assets_root=assets,
        cache_root=tmp_path / "cache",
        allow_robot_substitution=True,
    )
    assert fallback.robots[0].default_joint_positions["joint_a"] == pytest.approx(0.3)
    assert fallback_run["robot_adaptation"]["joint_pose_source_task"] == "pick_block_color"


def test_geniesim_robot_spawn_pose_accepts_direct_and_workspace_source_forms():
    from roboverse_pack.tasks.icl.geniesim_scene import _robot_spawn_pose

    direct = {"position": [0.1, -0.2, 0.3], "quaternion": [1.0, 0.0, 0.0, 0.0]}
    workspace = {"workspace_00": direct}
    assert _robot_spawn_pose(direct, run_id="direct") == direct
    assert _robot_spawn_pose(workspace, run_id="workspace") == direct
    assert _robot_spawn_pose({}, run_id="empty") is None


def test_geniesim_source_reset_uses_names_and_rejects_invalid_arm_limit():
    from types import SimpleNamespace

    from roboverse_pack.tasks.icl.control import prepare_geniesim_reset_pose

    robot = SimpleNamespace(
        default_joint_positions={"joint_b": 0.2, "joint_a": 0.0},
        joint_limits={"joint_a": (-1.0, 1.0), "joint_b": (-1.0, 1.0)},
    )
    reset = prepare_geniesim_reset_pose(robot, {"joint_a": 0.4})
    assert reset.joint_positions == {"joint_a": 0.4, "joint_b": 0.2}
    with pytest.raises(ValueError, match="outside"):
        prepare_geniesim_reset_pose(robot, {"joint_a": 2.0})
    with pytest.raises(ValueError, match="missing"):
        prepare_geniesim_reset_pose(robot, {"unknown": 0.0})


def test_geniesim_g1_source_reset_respects_gripper_loop_constraints():
    from roboverse_pack.tasks.icl.control import prepare_geniesim_reset_pose
    from roboverse_pack.tasks.icl.geniesim_benchmark import ROBOT_ROOT, SPEC_PATH, build_geniesim_g1_robot
    from roboverse_pack.tasks.icl.geniesim_scene import load_run_instance

    if not SPEC_PATH.is_file() or not ROBOT_ROOT.joinpath("provenance.json").is_file():
        pytest.skip("Prepare the source-scene snapshot and G1 robot first")
    run, _, _ = load_run_instance(SPEC_PATH, "g1op_s2r_bimanual_chip_handover")
    reset = prepare_geniesim_reset_pose(build_geniesim_g1_robot(ROBOT_ROOT), run["source_initial_positions"])
    assert reset.max_equality_residual < 1e-4
    assert set(reset.clipped_source_positions) == {"idx41_gripper_l_outer_joint1", "idx81_gripper_r_outer_joint1"}
    assert all(item["source"] == 1.0 and item["applied"] < 1.0 for item in reset.clipped_source_positions.values())


@pytest.mark.parametrize("simulator", ("mujoco", "newton"))
def test_geniesim_newton_reset_applies_tree_joints_sharing_loop_children(simulator):
    """Loop equalities must not hide the child body's ordinary hinge joint."""
    if importlib.util.find_spec(simulator) is None:
        pytest.skip(f"{simulator} is not installed")
    if simulator == "newton":
        import torch

        if not torch.cuda.is_available():
            pytest.skip("Newton integration test requires a CUDA GPU")
    from roboverse_pack.tasks.icl.geniesim_benchmark import ROBOT_ROOT, SPEC_PATH, GenieSimBenchmarkTask

    if not SPEC_PATH.is_file() or not (ROBOT_ROOT / "provenance.json").is_file():
        pytest.skip("Prepare the source snapshot and G1 robot model")
    task = GenieSimBenchmarkTask(
        run_id="g1op_s2r_bimanual_chip_handover",
        simulator=simulator,
        image_size=64,
        headless=True,
        cameras=(),
        max_episode_steps=3,
    )
    try:
        observation, _ = task.reset()
        robot_cfg = task.scenario.robots[0]
        joint_names = [name.split("/")[-1] for name in task.handler.get_joint_names(task.robot_name, sort=True)]
        joint_pos = observation.robots[task.robot_name].joint_pos[0]
        loop_child_tree_joints = (
            "idx33_gripper_l_inner_joint4",
            "idx43_gripper_l_outer_joint4",
            "idx73_gripper_r_inner_joint4",
            "idx83_gripper_r_outer_joint4",
        )
        for name in loop_child_tree_joints:
            assert name in joint_names
            assert float(joint_pos[joint_names.index(name)]) == pytest.approx(
                robot_cfg.default_joint_positions[name], abs=1e-5
            )

        held_targets = joint_pos[task._control_indices].detach().cpu().numpy().copy()
        after, *_ = task.step(held_targets)
        after_pos = after.robots[task.robot_name].joint_pos[0]
        for name in ("idx41_gripper_l_outer_joint1", "idx81_gripper_r_outer_joint1"):
            assert abs(float(after_pos[joint_names.index(name)] - joint_pos[joint_names.index(name)])) < 1e-3
    finally:
        task.close()


def test_geniesim_control_parity_uses_matching_targets_and_simulation_timestamps():
    from copy import deepcopy

    from tools.parity.parity_geniesim_control import _orientation_difference, compare_trajectories, position_targets

    names = ("idx23_arm_l_joint3", "idx63_arm_r_joint3", "other")
    initial = np.array([0.1, 0.2, 0.3])
    np.testing.assert_allclose(position_targets(initial, names, 0, 4), initial)
    np.testing.assert_allclose(position_targets(initial, names, 2, 4), [0.18, 0.12, 0.3])
    np.testing.assert_allclose(position_targets(initial, names, 4, 4), initial)
    sample = {
        "step": 0,
        "time_s": 0.0,
        "target": initial.tolist(),
        "joint_pos": initial.tolist(),
        "gripper_link_poses": {"left": [0, 0, 0, 1, 0, 0, 0]},
        "object_poses": {"box": [0, 0, 0, 1, 0, 0, 0]},
    }
    reference = {"run_id": "task", "simulator": "mujoco", "control_joint_names": list(names), "samples": [sample]}
    actual = deepcopy(reference)
    actual["simulator"] = "newton"
    actual["samples"][0]["gripper_link_poses"]["left"][0] = 0.01
    assert compare_trajectories(reference, actual)["max_gripper_position_difference_m"] == pytest.approx(0.01)
    assert _orientation_difference([1, 0, 0, 0], [-1, 0, 0, 0]) == pytest.approx(0.0)
    assert _orientation_difference([1, 0, 0, 0], [0, 1, 0, 0]) == pytest.approx(np.pi)
    actual["samples"][0]["time_s"] = 0.01
    with pytest.raises(ValueError, match="timestamps"):
        compare_trajectories(reference, actual)
    actual["samples"][0]["time_s"] = 0.0
    reference["source_snapshot_sha256"] = "source-a"
    actual["source_snapshot_sha256"] = "source-b"
    with pytest.raises(ValueError, match="source_snapshot_sha256"):
        compare_trajectories(reference, actual)


def test_geniesim_camera_selection_uses_configured_presets():
    from metasim.scenario.cameras import PinholeCameraCfg
    from roboverse_pack.tasks.icl.geniesim_benchmark import _select_cameras

    cameras = [PinholeCameraCfg(name=name) for name in ("head_rgb", "head_fisheye_left", "head_fisheye_right")]
    assert [camera.name for camera in _select_cameras(cameras, None)] == [camera.name for camera in cameras]
    assert [camera.name for camera in _select_cameras(cameras, ("head_fisheye_right", "head_rgb"))] == [
        "head_rgb",
        "head_fisheye_right",
    ]
    for names in ((), ("head_rgb", "head_rgb"), ("missing",)):
        with pytest.raises(ValueError):
            _select_cameras(cameras, names)


@pytest.mark.examples
def test_geniesim_genesis_floating_source_object_falls():
    """A dynamic MJCF object must not stay welded at its imported spawn."""
    if "genesis" not in set(os.environ.get("ROBOVERSE_ICL_TEST_BACKENDS", "").split(",")):
        pytest.skip("Set ROBOVERSE_ICL_TEST_BACKENDS to include genesis")
    from roboverse_pack.tasks.icl.geniesim_benchmark import ROBOT_ROOT, SPEC_PATH, GenieSimBenchmarkTask

    if not SPEC_PATH.is_file() or not (ROBOT_ROOT / "provenance.json").is_file():
        pytest.skip("Prepare the source-scene snapshot and G1 robot first")
    task = GenieSimBenchmarkTask(
        run_id="g1op_s2r_bimanual_chip_handover",
        simulator="genesis",
        image_size=64,
        max_episode_steps=10,
    )
    try:
        observation, _ = task.reset()
        name = "chip_a1b2c3d4"
        initial_z = float(observation.objects[name].root_state[0, 2])
        command = [task.scenario.robots[0].default_joint_positions[n] for n in task.control_joint_names]
        for _ in range(5):
            observation, *_ = task.step(command)
        assert float(observation.objects[name].root_state[0, 2]) < initial_z - 0.005
    finally:
        task.close()


@pytest.mark.examples
@pytest.mark.parametrize("simulator", ["mujoco", "genesis", "newton"])
@pytest.mark.parametrize(
    ("run_id", "robot_variant"),
    [("g1op_s2r_select_color", "G2"), ("g2op_if_pick_billiards_color", "G1")],
)
def test_geniesim_cross_robot_scene_loads_with_three_head_cameras(simulator, run_id, robot_variant):
    """Both model choices load the same source scene without claiming task success."""
    requested = set(os.environ.get("ROBOVERSE_ICL_TEST_BACKENDS", "").split(","))
    if simulator != "mujoco" and simulator not in requested:
        pytest.skip(f"Set ROBOVERSE_ICL_TEST_BACKENDS to include {simulator}")
    if importlib.util.find_spec(simulator) is None:
        pytest.skip(f"{simulator} is not installed")
    from roboverse_pack.tasks.icl.geniesim_benchmark import G2_ROBOT_ROOT, ROBOT_ROOT, SPEC_PATH, GenieSimBenchmarkTask

    if (
        not SPEC_PATH.is_file()
        or not (ROBOT_ROOT / "provenance.json").is_file()
        or not (G2_ROBOT_ROOT / "provenance.json").is_file()
    ):
        pytest.skip("Prepare the source snapshot and both G1/G2 robot models")
    task = GenieSimBenchmarkTask(
        run_id=run_id,
        simulator=simulator,
        robot_variant=robot_variant,
        image_size=64,
        max_episode_steps=3,
    )
    try:
        observation, _ = task.reset()
        assert task.robot_type == robot_variant and task.source_robot != robot_variant
        assert set(observation.cameras) == {"head_rgb", "head_fisheye_left", "head_fisheye_right"}
        assert all(
            np.isfinite(c.rgb.cpu().numpy()).all() and float(c.rgb.float().std()) > 1
            for c in observation.cameras.values()
        )
        targets = [task.scenario.robots[0].default_joint_positions[n] for n in task.control_joint_names]
        _, _, _, _, info = task.step(targets)
        assert info["geniesim_runtime_robot"] == robot_variant
        assert info["geniesim_robot_adaptation"]["substituted"] is True
        assert info["geniesim_robot_adaptation"]["cross_robot_pose_and_reach_validated"] is False
    finally:
        task.close()


def test_geniesim_scene_probe_cli_documents_supported_backends():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "tools/icl/probe_geniesim_scene.py", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--run-id" in result.stdout
    assert "{mujoco,genesis,newton}" in result.stdout
    assert "--instance-index" in result.stdout


def test_geniesim_action_replay_validates_identity_cadence_and_finite_values(tmp_path):
    from tools.icl.record_geniesim_benchmark_video import load_action_sequence

    path = tmp_path / "actions.json"
    payload = {
        "format": "roboverse.icl.benchmark_action_sequence.v1",
        "run_id": "g1op_s2r_select_color",
        "instance_index": 0,
        "problem_index": 1,
        "action_mode": "tcp_delta",
        "benchmark_source_content_sha256": "a" * 64,
        "step_seconds": 1 / 30,
        "source": {"type": "manual_test"},
        "actions": [[0.0] * 7],
    }

    def load():
        path.write_text(json.dumps(payload), encoding="utf-8")
        return load_action_sequence(path, run_id="g1op_s2r_select_color", instance_index=0, problem_index=1)

    assert load()["actions"] == [[0.0] * 7]
    payload["instance_index"] = 1
    with pytest.raises(ValueError, match="does not match"):
        load()
    payload["instance_index"] = 0
    payload["step_seconds"] = 0.0
    with pytest.raises(ValueError, match="step_seconds"):
        load()
    payload["step_seconds"] = 1 / 30
    payload["actions"] = [[float("nan")]]
    with pytest.raises(ValueError, match="finite numbers"):
        load()
    payload["actions"] = [[0.0] * 7]
    payload["tcp_arm"] = "both"
    with pytest.raises(ValueError, match="tcp_arm"):
        load()
    payload["tcp_arm"] = "left"
    assert load()["tcp_arm"] == "left"
    payload["action_mode"] = "tcp_waypoint"
    payload["tcp_arm"] = "right"
    assert load()["action_mode"] == "tcp_waypoint"
    assert load()["tcp_arm"] == "right"


def test_geniesim_tcp_waypoint_policy_limits_feedback_steps():
    from tools.icl.record_geniesim_benchmark_video import _tcp_waypoint_joint_targets

    class Controller:
        def tcp_pose(self, controls):
            return np.zeros(3), Rotation.identity()

        def joint_targets(self, controls, action):
            return np.asarray(action)

    target = [0.02, 0.0, 0.0, 0.0, 0.0, np.pi, 0.4]
    action = _tcp_waypoint_joint_targets(Controller(), np.zeros(20), target)
    np.testing.assert_allclose(action, [0.008, 0.0, 0.0, 0.0, 0.0, 0.1, 0.4], atol=1e-8, rtol=0.0)


def test_geniesim_tcp_waypoint_bounds_each_command_to_latched_subtarget():
    from tools.icl.record_geniesim_benchmark_video import _tcp_waypoint_joint_targets

    class Controller:
        def tcp_pose(self, controls):
            return np.zeros(3), Rotation.identity()

        def joint_targets(self, controls, action):
            return np.asarray(action)

    target = [0.2, 0.0, 0.0, 0.0, 0.0, 1.0, 0.4]
    latched_target = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 1.0])
    action = _tcp_waypoint_joint_targets(
        Controller(),
        np.zeros(20),
        target,
        max_translation_step=0.02,
        max_rotation_step=0.2,
        target_pose=latched_target,
    )
    np.testing.assert_allclose(action, [0.02, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4], atol=1e-8, rtol=0.0)


def test_geniesim_tcp_waypoint_subtargets_are_bounded_and_latched():
    from tools.icl.record_geniesim_benchmark_video import _tcp_waypoint_subtarget_pose

    class Controller:
        position = np.zeros(3)
        rotation = Rotation.identity()

        def tcp_pose(self, controls):
            return self.position.copy(), self.rotation

    controller = Controller()
    target = [0.02, 0.0, 0.0, 0.0, 0.0, np.pi, 0.4]
    first, first_is_final = _tcp_waypoint_subtarget_pose(controller, np.zeros(20), target)
    np.testing.assert_allclose(first, [0.008, 0.0, 0.0, 0.0, 0.0, 0.1], atol=1e-8, rtol=0.0)
    assert not first_is_final

    controller.position = first[:3].copy()
    controller.rotation = Rotation.from_rotvec(first[3:])
    second, second_is_final = _tcp_waypoint_subtarget_pose(controller, np.zeros(20), target)
    np.testing.assert_allclose(second[:3], [0.016, 0.0, 0.0], atol=1e-8, rtol=0.0)
    assert not second_is_final


def test_geniesim_recording_identifies_selected_robot_files(tmp_path):
    import hashlib

    from tools.icl.record_geniesim_benchmark_video import _robot_asset_info

    for name in ("provenance.json", "G2_crs_omnipicker.urdf", "G2_crs_omnipicker.xml"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    info = _robot_asset_info("g2_op", tmp_path)
    assert info["root"] == str(tmp_path.resolve())
    assert info["files_sha256"]["mjcf"] == hashlib.sha256(b"G2_crs_omnipicker.xml").hexdigest()
    (tmp_path / "G2_crs_omnipicker.xml").unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        _robot_asset_info("g2_op", tmp_path)


def test_unique_scene_matrix_plan_selects_one_source_run_and_segments_deterministically():
    from tools.icl.record_geniesim_benchmark_video import build_unique_scene_plan

    pack = {
        "runs": [
            {"id": "z_run", "instances": ["b/0", "a/1"]},
            {"id": "a_run", "instances": ["a/1", "c/2"]},
        ],
        "instances": {"a/1": {}, "b/0": {}, "c/2": {}},
    }
    plan = build_unique_scene_plan(
        pack, simulators=("newton", "mujoco"), robot_variants=("G2", "G1"), scene_start=1, scene_count=1
    )
    assert plan["referenced_scene_count"] == 3
    assert plan["scene_count"] == 1
    assert len(plan["jobs"]) == 4
    assert {(job["robot_variant"], job["simulator"]) for job in plan["jobs"]} == {
        ("G1", "mujoco"),
        ("G1", "newton"),
        ("G2", "mujoco"),
        ("G2", "newton"),
    }
    assert all(
        job["scene"] == "b/0" and job["run_id"] == "z_run" and job["instance_index"] == 0 for job in plan["jobs"]
    )
    first = build_unique_scene_plan(pack, simulators=("mujoco",), robot_variants=("G1",), scene_count=1)
    assert first["jobs"][0]["run_id"] == "a_run" and first["jobs"][0]["instance_index"] == 0
    with pytest.raises(ValueError, match="duplicates"):
        build_unique_scene_plan(pack, simulators=("mujoco", "mujoco"), robot_variants=("G1",))
    with pytest.raises(ValueError, match="outside"):
        build_unique_scene_plan(pack, simulators=("mujoco",), robot_variants=("G1",), scene_start=3)


def test_unique_scene_matrix_plan_covers_every_referenced_snapshot_scene():
    from tools.icl.record_geniesim_benchmark_video import SPEC_PATH, build_unique_scene_plan

    if not SPEC_PATH.is_file():
        pytest.skip("Prepare the GenieSim benchmark source snapshot first")
    pack = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    plan = build_unique_scene_plan(pack, simulators=("mujoco", "genesis", "newton"), robot_variants=("G1", "G2"))
    referenced = {key for run in pack["runs"] for key in run["instances"]}
    assert {job["scene"] for job in plan["jobs"]} == referenced
    assert len(plan["jobs"]) == 6 * len(referenced)


def test_camera_override_preserves_mount_and_checks_domain():
    from metasim.scenario.cameras import FisheyeCameraCfg, PinholeCameraCfg
    from roboverse_pack.tasks.icl.geniesim_benchmark import _apply_camera_calibrations

    source = PinholeCameraCfg(
        name="left",
        width=256,
        height=200,
        data_types=["rgb"],
        mount_to="g1",
        mount_link="head",
        mount_pos=(0.1, 0.2, 0.3),
        mount_quat=(1, 0, 0, 0),
    )
    calibration = {
        "projection_model": "opencv_fisheye",
        "calibration_size": [1000, 800],
        "optical_center": [500, 400],
        "focal_length_px": [300, 300],
        "coefficients": [0, 0, 0, 0],
        "max_fov_rad": 3.0,
        "calibration_source": "measured",
    }
    (result,) = _apply_camera_calibrations([source], {"left": calibration})
    assert isinstance(result, FisheyeCameraCfg)
    assert (result.width, result.height) == (256, 205)
    assert result.mount_to == source.mount_to and result.mount_quat == source.mount_quat
    assert result.calibration_source == "measured"
    with pytest.raises(ValueError, match="unselected"):
        _apply_camera_calibrations([source], {"other": calibration})
    with pytest.raises(ValueError, match="invertible"):
        _apply_camera_calibrations([source], {"left": {**calibration, "coefficients": [-1, 0, 0, 0]}})


def test_geniesim_recording_camera_selection_validation():
    from tools.icl.record_geniesim_benchmark_video import _selected_camera_names

    assert _selected_camera_names("hardware", None) == ("head_rgb", "head_fisheye_left", "head_fisheye_right")
    assert _selected_camera_names("hardware", ("head_fisheye_right",)) == ("head_fisheye_right",)
    assert _selected_camera_names("benchmark", ("hand_left",)) == ("hand_left",)
    for names in ((), ("head_rgb", "head_rgb"), ("missing",), ("hand_left",), ("",)):
        with pytest.raises(ValueError):
            _selected_camera_names("hardware", names)


def test_geniesim_recording_camera_config_identity_and_rejection(tmp_path):
    from tools.icl.record_geniesim_benchmark_video import _camera_config_info

    path = tmp_path / "calibration.json"
    valid = {
        "projection_model": "opencv_fisheye",
        "calibration_size": [640, 480],
        "optical_center": [320, 240],
        "focal_length_px": [250, 250],
        "coefficients": [0, 0, 0, 0],
        "max_fov_rad": 3.0,
    }
    path.write_text(
        json.dumps({"format": "roboverse.icl.geniesim_camera_calibration.v1", "cameras": {"head_rgb": valid}}),
        encoding="utf-8",
    )
    info = _camera_config_info(path, ("head_rgb",))
    assert info["content"]["cameras"]["head_rgb"] == valid
    assert len(info["sha256"]) == 64
    with pytest.raises(ValueError, match="unselected"):
        _camera_config_info(path, ("head_fisheye_left",))
    path.write_text(
        '{"format":"roboverse.icl.geniesim_camera_calibration.v1","cameras":{"head_rgb":{"fx":1e999}}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Non-finite"):
        _camera_config_info(path, ("head_rgb",))
    path.write_text('{"format":"x","format":"y","cameras":{}}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        _camera_config_info(path, ("head_rgb",))
    for bad in (
        {**valid, "unknown": 1},
        {**valid, "coefficients": [0, 0]},
        {**valid, "focal_length_px": [-1, 250]},
        {**valid, "max_fov_rad": 7},
        {**valid, "coefficients": [-1, 0, 0, 0]},
    ):
        path.write_text(
            json.dumps({"format": "roboverse.icl.geniesim_camera_calibration.v1", "cameras": {"head_rgb": bad}}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            _camera_config_info(path, ("head_rgb",))
    ftheta = {
        "projection_model": "ftheta",
        "calibration_size": [640, 480],
        "optical_center": [320, 240],
        "coefficients": [0, 250, 0, 0, 0],
        "max_fov_rad": 3.0,
        "calibration_source": "example",
    }
    path.write_text(
        json.dumps({"format": "roboverse.icl.geniesim_camera_calibration.v1", "cameras": {"head_rgb": ftheta}}),
        encoding="utf-8",
    )
    assert _camera_config_info(path, ("head_rgb",))["content"]["cameras"]["head_rgb"] == ftheta


def test_geniesim_camera_protocol_records_fisheye_calibration_and_valid_pixels():
    from types import SimpleNamespace

    from metasim.scenario.cameras import FisheyeCameraCfg
    from tools.icl.record_geniesim_benchmark_video import _camera_protocol

    camera = FisheyeCameraCfg(
        name="head_fisheye_left",
        width=5,
        height=5,
        calibration_size=(5, 5),
        optical_center=(2.5, 2.5),
        focal_length_px=(1, 1),
        coefficients=(0, 0, 0, 0),
        max_fov_rad=4.4,
        calibration_source="borrowed_g2_usd",
        mount_link="head_link2",
    )
    entry = _camera_protocol(SimpleNamespace(scenario=SimpleNamespace(cameras=[camera])))[0]
    assert entry["projection_model"] == "opencv_fisheye"
    assert entry["calibration_source"] == "borrowed_g2_usd"
    assert entry["mount_link"] == "head_link2"
    validity = entry["geometric_pixel_validity"]
    assert validity["total_pixels"] == 25
    assert 0 < validity["valid_pixels"] < 25
    assert validity["valid_fraction"] == pytest.approx(validity["valid_pixels"] / 25)


def test_unique_scene_matrix_resumes_failed_jobs_without_overwriting(tmp_path, monkeypatch):
    import hashlib

    from tools.icl import record_geniesim_benchmark_video as recorder

    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps({
            "source": {"content_sha256": "a" * 64},
            "runs": [{"id": "run_a", "instances": ["task/0"]}],
            "instances": {"task/0": {}},
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(recorder, "SPEC_PATH", spec)
    monkeypatch.setattr(recorder, "source_content_sha256", lambda pack: "a" * 64)
    assets = {
        "g1_op": {"root": "g1", "model_version": "v1", "files_sha256": {"mjcf": "g1_hash"}},
        "g2_op": {"root": "g2", "model_version": "v2", "files_sha256": {"mjcf": "g2_hash"}},
    }
    monkeypatch.setattr(recorder, "_robot_asset_info", lambda robot, root: assets[robot])
    monkeypatch.setattr(recorder, "_runtime_file_hashes", lambda sim: {sim: "runtime_hash"})
    decode = {"frames": 5, "width": 64, "height": 64, "error": None}
    monkeypatch.setattr(recorder, "_decode_video", lambda path: decode)
    calls = []

    class Probe:
        def __init__(self, command, cwd, stdout, stderr):
            self.command = command
            self.variant = command[command.index("--robot-variant") + 1]
            self.sim = command[command.index("--sim") + 1]
            self.output = Path(command[command.index("--output") + 1])
            calls.append((self.variant, self.sim))
            self.failed = self.variant == "G1" and self.sim == "newton" and calls.count(("G1", "newton")) == 1
            stdout.write("retained probe output\n")

        def wait(self):
            if self.failed:
                return 7
            self.output.mkdir(parents=True)
            names = self.command[self.command.index("--camera-names") + 1].split(",")
            files = [f"{self.sim}_{name}.mp4" for name in names]
            for name in files:
                (self.output / name).write_bytes(name.encode())
            config_info = recorder._camera_config_info(
                Path(self.command[self.command.index("--camera-config") + 1])
                if "--camera-config" in self.command
                else None,
                tuple(names),
            )
            calibrations = config_info["content"]["cameras"] if config_info else {}
            report = {
                "format": "roboverse.icl.benchmark_video_probe.v1",
                "run_id": "run_a",
                "instance_index": 0,
                "problem_index": 0,
                "runtime_robot": self.variant,
                "simulator": self.sim,
                "source_content_sha256": "a" * 64,
                "source_snapshot_file_sha256": hashlib.sha256(spec.read_bytes()).hexdigest(),
                "robot_asset": assets["g1_op" if self.variant == "G1" else "g2_op"],
                "runtime_file_sha256": {self.sim: "runtime_hash"},
                "camera_profile": "hardware",
                "camera_names": names,
                "camera_config": config_info,
                "motion_mode": "head_pan",
                "capture": {"fps": 30, "frames": 5, "camera_image_size_argument": 64},
                "runtime_files_changed_during_recording": False,
                "source_snapshot_changed_during_recording": False,
                "robot_asset_changed_during_recording": False,
                "camera_config_changed_during_recording": False,
                "camera_protocol": [
                    {
                        "name": name,
                        **({"projection": "FisheyeCameraCfg", **calibrations[name]} if name in calibrations else {}),
                    }
                    for name in names
                ],
                "physics_protocol": {"environment_step_seconds": 1 / 30},
                "head_pan": {"finite": True, "max_deviation_from_start": 0.01},
                "video_files": files,
                "video_decode": {name: decode for name in files},
                "video_sha256": {name: hashlib.sha256(name.encode()).hexdigest() for name in files},
            }
            report["report_payload_sha256"] = hashlib.sha256(
                json.dumps(report, ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest()
            (self.output / "scene_video_report.json").write_text(json.dumps(report), encoding="utf-8")
            return 0

    monkeypatch.setattr(recorder.subprocess, "Popen", Probe)
    config = tmp_path / "camera.json"
    config.write_text(
        json.dumps({
            "format": "roboverse.icl.geniesim_camera_calibration.v1",
            "cameras": {
                "head_fisheye_left": {
                    "projection_model": "opencv_fisheye",
                    "calibration_size": [640, 480],
                    "optical_center": [320, 240],
                    "focal_length_px": [250, 250],
                    "coefficients": [0, 0, 0, 0],
                    "max_fov_rad": 3.0,
                }
            },
        }),
        encoding="utf-8",
    )
    kwargs = dict(
        output_root=tmp_path / "matrix",
        simulators=("mujoco", "newton"),
        robot_variants=("G1", "G2"),
        scene_count=1,
        image_size=64,
        frames=5,
        camera_names=("head_rgb", "head_fisheye_left"),
        camera_config=config,
    )
    first = recorder.record_unique_scene_matrix(**kwargs)
    assert [job["status"] for job in first["jobs"].values()].count("failed") == 1
    assert len(calls) == 4  # later jobs still ran after one failure
    prior_log = tmp_path / "matrix/logs/task/0/G1/newton/attempt_0001.stdout.log"
    assert prior_log.read_text(encoding="utf-8") == "retained probe output\n"
    resumed = recorder.record_unique_scene_matrix(**kwargs, resume=True)
    assert all(job["status"] == "recorded" for job in resumed["jobs"].values())
    assert len(calls) == 5  # only the failed job was retried
    assert prior_log.read_text(encoding="utf-8") == "retained probe output\n"
    assert len(resumed["jobs"]["task/0/G1/newton"]["attempts"]) == 2
    plan = json.loads((tmp_path / "matrix/matrix_plan.json").read_text(encoding="utf-8"))
    job = next(item for item in plan["jobs"] if item["output_relative"] == "task/0/G1/mujoco")
    recorded = tmp_path / "matrix/task/0/G1/mujoco/attempt_0001"
    report = json.loads((recorded / "scene_video_report.json").read_text(encoding="utf-8"))
    report["camera_protocol"][1]["focal_length_px"] = [100, 100]
    report.pop("report_payload_sha256")
    report["report_payload_sha256"] = hashlib.sha256(
        json.dumps(report, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    assert "calibration override was not applied" in recorder._matrix_report_error(report, recorded, job, plan)
    with pytest.raises(ValueError, match="unselected cameras"):
        recorder.record_unique_scene_matrix(**{**kwargs, "camera_names": ("head_rgb",)}, resume=True)
    config.write_text(config.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(FileExistsError, match="exact same source, models and settings"):
        recorder.record_unique_scene_matrix(**kwargs, resume=True)
    config.write_text(config.read_text(encoding="utf-8").rstrip(), encoding="utf-8")
    assets["g1_op"]["files_sha256"]["mjcf"] = "changed"
    with pytest.raises(FileExistsError, match="exact same source, models and settings"):
        recorder.record_unique_scene_matrix(**kwargs, resume=True)


def test_geniesim_scene_recording_reuses_only_matching_robot_and_source(tmp_path, monkeypatch):
    import hashlib

    import roboverse_pack.tasks.icl.geniesim_scene as scene
    from tools.icl import record_geniesim_benchmark_video as recorder

    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"source": {"content_sha256": "a" * 64}}), encoding="utf-8")
    monkeypatch.setattr(recorder, "SPEC_PATH", spec)
    monkeypatch.setattr(recorder, "source_content_sha256", lambda pack: "a" * 64)
    monkeypatch.setattr(scene, "load_run_instance", lambda *args: ({"robot": "g1_op"}, {}, {}))
    asset = {"root": str(tmp_path), "files_sha256": {"mjcf": "model_a"}}
    monkeypatch.setattr(recorder, "_robot_asset_info", lambda *args: asset)
    output = tmp_path / "output"
    output.mkdir()
    (output / "mujoco_head_rgb.mp4").write_bytes(b"video")
    existing = {
        "format": "roboverse.icl.benchmark_video_probe.v1",
        "simulator": "mujoco",
        "run_id": "run",
        "source_content_sha256": "a" * 64,
        "source_snapshot_file_sha256": hashlib.sha256(spec.read_bytes()).hexdigest(),
        "source_snapshot_changed_during_recording": False,
        "robot_asset": asset,
        "robot_asset_changed_during_recording": False,
        "runtime_file_sha256": recorder._runtime_file_hashes("mujoco"),
        "runtime_files_changed_during_recording": False,
        "source_robot_variant": "G1",
        "runtime_robot": "G1",
        "instance_index": 0,
        "problem_index": 0,
        "camera_profile": "hardware",
        "camera_names": ["head_rgb"],
        "camera_config": None,
        "camera_config_changed_during_recording": False,
        "capture": {"fps": 30, "frames": 2, "camera_image_size_argument": 64},
        "video_files": ["mujoco_head_rgb.mp4"],
        "video_sha256": {"mujoco_head_rgb.mp4": hashlib.sha256(b"video").hexdigest()},
        "video_decode": {"mujoco_head_rgb.mp4": {"frames": 2, "error": None}},
    }
    (output / "scene_video_report.json").write_text(json.dumps(existing), encoding="utf-8")
    assert (
        recorder.record_one(
            simulator="mujoco", run_id="run", output=output, image_size=64, frames=2, camera_names=("head_rgb",)
        )
        == existing
    )
    monkeypatch.setattr(recorder, "_robot_asset_info", lambda *args: {**asset, "files_sha256": {"mjcf": "model_b"}})
    with pytest.raises(FileExistsError, match="different or incomplete"):
        recorder.record_one(
            simulator="mujoco", run_id="run", output=output, image_size=64, frames=2, camera_names=("head_rgb",)
        )


def test_geniesim_action_replay_retains_video_and_exception(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch

    import roboverse_pack.tasks.icl.geniesim_benchmark as benchmark
    import roboverse_pack.tasks.icl.geniesim_scene as scene
    from tools.icl import record_geniesim_benchmark_video as recorder

    observation = SimpleNamespace(
        cameras={"head_rgb": SimpleNamespace(rgb=torch.tensor([[[[0, 0, 0], [255, 255, 255]]]], dtype=torch.uint8))},
        objects={"block": SimpleNamespace(root_state=torch.zeros((1, 7)))},
        robots={"robot": SimpleNamespace(joint_pos=torch.zeros((1, 1)))},
    )
    received = []

    class DummyTask:
        def __init__(self, **kwargs):
            received.append(kwargs)
            self.handler = SimpleNamespace(env_step_s=1 / 30)
            self.robot_name = "robot"
            self._control_indices = [0]
            self.calls = 0
            self.run = {"robot_adaptation": {"runtime_robot": kwargs.get("robot_variant") or "G1"}}
            self.scenario = SimpleNamespace(cameras=[SimpleNamespace(name="head_rgb", width=2, height=1)])

        def reset(self):
            return observation, None

        def step(self, action):
            self.calls += 1
            if self.calls == 2:
                raise ValueError("unreachable")
            return (
                observation,
                torch.tensor([0.0]),
                torch.tensor([False]),
                torch.tensor([False]),
                {
                    "geniesim_checker_status": "running",
                    "geniesim_internal_target_clipping": {},
                },
            )

        def close(self):
            pass

    class DummyWriter:
        def __init__(self, path, width, height, fps):
            self.path = path

        def append(self, frame):
            pass

        def close(self):
            self.path.write_bytes(b"video")

    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"source": {"content_sha256": "a" * 64}}), encoding="utf-8")
    monkeypatch.setattr(recorder, "SPEC_PATH", spec)
    monkeypatch.setattr(recorder, "source_content_sha256", lambda pack: "a" * 64)
    monkeypatch.setattr(recorder, "RGBVideoWriter", DummyWriter)
    monkeypatch.setattr(
        recorder,
        "_robot_asset_info",
        lambda robot_type, robot_root: {"root": str(robot_root), "files_sha256": {"mjcf": "expected"}},
    )
    monkeypatch.setattr(benchmark, "GenieSimBenchmarkTask", DummyTask)
    monkeypatch.setattr(
        scene, "load_run_instance", lambda *args: ({"robot": "g1_op"}, {"task": "t", "instance_id": "0"}, {})
    )
    path = tmp_path / "actions.json"
    path.write_text(
        json.dumps({
            "format": "roboverse.icl.benchmark_action_sequence.v1",
            "run_id": "run",
            "instance_index": 0,
            "problem_index": 0,
            "action_mode": "tcp_delta",
            "benchmark_source_content_sha256": "a" * 64,
            "step_seconds": 1 / 30,
            "source": {"type": "test"},
            "actions": [[0.0] * 7, [0.0] * 7],
        }),
        encoding="utf-8",
    )
    output = tmp_path / "replay"
    camera_config = tmp_path / "cameras.json"
    calibration = {
        "projection_model": "opencv_fisheye",
        "calibration_size": [640, 480],
        "optical_center": [320, 240],
        "focal_length_px": [250, 250],
        "coefficients": [0, 0, 0, 0],
        "max_fov_rad": 3.0,
    }
    camera_config.write_text(
        json.dumps({
            "format": "roboverse.icl.geniesim_camera_calibration.v1",
            "cameras": {"head_rgb": calibration},
        }),
        encoding="utf-8",
    )
    report = recorder.record_action_replay(
        simulator="mujoco",
        run_id="run",
        output=output,
        actions_path=path,
        image_size=64,
        robot_root=tmp_path,
        camera_names=("head_rgb",),
        camera_config=camera_config,
    )
    assert report["outcome"] == "action_exception"
    assert report["exception"]["action_index"] == 1
    assert len(report["steps"]) == 1
    assert (output / "mujoco_head_rgb.mp4").read_bytes() == b"video"
    assert (output / "actions.json").read_bytes() == path.read_bytes()
    assert received[0]["robot_root"] == tmp_path
    assert received[0]["camera_names"] == ("head_rgb",)
    assert received[0]["camera_calibrations"] == {"head_rgb": calibration}
    assert report["camera_names"] == ["head_rgb"]
    assert report["camera_config"]["sha256"] == recorder._digest(camera_config.read_bytes())
    assert report["camera_config_changed_during_replay"] is False
    assert report["robot_asset"]["files_sha256"] == {"mjcf": "expected"}
    assert report["action_sequence_robot_asset_pinned"] is False
    assert report["source_robot"] == report["runtime_robot"] == "G1"
    assert report["action_robot_variant_inferred_from_source"] is True
    with pytest.raises(ValueError, match=r"sequence is for G1.*runtime robot is G2"):
        recorder.record_action_replay(
            simulator="mujoco",
            run_id="run",
            output=tmp_path / "cross_robot",
            actions_path=path,
            image_size=64,
            robot_variant="G2",
        )
    mismatched = json.loads(path.read_text(encoding="utf-8"))
    mismatched["benchmark_source_content_sha256"] = "b" * 64
    path.write_text(json.dumps(mismatched), encoding="utf-8")
    with pytest.raises(ValueError, match="different benchmark source snapshot"):
        recorder.record_action_replay(
            simulator="mujoco", run_id="run", output=tmp_path / "different", actions_path=path, image_size=64
        )
    mismatched["benchmark_source_content_sha256"] = "a" * 64
    mismatched["robot_asset_sha256"] = {"mjcf": "wrong"}
    path.write_text(json.dumps(mismatched), encoding="utf-8")
    with pytest.raises(ValueError, match="different robot asset"):
        recorder.record_action_replay(
            simulator="mujoco", run_id="run", output=tmp_path / "different_robot", actions_path=path, image_size=64
        )
    mismatched.pop("robot_asset_sha256")
    mismatched["robot_variant"] = "G2"
    path.write_text(json.dumps(mismatched), encoding="utf-8")
    cross = recorder.record_action_replay(
        simulator="mujoco",
        run_id="run",
        output=tmp_path / "cross_robot",
        actions_path=path,
        image_size=64,
        robot_variant="G2",
        camera_names=("head_rgb",),
    )
    assert cross["source_robot"] == "G1" and cross["runtime_robot"] == "G2"
    assert cross["action_robot_variant"] == "G2" and cross["action_robot_variant_inferred_from_source"] is False
    assert cross["robot_adaptation"]["runtime_robot"] == "G2"


def test_geniesim_benchmark_task_action_vector_has_stable_joint_order():
    from types import SimpleNamespace

    import torch

    from roboverse_pack.tasks.icl.fruit_sorting import CONTROL_JOINTS
    from roboverse_pack.tasks.icl.geniesim_benchmark import GenieSimBenchmarkTask

    task = object.__new__(GenieSimBenchmarkTask)
    task.action_mode = "joint_position"
    task.control_joint_names = CONTROL_JOINTS
    task.robot_name = "agibot_g1"
    task.scenario = SimpleNamespace(
        robots=[SimpleNamespace(joint_limits={name: (-1.0, 1.0) for name in CONTROL_JOINTS})]
    )
    values = torch.linspace(-0.5, 0.5, len(CONTROL_JOINTS))
    action = task._process_action(values)
    assert list(action[0]["agibot_g1"]["dof_pos_target"]) == list(CONTROL_JOINTS)
    np.testing.assert_allclose(list(action[0]["agibot_g1"]["dof_pos_target"].values()), values.numpy())
    with pytest.raises(ValueError, match="finite joint position targets"):
        task._process_action(torch.full((len(CONTROL_JOINTS),), torch.nan))
    with pytest.raises(ValueError, match="outside its joint limits"):
        task._process_action(torch.full((len(CONTROL_JOINTS),), 1.1))
    gripper = "idx41_gripper_l_outer_joint1"
    task.scenario.robots[0].joint_limits[gripper] = (0.0, np.pi / 4)
    at_limit = torch.zeros(len(CONTROL_JOINTS), dtype=torch.float32)
    at_limit[CONTROL_JOINTS.index(gripper)] = np.pi / 4
    clamped = task._process_action(at_limit)[0]["agibot_g1"]["dof_pos_target"][gripper]
    assert clamped == pytest.approx(np.pi / 4, abs=1e-12)


def test_geniesim_reset_settling_holds_joint_pose_without_consuming_episode_steps():
    from types import SimpleNamespace

    import torch

    from roboverse_pack.tasks.icl.geniesim_benchmark import GenieSimBenchmarkTask

    class Handler:
        def __init__(self):
            self.targets = None
            self.steps = 0

        def set_dof_targets(self, targets):
            self.targets = targets

        def simulate(self):
            self.steps += 1

        def get_states(self, mode):
            assert mode == "tensor"
            return "settled state"

    task = object.__new__(GenieSimBenchmarkTask)
    task.reset_settle_seconds = 0.21
    task._delta_time = 0.1
    task._control_indices = [0, 2]
    task.robot_name = "robot"
    task.handler = Handler()
    task.position_action = lambda targets: targets.detach().cpu().tolist()
    task._observation = lambda states: states
    task._last_reset_settle_steps = 0
    observation = SimpleNamespace(robots={"robot": SimpleNamespace(joint_pos=torch.tensor([[0.25, -0.5, 0.75]]))})

    result = task._settle_reset_state(observation)

    assert result == "settled state"
    assert task.handler.targets == [0.25, 0.75]
    assert task.handler.steps == 3
    assert task._last_reset_settle_steps == 3


def test_geniesim_benchmark_cartesian_action_maps_to_named_joint_targets():
    from types import SimpleNamespace

    import torch

    from roboverse_pack.tasks.icl.fruit_sorting import CONTROL_JOINTS
    from roboverse_pack.tasks.icl.geniesim_benchmark import GenieSimBenchmarkTask

    class CartesianController:
        def joint_targets(self, current_controls, action):
            np.testing.assert_array_equal(current_controls, np.arange(len(CONTROL_JOINTS)))
            np.testing.assert_allclose(action, [0.001, 0, 0, 0, 0, 0, 1])
            return np.linspace(-0.5, 0.5, len(CONTROL_JOINTS))

    task = object.__new__(GenieSimBenchmarkTask)
    task.action_mode = "tcp_delta"
    task.control_joint_names = CONTROL_JOINTS
    task.robot_name = "agibot_g1"
    task._control_indices = list(range(len(CONTROL_JOINTS)))
    task._cartesian_controller = CartesianController()
    task.handler = SimpleNamespace(
        get_states=lambda mode: SimpleNamespace(
            robots={"agibot_g1": SimpleNamespace(joint_pos=torch.arange(len(CONTROL_JOINTS)).reshape(1, -1))}
        )
    )
    task.scenario = SimpleNamespace(
        robots=[SimpleNamespace(joint_limits={name: (-1.0, 1.0) for name in CONTROL_JOINTS})]
    )
    # SynthICL commonly supplies a batched torch tensor (possibly on CUDA).
    action = torch.tensor([[0.001, 0, 0, 0, 0, 0, 1]], dtype=torch.float32)
    assert task._action_space().shape == (7,)
    result = task._process_action(action)
    assert list(result[0]["agibot_g1"]["dof_pos_target"]) == list(CONTROL_JOINTS)
    np.testing.assert_allclose(list(result[0]["agibot_g1"]["dof_pos_target"].values()), np.linspace(-0.5, 0.5, 20))


def test_geniesim_cartesian_action_clips_out_of_range_measured_hold_targets():
    from types import SimpleNamespace

    import torch

    from roboverse_pack.tasks.icl.fruit_sorting import CONTROL_JOINTS
    from roboverse_pack.tasks.icl.geniesim_benchmark import GenieSimBenchmarkTask

    gripper = "idx41_gripper_l_outer_joint1"
    index = CONTROL_JOINTS.index(gripper)
    measured = torch.zeros((1, len(CONTROL_JOINTS)))
    measured[0, index] = 1.05

    class RightArmController:
        def joint_targets(self, current, action):
            return current.copy()

    task = object.__new__(GenieSimBenchmarkTask)
    task.action_mode = "tcp_delta"
    task.control_joint_names = CONTROL_JOINTS
    task.robot_name = "agibot_g1"
    task._control_indices = list(range(len(CONTROL_JOINTS)))
    task._cartesian_controller = RightArmController()
    task.handler = SimpleNamespace(
        get_states=lambda mode: SimpleNamespace(robots={"agibot_g1": SimpleNamespace(joint_pos=measured)})
    )
    task.scenario = SimpleNamespace(
        robots=[SimpleNamespace(joint_limits={name: (-1.0, 1.0) for name in CONTROL_JOINTS})]
    )
    targets = task._process_action([0, 0, 0, 0, 0, 0, 1])[0]["agibot_g1"]["dof_pos_target"]
    assert targets[gripper] == 1.0
    diagnostic = task._internal_target_clipping[gripper]
    assert diagnostic["measured_position"] == pytest.approx(1.05)
    assert diagnostic["unclipped_target"] == pytest.approx(1.05)
    assert diagnostic["applied_target"] == 1.0
    with pytest.raises(ValueError, match="outside its joint limits"):
        task.position_action(measured[0])


def test_geniesim_single_tcp_arm_uses_source_gripper_and_rejects_ambiguity():
    from roboverse_pack.tasks.icl.geniesim_benchmark import _single_tcp_arm

    instance = {
        "instructions": {"instructions": [{"instruction": "Pick up the drink", "gripper": {"1": {"side": "left"}}}]},
        "checker_graphs": {"problem1": {"Follow": "bottle|[0.1,0.2,0.1]|left_gripper"}},
    }
    assert _single_tcp_arm(instance, "problem1", 0, None) == "left"
    assert _single_tcp_arm(instance, "problem1", 0, "right") == "right"
    instance["checker_graphs"]["problem1"] = {"Follow": "bottle|[0.1,0.2,0.1]|right_gripper"}
    with pytest.raises(ValueError, match="conflicts with the checker"):
        _single_tcp_arm(instance, "problem1", 0, None)
    instance["instructions"]["instructions"][0]["gripper"]["1"]["side"] = "both"
    with pytest.raises(ValueError, match="both arms"):
        _single_tcp_arm(instance, "problem1", 0, None)


def test_geniesim_bimanual_tcp_combines_independent_arm_targets():
    from types import SimpleNamespace

    import torch

    from roboverse_pack.tasks.icl.fruit_sorting import CONTROL_JOINTS
    from roboverse_pack.tasks.icl.geniesim_benchmark import GenieSimBenchmarkTask

    class Controller:
        def __init__(self, arm_name, gripper_name, value):
            self.arm_input_indices = np.array([CONTROL_JOINTS.index(arm_name)])
            self.gripper_name = gripper_name
            self.value = value

        def joint_targets(self, current, action):
            np.testing.assert_array_equal(current, np.zeros(len(CONTROL_JOINTS)))
            result = current.copy()
            result[self.arm_input_indices] = self.value
            result[CONTROL_JOINTS.index(self.gripper_name)] = action[6]
            return result

    task = object.__new__(GenieSimBenchmarkTask)
    task.action_mode = "bimanual_tcp_delta"
    task.control_joint_names = CONTROL_JOINTS
    task.robot_name = "agibot_g1"
    task._control_indices = list(range(len(CONTROL_JOINTS)))
    task._bimanual_controllers = (
        Controller("idx21_arm_l_joint1", "idx41_gripper_l_outer_joint1", 0.2),
        Controller("idx61_arm_r_joint1", "idx81_gripper_r_outer_joint1", -0.3),
    )
    task.handler = SimpleNamespace(
        get_states=lambda mode: SimpleNamespace(
            robots={"agibot_g1": SimpleNamespace(joint_pos=torch.zeros((1, len(CONTROL_JOINTS))))}
        )
    )
    task.scenario = SimpleNamespace(
        robots=[SimpleNamespace(joint_limits={name: (-1.0, 1.0) for name in CONTROL_JOINTS})]
    )
    assert task._action_space().shape == (14,)
    result = task._process_action([0, 0, 0, 0, 0, 0, 0.4, 0, 0, 0, 0, 0, 0, 0.7])
    targets = result[0]["agibot_g1"]["dof_pos_target"]
    assert targets["idx21_arm_l_joint1"] == pytest.approx(0.2)
    assert targets["idx61_arm_r_joint1"] == pytest.approx(-0.3)
    assert targets["idx41_gripper_l_outer_joint1"] == pytest.approx(0.4)
    assert targets["idx81_gripper_r_outer_joint1"] == pytest.approx(0.7)
    with pytest.raises(ValueError, match="14 finite values"):
        task._process_action([0.0] * 7)


@pytest.mark.parametrize("robot_name", ["G1", "G2"])
def test_geniesim_cartesian_controllers_move_either_arm_in_world_frame(robot_name):
    pytest.importorskip("mujoco")
    from roboverse_pack.tasks.icl.control import G1CartesianController, G2CartesianController
    from roboverse_pack.tasks.icl.fruit_sorting import CONTROL_JOINTS, ROBOT_ROOT, build_geniesim_g1_robot
    from roboverse_pack.tasks.icl.geniesim_benchmark import G2_CONTROL_JOINTS, G2_ROBOT_ROOT, build_geniesim_g2_robot

    if robot_name == "G1":
        root, names, builder, controller_class = (
            ROBOT_ROOT,
            CONTROL_JOINTS,
            build_geniesim_g1_robot,
            G1CartesianController,
        )
    else:
        root, names, builder, controller_class = (
            G2_ROBOT_ROOT,
            G2_CONTROL_JOINTS,
            build_geniesim_g2_robot,
            G2CartesianController,
        )
    if not (root / "provenance.json").is_file():
        pytest.skip(f"Prepare the {robot_name} Omnipicker before testing its kinematics")
    robot = builder(root)
    controls = np.array([robot.default_joint_positions[name] for name in names])
    for side in ("left", "right"):
        controller = controller_class(robot, names, arm_side=side)
        before, _ = controller.tcp_pose(controls)
        targets = controller.joint_targets(controls, [0.005, 0, 0, 0, 0, 0, 0.5])
        after, _ = controller.tcp_pose(targets)
        np.testing.assert_allclose(after - before, [0.005, 0, 0], atol=0.001, rtol=0.0)
        other_prefix = "arm_r" if side == "left" else "arm_l"
        other_indices = [index for index, name in enumerate(names) if other_prefix in name]
        np.testing.assert_array_equal(targets[other_indices], controls[other_indices])


def test_geniesim_g1_cartesian_ik_refines_reachable_pose_near_joint_limit(monkeypatch):
    pytest.importorskip("mujoco")
    from roboverse_pack.tasks.icl import control as control_module
    from roboverse_pack.tasks.icl.control import G1CartesianController
    from roboverse_pack.tasks.icl.fruit_sorting import CONTROL_JOINTS, ROBOT_ROOT, build_geniesim_g1_robot

    if not (ROBOT_ROOT / "provenance.json").is_file():
        pytest.skip("Prepare the G1 Omnipicker before testing its Cartesian IK fallback")
    robot = build_geniesim_g1_robot(ROBOT_ROOT)
    controller = G1CartesianController(robot, CONTROL_JOINTS, arm_side="right")
    controls = np.asarray([robot.default_joint_positions[name] for name in CONTROL_JOINTS], dtype=float)
    cases = (
        (
            np.asarray([1.74554693, -1.26010207, -0.56808433, 0.00422722, -1.09478704, -1.57676082, -0.28597945]),
            np.asarray([0.0029326001, 0.0015585398, -0.0022373060, 0.0, 0.0, 0.0, 1.0]),
            None,
        ),
        (
            np.asarray([1.748, -1.480, -0.626, 0.0, -1.015, -1.056, -0.286]),
            np.asarray([0.0, 0.0, -0.0025, 0.0, 0.0, 0.0, 1.0]),
            0.1,
        ),
    )

    real_least_squares = control_module.least_squares
    fallback_calls = 0

    def track_fallback(*args, **kwargs):
        nonlocal fallback_calls
        fallback_calls += 1
        return real_least_squares(*args, **kwargs)

    monkeypatch.setattr(control_module, "least_squares", track_fallback)
    for source_arm_pose, action, max_joint_step in cases:
        controls[controller.arm_input_indices] = source_arm_pose
        start_position, start_rotation = controller.tcp_pose(controls)
        targets = controller.joint_targets(controls, action)
        final_position, final_rotation = controller.tcp_pose(targets)

        np.testing.assert_allclose(final_position, start_position + action[:3], atol=0.001, rtol=0.0)
        assert np.linalg.norm((start_rotation * final_rotation.inv()).as_rotvec()) < 0.02
        if max_joint_step is not None:
            assert np.max(np.abs(targets[controller.arm_input_indices] - source_arm_pose)) < max_joint_step

    assert fallback_calls == len(cases)


@pytest.mark.general
def test_geniesim_g1_checker_graphs_compile_for_all_imported_instances():
    from roboverse_pack.tasks.icl.geniesim_checker import GenieSimChecker, available_problem_ids

    root = Path(__file__).resolve().parents[1]
    spec_path = root / "roboverse_data/icl/geniesim_benchmark_specs/benchmark.json"
    if not spec_path.is_file():
        pytest.skip("Prepare the imported Genie Sim benchmark specs first")
    pack = json.loads(spec_path.read_text(encoding="utf-8"))
    instance_keys = set()
    compiled_graphs = 0
    for run in pack["runs"]:
        evaluation = pack["evaluation_configs"][run["evaluation_config"]]["config"]
        if evaluation.get("robot", {}).get("robot_id") != "G1":
            continue
        for instance_key in run["instances"]:
            if instance_key in instance_keys:
                continue
            instance_keys.add(instance_key)
            instance = pack["instances"][instance_key]
            problem_ids = available_problem_ids(instance)
            assert problem_ids, f"No imported checker graphs for {instance_key}"
            for problem_index in range(len(problem_ids)):
                GenieSimChecker.from_instance(instance, problem_index=problem_index)
                compiled_graphs += 1
    assert len(instance_keys) == 145
    assert compiled_graphs == 310


@pytest.mark.general
def test_geniesim_g1_scenarios_build_for_all_source_instances(tmp_path):
    from roboverse_pack.tasks.icl.fruit_sorting import ROBOT_ROOT, build_geniesim_g1_robot
    from roboverse_pack.tasks.icl.geniesim_scene import build_run_scenario

    root = Path(__file__).resolve().parents[1]
    spec_path = root / "roboverse_data/icl/geniesim_benchmark_specs_source_scene_v2/benchmark.json"
    assets_root = root / "roboverse_data/icl/objects/geniesim"
    robot_root = Path(os.environ.get("ROBOVERSE_GENIESIM_G1_CLOSED_ASSET", ROBOT_ROOT))
    if not spec_path.is_file() or not robot_root.joinpath("provenance.json").is_file():
        pytest.skip("Prepare the imported Genie Sim specs and G1 Omnipicker before scenario integration tests")
    pack = json.loads(spec_path.read_text(encoding="utf-8"))
    g1_runs = [
        run
        for run in pack["runs"]
        if pack["evaluation_configs"][run["evaluation_config"]]["config"].get("robot", {}).get("robot_id") == "G1"
    ]
    required_assets = {
        asset_id
        for run in g1_runs
        for instance_key in run["instances"]
        for asset_id in pack["instances"][instance_key].get("asset_ids", [])
    }
    missing_assets = sorted(
        asset_id for asset_id in required_assets if not (assets_root / asset_id / "provenance.json").is_file()
    )
    if missing_assets:
        pytest.skip(f"Prepare all referenced G1 scene assets first; missing: {missing_assets[:8]}")
    robot = build_geniesim_g1_robot(robot_root)
    built = 0
    for run in g1_runs:
        for instance_index in range(len(run["instances"])):
            scenario, _, instance, _ = build_run_scenario(
                spec_path=spec_path,
                run_id=run["id"],
                simulator="mujoco",
                robot=robot,
                assets_root=assets_root,
                cache_root=tmp_path / "wrappers",
                instance_index=instance_index,
                headless=True,
                cameras=(),
            )
            assert scenario.robots[0].name == "agibot_g1"
            assert scenario.simulator == "mujoco"
            assert instance["scene_info"].get("layout")
            built += 1
    assert built == 145


@pytest.mark.examples
@pytest.mark.parametrize("simulator", ["mujoco", "genesis", "newton"])
@pytest.mark.parametrize(
    "run_id",
    [
        "g1op_s2r_bimanual_chip_handover",
        "g1op_s2r_grasp_targets",
        "g1op_s2r_organize_items",
        "g1op_s2r_pack_in_supermarket",
        "g1op_s2r_place_block_into_drawer",
        "g1op_s2r_select_color",
        "g1op_s2r_size_recognize",
        "g1op_s2r_sort_fruit",
    ],
)
def test_geniesim_registered_task_runs_checker_on_shared_backend_states(simulator, run_id):
    import torch

    requested = set(os.environ.get("ROBOVERSE_ICL_TEST_BACKENDS", "").split(","))
    if simulator != "mujoco" and simulator not in requested:
        pytest.skip(f"Set ROBOVERSE_ICL_TEST_BACKENDS to include {simulator}")
    if importlib.util.find_spec(simulator) is None:
        pytest.skip(f"{simulator} is not installed")
    root = Path(__file__).resolve().parents[1]
    spec_path = root / "roboverse_data/icl/geniesim_benchmark_specs_source_scene_v2/benchmark.json"
    assets_root = root / "roboverse_data/icl/objects/geniesim"
    robot_root = root / "roboverse_data/icl/robots/geniesim_g1_omnipicker_closed_v6"
    if not spec_path.is_file() or not robot_root.joinpath("provenance.json").is_file():
        pytest.skip("Prepare the local Genie Sim specs and G1 robot before simulator integration tests")
    from roboverse_pack.tasks.icl.geniesim_scene import load_run_instance

    _, source_instance, _ = load_run_instance(spec_path, run_id, instance_index=0)
    required_assets = source_instance.get("asset_ids", [])
    if any(not (assets_root / name / "provenance.json").is_file() for name in required_assets):
        pytest.skip(f"Prepare the scene assets for {run_id}")
    if simulator == "newton":
        if not torch.cuda.is_available():
            pytest.skip("Newton integration test requires a CUDA GPU")

    from roboverse_pack.tasks.icl.fruit_sorting import CONTROL_JOINTS
    from roboverse_pack.tasks.icl.geniesim_benchmark import GenieSimBenchmarkTask

    task = GenieSimBenchmarkTask(
        run_id=run_id,
        simulator=simulator,
        spec_path=spec_path,
        assets_root=assets_root,
        cache_root=root / "roboverse_data/icl/cache/geniesim_scene_wrappers_v6",
        robot_root=robot_root,
        instance_index=0,
        problem_index=0,
        image_size=64,
        max_episode_steps=3,
    )
    try:
        observation, _ = task.reset()
        assert set(observation.cameras) == {"head_rgb", "head_fisheye_left", "head_fisheye_right"}
        for camera in observation.cameras.values():
            frame = camera.rgb[0].detach().cpu().numpy()
            assert frame.shape[-1] == 3 and frame.std() > 1.0
        for name in ("head_fisheye_left", "head_fisheye_right"):
            camera = observation.cameras[name]
            assert camera.valid_mask is not None
            assert camera.valid_mask.dtype == torch.bool
            assert camera.valid_mask.shape == camera.rgb.shape[:3]
            assert camera.valid_mask.any() and not camera.valid_mask.all()
        default_targets = [task.scenario.robots[0].default_joint_positions[name] for name in CONTROL_JOINTS]
        for _ in range(2):
            _, _, terminated, _, info = task.step(default_targets)
            assert terminated.shape == (1,)
            assert info["geniesim_run_id"] == run_id
            assert info["geniesim_checker_status"] in {"running", "success", "failure"}
    finally:
        task.close()


@pytest.mark.examples
@pytest.mark.parametrize("simulator", ["mujoco", "genesis", "newton"])
def test_geniesim_g1_task_accepts_synthicl_cartesian_actions(simulator):
    requested = set(os.environ.get("ROBOVERSE_ICL_TEST_BACKENDS", "").split(","))
    if simulator != "mujoco" and simulator not in requested:
        pytest.skip(f"Set ROBOVERSE_ICL_TEST_BACKENDS to include {simulator}")
    if importlib.util.find_spec(simulator) is None:
        pytest.skip(f"{simulator} is not installed")
    if simulator == "newton":
        import torch

        if not torch.cuda.is_available():
            pytest.skip("Newton integration test requires a CUDA GPU")
    root = Path(__file__).resolve().parents[1]
    spec_path = root / "roboverse_data/icl/geniesim_benchmark_specs/benchmark.json"
    assets_root = root / "roboverse_data/icl/objects/geniesim"
    robot_root = root / "roboverse_data/icl/robots/geniesim_g1_omnipicker_closed_v5"
    if not spec_path.is_file() or not robot_root.joinpath("provenance.json").is_file():
        pytest.skip("Prepare the local Genie Sim specs and G1 robot before simulator integration tests")
    from roboverse_pack.tasks.icl.geniesim_scene import load_run_instance

    _, source_instance, _ = load_run_instance(spec_path, "g1op_s2r_sort_fruit", instance_index=0)
    if any(not (assets_root / name / "provenance.json").is_file() for name in source_instance.get("asset_ids", [])):
        pytest.skip("Prepare the Genie Sim sort-fruit scene assets first")
    import torch

    from roboverse_learn.il.icl.data import ACTION_SPEC
    from roboverse_pack.tasks.icl.geniesim_benchmark import GenieSimBenchmarkTask

    task = GenieSimBenchmarkTask(
        run_id="g1op_s2r_sort_fruit",
        simulator=simulator,
        spec_path=spec_path,
        assets_root=assets_root,
        cache_root=root / "roboverse_data/icl/cache/geniesim_scene_wrappers_v6",
        robot_root=robot_root,
        image_size=64,
        action_mode="tcp_delta",
        max_episode_steps=3,
    )
    try:
        observation, _ = task.reset()
        assert task.action_space.shape == (7,)
        assert set(observation.cameras) == {"head_rgb", "head_fisheye_left", "head_fisheye_right"}
        import torch

        action = torch.tensor([[0.001, 0, 0, 0, 0, 0, 1]], dtype=torch.float32)
        _, _, _, _, info = task.step(action)
        assert info["geniesim_action_spec"] == ACTION_SPEC
        assert info["geniesim_checker_status"] in {"running", "success", "failure"}
    finally:
        task.close()


def test_geniesim_g1_camera_bundle_uses_three_source_calibrations(tmp_path):
    from roboverse_pack.tasks.icl.geniesim_scene import geniesim_g1_cameras, geniesim_g2_cameras

    robot = tmp_path / "robot"
    robot.mkdir()
    names = ("Head_Camera", "Fisheye_Camera_01", "Fisheye_Camera_02")
    cameras = {}
    for index, name in enumerate(names):
        matrix = np.eye(4)
        matrix[3, :3] = (0.01 * index, 0.02 * index, 0.03 * index)
        cameras[name] = {
            "focalLength": 1.93,
            "horizontalAperture": 3.896,
            "verticalAperture": 2.453,
            "cameraProjectionType": "fisheyePolynomial" if index else "pinhole",
            "head_link_from_camera_usd_row_matrix": matrix.tolist(),
        }
    cameras["Head_Camera"]["native_image_size"] = [1280, 800]
    cameras["Head_Camera"]["opencv_pinhole"] = {"fx": 646.0, "fy": 646.0, "cx": 640.0, "cy": 400.0}
    (robot / "provenance.json").write_text(
        json.dumps({
            "source_revision": "a0813aad7c16165daffbe6c2737754e0344809ee",
            "cameras": cameras,
        }),
        encoding="utf-8",
    )

    result = geniesim_g1_cameras(robot, image_size=320, fisheye_projection="pinhole")
    g2 = geniesim_g2_cameras(320, camera_profile="hardware", fisheye_projection="pinhole")
    assert [camera.name for camera in result] == ["head_rgb", "head_fisheye_left", "head_fisheye_right"]
    assert all(camera.mount_to == "agibot_g1" and camera.mount_link == "head_link2" for camera in result)
    assert [(camera.width, camera.height) for camera in result] == [(320, 200), (320, 256), (320, 256)]
    assert result[0].focal_length == pytest.approx(g2[0].focal_length)
    np.testing.assert_allclose(result[0].intrinsic, g2[0].intrinsic)
    assert result[1].mount_pos == pytest.approx((0.01, 0.02, 0.03))
    assert result[1].focal_length == pytest.approx(g2[1].focal_length)


def test_prepared_g1_v6_head_calibration_has_verified_source_and_native_size():
    import hashlib

    from roboverse_pack.tasks.icl.geniesim_scene import geniesim_g1_cameras, geniesim_g2_cameras

    root = Path(__file__).resolve().parents[1]
    robot = root / "roboverse_data/icl/robots/geniesim_g1_omnipicker_closed_v6"
    source = root / "datasets/icl/sources/geniesim-assets/robot/G1_omnipicker/robot.usda"
    if not (robot / "provenance.json").is_file() or not source.is_file():
        pytest.skip("Prepare the verified G1 v6 robot and source USD first")
    info = json.loads((robot / "provenance.json").read_text(encoding="utf-8"))
    assert info["camera_calibration_source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert info["cameras"]["Head_Camera"]["native_image_size"] == [1280, 800]
    result = geniesim_g1_cameras(robot, image_size=640, fisheye_projection="pinhole")
    g2 = geniesim_g2_cameras(640, camera_profile="hardware", fisheye_projection="pinhole")
    assert [(camera.width, camera.height) for camera in result] == [(640, 400), (640, 512), (640, 512)]
    np.testing.assert_allclose(result[0].intrinsic, g2[0].intrinsic)
    # The source has no native head-fisheye pixel dimensions. Those two
    # pinhole approximations use the authored aperture ratio, not fabricated calibration.
    assert "native_image_size" not in info["cameras"]["Fisheye_Camera_01"]
    assert "native_image_size" not in info["cameras"]["Fisheye_Camera_02"]


def test_geniesim_g1_borrows_all_g2_optics_but_keeps_g1_mounts():
    from metasim.scenario.cameras import FisheyeCameraCfg
    from roboverse_pack.tasks.icl.geniesim_scene import geniesim_g1_cameras, geniesim_g2_cameras

    root = Path(__file__).resolve().parents[1]
    robot = root / "roboverse_data/icl/robots/geniesim_g1_omnipicker_closed_v6"
    source = root / "datasets/icl/sources/geniesim-assets/robot/G2_omnipicker/robot.usda"
    if not (robot / "provenance.json").is_file() or not source.is_file():
        pytest.skip("Prepare the verified G1 robot and Genie Sim G2 source USD first")
    info = json.loads((robot / "provenance.json").read_text(encoding="utf-8"))
    g1 = geniesim_g1_cameras(robot, image_size=320)
    g2 = geniesim_g2_cameras(320, camera_profile="hardware", source_robot_usd=source)
    assert g1[0].focal_length == pytest.approx(g2[0].focal_length)
    assert g1[0].horizontal_aperture == pytest.approx(g2[0].horizontal_aperture)
    np.testing.assert_allclose(g1[0].intrinsic, g2[0].intrinsic)
    assert g1[0].opencv_distortion == pytest.approx(g2[0].opencv_distortion)
    assert g1[0].distortion_source == g2[0].distortion_source
    assert g1[0].distortion_source.startswith("geniesim_g2_usd:")
    assert g1[0].mount_to == "agibot_g1" and g1[0].mount_link == "head_link2"
    expected_g1_mount = info["cameras"]["Head_Camera"]["head_link_from_camera_usd_row_matrix"][3][:3]
    assert g1[0].mount_pos == pytest.approx(expected_g1_mount)
    for left, right in zip(g1[1:], g2[1:], strict=True):
        assert isinstance(left, FisheyeCameraCfg) and isinstance(right, FisheyeCameraCfg)
        assert left.projection_model == right.projection_model == "opencv_fisheye"
        assert left.calibration_size == right.calibration_size == (1920, 1536)
        assert left.focal_length_px == pytest.approx(right.focal_length_px)
        assert left.optical_center == pytest.approx(right.optical_center)
        assert left.coefficients == pytest.approx(right.coefficients)
        assert left.max_fov_rad == pytest.approx(right.max_fov_rad)
        assert left.mount_to == "agibot_g1" and left.mount_link == "head_link2"
        assert right.mount_to == "agibot_g2" and right.mount_link == "head_link3"
        assert left.data_types == ["rgb"] and right.data_types == ["rgb"]


def test_geniesim_observation_returns_distorted_rgb_and_validity_mask():
    pytest.importorskip("cv2")
    from types import SimpleNamespace

    import torch

    from metasim.scenario.cameras import PinholeCameraCfg
    from metasim.types import CameraState, TensorState
    from roboverse_pack.tasks.icl.geniesim_benchmark import GenieSimBenchmarkTask

    camera = PinholeCameraCfg(
        name="head_rgb",
        width=64,
        height=48,
        intrinsic=[[48.0, 0.0, 31.5], [0.0, 48.0, 23.5], [0.0, 0.0, 1.0]],
        opencv_distortion=(-0.35, 0.08, 0.0, 0.0, 0.0),
    )
    task = object.__new__(GenieSimBenchmarkTask)
    task.scenario = SimpleNamespace(cameras=[camera])
    rgb = torch.zeros((1, 48, 64, 3), dtype=torch.uint8)
    rgb[:, :, 32:, :] = 255
    raw_state = TensorState(objects={}, robots={}, cameras={"head_rgb": CameraState(rgb=rgb, depth=None)})

    observation = task._observation(raw_state)

    assert observation is not raw_state
    assert not torch.equal(observation.cameras["head_rgb"].rgb, rgb)
    assert observation.cameras["head_rgb"].valid_mask.shape == (1, 48, 64)
    assert observation.cameras["head_rgb"].valid_mask.dtype == torch.bool
    assert torch.equal(raw_state.cameras["head_rgb"].rgb, rgb)


def test_genesis_fixed_collisionless_mjcf_assets_use_kinematic_material(tmp_path):
    pytest.importorskip("genesis")
    import genesis as gs

    from metasim.sim.genesis.genesis import (
        _mjcf_entity_material,
        _mjcf_meshes_are_explicitly_noncolliding,
    )

    collisionless = tmp_path / "collisionless.xml"
    collisionless.write_text(
        '<mujoco><worldbody><body name="backdrop"><inertial mass="1" diaginertia="1 1 1"/>'
        '<geom type="mesh" contype="0" conaffinity="0"/></body></worldbody></mujoco>',
        encoding="utf-8",
    )
    assert _mjcf_meshes_are_explicitly_noncolliding(str(collisionless))
    assert isinstance(_mjcf_entity_material(str(collisionless), fixed=True), gs.materials.Kinematic)
    assert isinstance(_mjcf_entity_material(str(collisionless), fixed=False), gs.materials.Rigid)

    colliding = tmp_path / "colliding.xml"
    colliding.write_text(
        '<mujoco><worldbody><body name="object"><inertial mass="1" diaginertia="1 1 1"/>'
        '<freejoint/><geom type="mesh" contype="1" conaffinity="0"/></body></worldbody></mujoco>',
        encoding="utf-8",
    )
    assert not _mjcf_meshes_are_explicitly_noncolliding(str(colliding))
    assert isinstance(_mjcf_entity_material(str(colliding), fixed=True), gs.materials.Rigid)

    inherited = tmp_path / "inherited.xml"
    inherited.write_text('<mujoco><worldbody><geom type="mesh"/></worldbody></mujoco>', encoding="utf-8")
    assert not _mjcf_meshes_are_explicitly_noncolliding(str(inherited))


def test_geniesim_g1_exports_authored_usd_collision_shapes(tmp_path):
    pytest.importorskip("pxr")
    import trimesh
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from tools.icl.prepare_geniesim_g1 import _usd_link_collision_elements

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/G01").GetPrim()
    UsdGeom.Xform.Define(stage, "/G01/body_link2")
    cube = UsdGeom.Cube.Define(stage, "/G01/body_link2/collisions/box")
    cube.CreateSizeAttr(0.2)
    UsdGeom.Xformable(cube.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.03, 0.0, 0.02))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    mesh = UsdGeom.Mesh.Define(stage, "/G01/body_link2/collisions/hull")
    mesh.CreatePointsAttr([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0), Gf.Vec3f(0, 0, 1)])
    mesh.CreateFaceVertexCountsAttr([3, 3, 3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3])
    UsdGeom.Xformable(mesh.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.5, 0.0, 0.0))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mesh_collision.CreateApproximationAttr().Set("convexHull")

    collisions, records = _usd_link_collision_elements(stage, root, "body_link2", tmp_path, {})

    assert [record["geometry_type"] for record in records] == ["box", "mesh"]
    assert all(collision.get("name") is None for collision in collisions)
    box_origin = collisions[0].find("origin")
    assert [float(value) for value in box_origin.get("xyz").split()] == pytest.approx([0.03, 0.0, 0.02])
    assert collisions[0].find("geometry/box").get("size") == "0.2 0.2 0.2"
    assert records[1]["source_approximation"] == "convexHull"
    assert records[1]["parts"][0]["watertight"] is True
    mesh_path = tmp_path / records[1]["parts"][0]["mesh"]
    exported = trimesh.load_mesh(mesh_path, process=True)
    np.testing.assert_allclose(exported.bounds[0], [0.5, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(exported.bounds[1], [1.5, 1.0, 1.0], atol=1e-6)


def test_geniesim_g2_camera_bundle_uses_source_resolution_and_evaluation_views(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, Usd, UsdGeom

    from roboverse_pack.tasks.icl.geniesim_scene import geniesim_g2_cameras

    source = tmp_path / "robot.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.Xform.Define(stage, "/genie")
    camera_specs = (
        ("head_link3", "head_front_Camera", (640, 400)),
        ("head_link3", "head_left_fisheye", (1920, 1536)),
        ("head_link3", "head_right_fisheye", (1920, 1536)),
        ("gripper_l_base_link", "Left_Camera", (1280, 1056)),
        ("gripper_r_base_link", "Right_Camera", (1280, 1056)),
    )
    for parent_name, camera_name, image_size in camera_specs:
        parent_path = f"/genie/{parent_name}"
        UsdGeom.Xform.Define(stage, parent_path)
        camera = UsdGeom.Camera.Define(stage, f"{parent_path}/{camera_name}")
        camera.CreateFocalLengthAttr(1.93)
        camera.CreateHorizontalApertureAttr(3.896)
        camera.CreateVerticalApertureAttr(2.453)
        projection = "opencvFisheye" if "fisheye" in camera_name else "opencvPinhole"
        camera.GetPrim().CreateAttribute(f"omni:lensdistortion:{projection}:imageSize", Sdf.ValueTypeNames.Int2).Set(
            Gf.Vec2i(*image_size)
        )
        if camera_name == "head_front_Camera":
            for key, value in {"fx": 306.7, "fy": 306.5, "cx": 319.9, "cy": 201.3}.items():
                camera.GetPrim().CreateAttribute(
                    f"omni:lensdistortion:opencvPinhole:{key}", Sdf.ValueTypeNames.Float
                ).Set(value)
    stage.GetRootLayer().Save()

    cameras = geniesim_g2_cameras(640, camera_profile="benchmark", source_robot_usd=source)
    assert [camera.name for camera in cameras] == ["head_rgb", "hand_left", "hand_right"]
    assert [(camera.width, camera.height) for camera in cameras] == [(640, 400), (640, 528), (640, 528)]
    assert cameras[0].focal_length == pytest.approx(3.896 * 306.7 / 640)
    np.testing.assert_allclose(cameras[0].intrinsic, [[306.7, 0, 319.9], [0, 306.5, 201.3], [0, 0, 1]], rtol=1e-6)
    scaled = geniesim_g2_cameras(320, camera_profile="hardware", source_robot_usd=source, fisheye_projection="pinhole")
    assert [camera.name for camera in scaled] == ["head_rgb", "head_fisheye_left", "head_fisheye_right"]
    assert [(camera.width, camera.height) for camera in scaled] == [(320, 200), (320, 256), (320, 256)]


@pytest.mark.general
def test_prepared_geniesim_g1_visuals_include_wrist_camera_housings_and_normals():
    root = Path(__file__).resolve().parents[1]
    asset_root = Path(os.environ.get("ROBOVERSE_DATA_DIR", root / "roboverse_data"))
    robot = asset_root / "icl/robots/geniesim_g1_omnipicker_closed_v5"
    provenance_path = robot / "provenance.json"
    if not provenance_path.is_file():
        pytest.skip("Prepare the pinned Genie Sim G1 Omnipicker v5 asset first")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    camera_visuals = provenance.get("wrist_camera_visuals", [])
    assert len(camera_visuals) == 4
    assert {record["link"] for record in camera_visuals} == {"gripper_l_base_link", "gripper_r_base_link"}
    assert all((robot / "meshes" / record["mesh"]).is_file() for record in camera_visuals)
    visual = robot / "meshes" / "base_link_visual_2.obj"
    assert visual.is_file()
    with visual.open(encoding="utf-8") as stream:
        lines = iter(stream)
        assert any(line.startswith("vn ") for line in lines)


@pytest.mark.examples
@pytest.mark.parametrize("simulator", ["mujoco", "genesis", "newton"])
def test_geniesim_g2_model_builds_on_selected_backends(tmp_path, simulator):
    requested = set(os.environ.get("ROBOVERSE_ICL_TEST_BACKENDS", "").split(","))
    if simulator != "mujoco" and simulator not in requested:
        pytest.skip(f"Set ROBOVERSE_ICL_TEST_BACKENDS to include {simulator}")
    if importlib.util.find_spec(simulator) is None:
        pytest.skip(f"{simulator} is not installed")
    root = Path(__file__).resolve().parents[1]
    asset = Path(
        os.environ.get(
            "ROBOVERSE_GENIESIM_G2_ASSET",
            root / "roboverse_data/icl/robots/geniesim_g2_omnipicker_v7",
        )
    )
    if not asset.joinpath("provenance.json").is_file():
        pytest.skip("Prepare the pinned Genie Sim G2 Omnipicker robot first")
    if simulator == "newton":
        import torch

        if not torch.cuda.is_available():
            pytest.skip("Newton integration test requires a CUDA GPU")
    output = tmp_path / "g2.png"
    result = subprocess.run(
        [
            sys.executable,
            "tools/icl/probe_agibot_g2.py",
            "--sim",
            simulator,
            "--asset",
            asset,
            "--width",
            "64",
            "--height",
            "64",
            "--output",
            str(output),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, "\n".join((result.stdout + result.stderr).splitlines()[-30:])
    from PIL import Image

    with Image.open(output) as image:
        assert image.size == (64, 64) and image.mode == "RGB"


def test_geniesim_revision_does_not_use_parent_workspace_git(tmp_path):
    import shutil

    from tools.icl.import_geniesim_specs import _revision
    from tools.icl.prepare_geniesim_g2 import _source_revision

    if shutil.which("git") is None:
        pytest.skip("Git is required for checkout provenance checks")
    repo = tmp_path / "checkout"
    repo.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.name", "RoboVerse Test")
    git("config", "user.email", "test@example.invalid")
    git("remote", "add", "origin", "https://github.com/AgibotTech/genie_sim.git")
    source = repo / "source/geniesim_benchmark/src/geniesim_benchmark"
    source.mkdir(parents=True)
    (source / "tracked.txt").write_text("source\n", encoding="utf-8")
    unrelated = repo / "vendor/unrelated/source/geniesim_benchmark/src/geniesim_benchmark"
    unrelated.mkdir(parents=True)
    git("add", ".")
    git("commit", "-qm", "fixture")
    assert _revision(unrelated) == "unknown"
    with pytest.raises(ValueError, match="independent robot-model Git checkout"):
        _source_revision(unrelated)
    assert _revision(source) == git("rev-parse", "HEAD")
    (source / "tracked.txt").write_text("modified\n", encoding="utf-8")
    assert _revision(source) == "unknown"


def test_geniesim_initial_pose_parser_never_executes_source(tmp_path, monkeypatch):
    from tools.icl.import_geniesim_specs import _source_initial_pose_snapshot

    source = tmp_path / "source"
    _write_fake_initial_pose_sources(source, ("pick_item",), monkeypatch)
    snapshot = _source_initial_pose_snapshot(source)
    assert len(snapshot["source_files"]) == 5
    assert snapshot["joint_positions"]["pick_item"]["G2_omnipicker"]["grip1"] == 0.785
    semantics = source / "utils/infer_pre_process.py"
    original_semantics = semantics.read_bytes()
    semantics.write_bytes(b"# changed reset semantics\n")
    with pytest.raises(ValueError, match="Unsupported Genie Sim reset semantics"):
        _source_initial_pose_snapshot(source)
    semantics.write_bytes(original_semantics)
    sentinel = tmp_path / "must_not_exist"
    (source / "benchmark/config/robot_init_states.py").write_text(
        f"TASK_INFO_DICT = __import__('pathlib').Path({str(sentinel)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported source expression"):
        _source_initial_pose_snapshot(source)
    assert not sentinel.exists()


def test_geniesim_catalog_historical_smoke_requires_matching_source_and_payload():
    from tools.icl.build_task_catalog import (
        _GENIESIM_VALIDATED_IMPORT_PAYLOAD_SHA256,
        _GENIESIM_VALIDATED_SOURCE_CONTENT_SHA256,
        _run_migration_status,
    )

    args = ("g2_op", "pick_object_type", "g2op_if_pick_object_type")
    assert _run_migration_status(*args, _GENIESIM_VALIDATED_SOURCE_CONTENT_SHA256, None) == "implemented_unverified"
    assert _run_migration_status(*args, "wrong-source", _GENIESIM_VALIDATED_IMPORT_PAYLOAD_SHA256) == (
        "implemented_unverified"
    )
    assert (
        _run_migration_status(
            *args, _GENIESIM_VALIDATED_SOURCE_CONTENT_SHA256, _GENIESIM_VALIDATED_IMPORT_PAYLOAD_SHA256
        )
        == "smoke_tested_with_asset_alias"
    )


def test_geniesim_g2_drive_provenance_correction_checks_all_inputs(tmp_path, monkeypatch):
    import hashlib

    from tools.icl import prepare_geniesim_g2 as converter

    source = tmp_path / "asset_source"
    drive_usd = source / "robot/G2_omnipicker/robot.usda"
    drive_usd.parent.mkdir(parents=True)
    drive_usd.write_bytes(b"fixture robot USD")
    (source / "LICENSE").write_bytes(b"fixture license")
    entries = []
    for relative in ("robot/G2_omnipicker/robot.usda", "LICENSE"):
        content = (source / relative).read_bytes()
        digest = hashlib.sha1(f"blob {len(content)}\0".encode() + content, usedforsecurity=False).hexdigest()
        entries.append({"rfilename": relative, "size": len(content), "blobId": digest})
    (source / "roboverse_source_manifest.json").write_text(
        json.dumps({
            "repo": "agibot-world/GenieSimAssets",
            "revision": converter.DRIVE_SOURCE_REVISION,
            "files": entries,
        }),
        encoding="utf-8",
    )
    output = tmp_path / "prepared"
    output.mkdir()
    (output / "model.urdf").write_bytes(b"prepared model")
    parameters = {
        name: {"stiffness": 1.0, "damping": 0.5, "effort_limit_sim": None} for name in converter.CONTROL_JOINTS
    }
    monkeypatch.setattr(converter, "_source_drive_parameters", lambda _: parameters)
    provenance = {
        "format": "roboverse.icl.geniesim_g2_robot.v2",
        "source_repository": converter.SOURCE_REPOSITORY,
        "source_revision": converter.SOURCE_REVISION,
        "robot_variant": converter.VARIANT,
        "file_sha256": {"model.urdf": hashlib.sha256(b"prepared model").hexdigest()},
        "drive_source": {
            "repository": converter.DRIVE_SOURCE_REPOSITORY,
            "revision": converter._LEGACY_INCORRECT_DRIVE_REVISION,
            "robot_usd": "robot/G2_omnipicker/robot.usda",
            "robot_usd_sha256": hashlib.sha256(drive_usd.read_bytes()).hexdigest(),
            "controlled_joints": sorted(converter.CONTROL_JOINTS),
            "parameters": parameters,
        },
    }
    target = output / "provenance.json"
    target.write_text(json.dumps(provenance), encoding="utf-8")
    result = converter.correct_drive_provenance(output=output, drive_usd=drive_usd)
    assert result["drive_source"]["revision"] == converter.DRIVE_SOURCE_REVISION
    assert (
        result["drive_source"]["revision_correction"]["previous_revision"] == converter._LEGACY_INCORRECT_DRIVE_REVISION
    )
    corrected_bytes = target.read_bytes()
    converter.correct_drive_provenance(output=output, drive_usd=drive_usd)
    assert target.read_bytes() == corrected_bytes
    (output / "model.urdf").write_bytes(b"modified model")
    with pytest.raises(ValueError, match="model checksum mismatch"):
        converter.correct_drive_provenance(output=output, drive_usd=drive_usd)
    assert target.read_bytes() == corrected_bytes


@pytest.mark.general
def test_geniesim_g2_robot_uses_pinned_source_drives_and_stable_inertias():
    root = Path(__file__).resolve().parents[1]
    asset = root / "roboverse_data/icl/robots/geniesim_g2_omnipicker_v7"
    if not asset.joinpath("provenance.json").is_file():
        pytest.skip("Prepare the v7 G2 model with source GenieSim joint drives first")
    from roboverse_pack.tasks.icl.geniesim_benchmark import G2_CONTROL_JOINTS, build_geniesim_g2_robot

    provenance = json.loads((asset / "provenance.json").read_text(encoding="utf-8"))
    robot = build_geniesim_g2_robot(asset)
    assert set(robot.actuators) == set(G2_CONTROL_JOINTS)
    assert provenance["drive_source"]["revision"] == "a0813aad7c16165daffbe6c2737754e0344809ee"
    arm = robot.actuators["idx21_arm_l_joint1"]
    assert arm.stiffness == pytest.approx(10000 * 180 / np.pi)
    assert arm.damping == pytest.approx(1000 * 180 / np.pi)
    assert arm.effort_limit_sim == pytest.approx(60.0)
    adjustments = provenance["model"]["inertia_floor_adjustments"]
    assert len(adjustments) >= 4
    assert all(min(adjustment["after"]) >= 1.1e-6 for adjustment in adjustments)


@pytest.mark.examples
@pytest.mark.parametrize("simulator", ["mujoco", "genesis", "newton"])
def test_geniesim_g2_benchmark_task_runs_checker_on_shared_backend_states(simulator):
    requested = set(os.environ.get("ROBOVERSE_ICL_TEST_BACKENDS", "").split(","))
    if simulator != "mujoco" and simulator not in requested:
        pytest.skip(f"Set ROBOVERSE_ICL_TEST_BACKENDS to include {simulator}")
    if importlib.util.find_spec(simulator) is None:
        pytest.skip(f"{simulator} is not installed")
    if simulator == "newton":
        import torch

        if not torch.cuda.is_available():
            pytest.skip("Newton integration test requires a CUDA GPU")

    root = Path(__file__).resolve().parents[1]
    spec_path = root / "roboverse_data/icl/geniesim_benchmark_specs_source_scene_v2/benchmark.json"
    assets_root = root / "roboverse_data/icl/objects/geniesim"
    robot_root = root / "roboverse_data/icl/robots/geniesim_g2_omnipicker_v11"
    if not spec_path.is_file() or not robot_root.joinpath("provenance.json").is_file():
        pytest.skip("Prepare the local Genie Sim specs and G2 Omnipicker robot first")

    run_id = "g2op_if_pick_billiards_color"
    from roboverse_pack.tasks.icl.geniesim_scene import _instance_scene_layout, _resolve_scene_asset, load_run_instance

    _, source_instance, _ = load_run_instance(spec_path, run_id, instance_index=0)
    layout = _instance_scene_layout(source_instance)
    entries_by_asset = {
        entry.get("usd"): entry for entry in layout.values() if isinstance(entry, dict) and entry.get("usd")
    }
    missing_assets = [
        name
        for name in entries_by_asset
        if not (
            assets_root / _resolve_scene_asset(assets_root, name, entries_by_asset[name])[0] / "provenance.json"
        ).is_file()
    ]
    if missing_assets:
        pytest.skip(f"Prepare the G2 scene assets for {run_id}: {missing_assets}")

    from roboverse_pack.tasks.icl.geniesim_benchmark import GenieSimBenchmarkTask

    task = GenieSimBenchmarkTask(
        run_id=run_id,
        simulator=simulator,
        spec_path=spec_path,
        assets_root=assets_root,
        cache_root=root / "roboverse_data/icl/cache/geniesim_scene_wrappers_v6",
        image_size=64,
        max_episode_steps=20,
    )
    try:
        observation, _ = task.reset()
        assert task.source_robot == "G2"
        assert task.action_space.shape == (24,)
        assert set(observation.cameras) == {"head_rgb", "head_fisheye_left", "head_fisheye_right"}
        for camera in observation.cameras.values():
            frame = camera.rgb[0].detach().cpu().numpy()
            assert frame.shape[-1] == 3 and frame.std() > 1.0
        target_positions = [task.scenario.robots[0].default_joint_positions[name] for name in task.control_joint_names]
        head_pitch_index = task.control_joint_names.index("idx13_head_joint3")
        target_positions[head_pitch_index] = 0.2
        initial = (
            task.handler
            .get_states(mode="tensor")
            .robots[task.robot_name]
            .joint_pos[0, task._control_indices]
            .detach()
            .cpu()
            .numpy()
        )
        for _ in range(10):
            _, _, terminated, _, info = task.step(target_positions)
            assert terminated.shape == (1,)
        assert info["geniesim_run_id"] == run_id
        assert info["geniesim_action_spec"] == "g2_control_joint_position_target_v1"
        assert info["geniesim_checker_status"] in {"running", "success", "failure"}
        final = (
            task.handler
            .get_states(mode="tensor")
            .robots[task.robot_name]
            .joint_pos[0, task._control_indices]
            .detach()
            .cpu()
            .numpy()
        )
        initial_error = abs(initial[head_pitch_index] - target_positions[head_pitch_index])
        final_error = abs(final[head_pitch_index] - target_positions[head_pitch_index])
        assert initial_error > 0.01 and final_error < 0.3 * initial_error
    finally:
        task.close()


@pytest.mark.examples
@pytest.mark.parametrize("simulator", ["mujoco", "genesis", "newton"])
@pytest.mark.parametrize("run_id", GENIESIM_G2_RUN_IDS)
def test_geniesim_g2_catalog_run_builds_and_steps_on_backend(simulator, run_id):
    requested = set(os.environ.get("ROBOVERSE_ICL_TEST_BACKENDS", "").split(","))
    if simulator != "mujoco" and simulator not in requested:
        pytest.skip(f"Set ROBOVERSE_ICL_TEST_BACKENDS to include {simulator}")
    if importlib.util.find_spec(simulator) is None:
        pytest.skip(f"{simulator} is not installed")
    if simulator == "newton":
        import torch

        if not torch.cuda.is_available():
            pytest.skip("Newton integration test requires a CUDA GPU")

    root = Path(__file__).resolve().parents[1]
    spec_path = root / "roboverse_data/icl/geniesim_benchmark_specs_source_scene_v2/benchmark.json"
    assets_root = root / "roboverse_data/icl/objects/geniesim"
    robot_root = root / "roboverse_data/icl/robots/geniesim_g2_omnipicker_v11"
    if not spec_path.is_file() or not robot_root.joinpath("provenance.json").is_file():
        pytest.skip("Prepare the imported Genie Sim specs and pinned G2 Omnipicker v11 model first")
    from roboverse_pack.tasks.icl.geniesim_scene import _instance_scene_layout, _resolve_scene_asset, load_run_instance

    _, source_instance, _ = load_run_instance(spec_path, run_id, instance_index=0)
    layout = _instance_scene_layout(source_instance)
    entries_by_asset = {
        entry.get("usd"): entry for entry in layout.values() if isinstance(entry, dict) and entry.get("usd")
    }
    missing_assets = [
        name
        for name in entries_by_asset
        if not (
            assets_root / _resolve_scene_asset(assets_root, name, entries_by_asset[name])[0] / "provenance.json"
        ).is_file()
    ]
    if missing_assets:
        pytest.skip(f"Prepare missing source assets for {run_id}: {missing_assets}")

    from roboverse_pack.tasks.icl.geniesim_benchmark import GenieSimBenchmarkTask

    task = GenieSimBenchmarkTask(
        run_id=run_id,
        simulator=simulator,
        spec_path=spec_path,
        assets_root=assets_root,
        cache_root=root / "roboverse_data/icl/cache/geniesim_scene_wrappers_v6",
        image_size=48,
        max_episode_steps=2,
    )
    try:
        observation, _ = task.reset()
        assert task.source_robot == "G2"
        assert task.action_space.shape == (24,)
        assert set(observation.cameras) == {"head_rgb", "head_fisheye_left", "head_fisheye_right"}
        assert all(np.isfinite(camera.rgb[0].detach().cpu().numpy()).all() for camera in observation.cameras.values())
        targets = [task.scenario.robots[0].default_joint_positions[name] for name in task.control_joint_names]
        _, _, terminated, _, info = task.step(targets)
        assert terminated.shape == (1,)
        assert info["geniesim_run_id"] == run_id
        assert info["geniesim_checker_status"] in {"running", "success", "failure"}
    finally:
        task.close()


@pytest.mark.examples
def test_geniesim_g2_spatial_task_accepts_synthicl_cartesian_actions():
    if importlib.util.find_spec("mujoco") is None:
        pytest.skip("mujoco is not installed")
    root = Path(__file__).resolve().parents[1]
    spec_path = root / "roboverse_data/icl/geniesim_benchmark_specs_source_scene_v2/benchmark.json"
    assets_root = root / "roboverse_data/icl/objects/geniesim"
    robot_root = root / "roboverse_data/icl/robots/geniesim_g2_omnipicker_v11"
    if not spec_path.is_file() or not robot_root.joinpath("provenance.json").is_file():
        pytest.skip("Prepare the local Genie Sim specs and G2 Omnipicker robot first")
    run_id = "g2op_spatial_stack_bowls"
    from roboverse_pack.tasks.icl.geniesim_scene import _instance_scene_layout, load_run_instance

    _, source_instance, _ = load_run_instance(spec_path, run_id, instance_index=0)
    if any(
        not (assets_root / entry["usd"] / "provenance.json").is_file()
        for entry in _instance_scene_layout(source_instance).values()
    ):
        pytest.skip(f"Prepare the G2 scene assets for {run_id}")

    import torch

    from roboverse_learn.il.icl.data import ACTION_SPEC
    from roboverse_pack.tasks.icl.geniesim_benchmark import GenieSimBenchmarkTask

    task = GenieSimBenchmarkTask(
        run_id=run_id,
        simulator="mujoco",
        spec_path=spec_path,
        assets_root=assets_root,
        cache_root=root / "roboverse_data/icl/cache/geniesim_scene_wrappers_v6",
        robot_root=robot_root,
        image_size=64,
        action_mode="tcp_delta",
        max_episode_steps=2,
    )
    try:
        observation, _ = task.reset()
        assert task.source_robot == "G2"
        assert task.action_space.shape == (7,)
        assert set(observation.cameras) == {"head_rgb", "head_fisheye_left", "head_fisheye_right"}
        action = torch.tensor([[0.001, 0, 0, 0, 0, 0, 1]], dtype=torch.float32)
        _, _, _, _, info = task.step(action)
        assert info["geniesim_action_spec"] == ACTION_SPEC
        assert info["geniesim_checker_status"] in {"running", "success", "failure"}
    finally:
        task.close()


def test_geniesim_usd_material_subsets_preserve_faces_and_colors():
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    from tools.icl.prepare_geniesim_g1 import _visual_mesh

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Scope.Define(stage, "/visuals")
    UsdGeom.Xform.Define(stage, "/visuals/head_link2")
    mesh = UsdGeom.Mesh.Define(stage, "/visuals/head_link2/mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
    material = UsdShade.Material.Define(stage, "/black")
    shader = UsdShade.Shader.Define(stage, "/black/shader")
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 0, 0))
    material2 = UsdShade.Material.Define(stage, "/white")
    shader2 = UsdShade.Shader.Define(stage, "/white/shader")
    shader2.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 1, 1))
    for index, selected in enumerate((material, material2)):
        subset = UsdGeom.Subset.Define(stage, f"/visuals/head_link2/mesh/part_{index}")
        subset.CreateElementTypeAttr("face")
        subset.CreateIndicesAttr([index])
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(selected)
    combined, parts = _visual_mesh(stage, "head_link2")
    assert len(combined.faces) == 2
    assert {part[0] for part in parts} == {(0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0)}
    assert all(len(part.faces) == 1 for _, part in parts)
    assert all(np.isfinite(part.vertex_normals).all() for _, part in parts)

    def triangle_geometry(mesh):
        return sorted(
            tuple(sorted(tuple(np.round(vertex, 7)) for vertex in mesh.vertices[face])) for face in mesh.faces
        )

    assert sorted(triangle for _, part in parts for triangle in triangle_geometry(part)) == triangle_geometry(combined)


def test_geniesim_rigid_object_converter_preserves_uv_multimaterial_mesh_and_source_hashes(tmp_path):
    pytest.importorskip("pxr")
    import xml.etree.ElementTree as ET

    from PIL import Image
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    from tools.icl.prepare_geniesim_g1 import REVISION
    from tools.icl.prepare_geniesim_objects import _sha256, prepare_object

    source = tmp_path / "source"
    asset = source / "objects/benchmark/test/toy_object"
    (asset / "textures").mkdir(parents=True)
    Image.new("RGB", (8, 8), (210, 50, 30)).save(asset / "textures/color.png")
    (asset / "Aligned.usd").write_text("#usda 1.0\n", encoding="utf-8")
    (asset / "object_parameters.json").write_text('{"mass": 0.1}', encoding="utf-8")
    (asset / "item.py").write_text('{"has_joint": False, "has_articulation": False}', encoding="utf-8")
    stage = Usd.Stage.CreateNew(str(asset / "Aligned.usda"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/entity")
    UsdGeom.Xform.Define(stage, "/World/entity/body")
    mesh = UsdGeom.Mesh.Define(stage, "/World/entity/body/visual")
    mesh.CreatePointsAttr([(-1, -1, -1), (1, -1, -1), (0, 1, -1), (0, 0, 1)])
    mesh.CreateFaceVertexCountsAttr([3, 3, 3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3])
    uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    uv.Set([(0, 0), (1, 0), (0.5, 1)] * 4)
    material = UsdShade.Material.Define(stage, "/World/Looks/material")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/material/shader")
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.5, 0.4, 0.3))
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("textures/color.png"))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    solid_material = UsdShade.Material.Define(stage, "/World/Looks/solid_material")
    solid_shader = UsdShade.Shader.Define(stage, "/World/Looks/solid_material/shader")
    solid_shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 1, 0))
    solid_material.CreateSurfaceOutput().ConnectToSource(solid_shader.ConnectableAPI(), "surface")
    for subset_name, face_ids, bound_material in (
        ("textured_faces", [0, 1], material),
        ("solid_faces", [2, 3], solid_material),
    ):
        subset = UsdGeom.Subset.Define(stage, f"{mesh.GetPath()}/{subset_name}")
        subset.CreateElementTypeAttr(UsdGeom.Tokens.face)
        subset.CreateIndicesAttr(face_ids)
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(bound_material)
    stage.GetRootLayer().Save()
    entries = []
    for path in (
        asset / "Aligned.usda",
        asset / "Aligned.usd",
        asset / "object_parameters.json",
        asset / "item.py",
        asset / "textures/color.png",
    ):
        rel = path.relative_to(source).as_posix()
        entries.append({"rfilename": rel, "size": path.stat().st_size, "lfs": {"sha256": _sha256(path)}})
    (source / "roboverse_source_manifest.json").write_text(
        json.dumps({"repo": "agibot-world/GenieSimAssets", "revision": REVISION, "files": entries}),
        encoding="utf-8",
    )

    output = tmp_path / "prepared" / "toy_object"
    provenance = prepare_object(source, output, "toy_object")
    textured_obj = (output / "visual_0.obj").read_text(encoding="utf-8")
    solid_obj = (output / "visual_1.obj").read_text(encoding="utf-8")
    assert provenance["asset_id"] == "toy_object"
    assert provenance["visual_mesh_count"] == 2
    assert provenance["urdf_textured_material_tint"] == "neutral_white"
    assert provenance["source_files_sha256"]["Aligned.usda"] == _sha256(asset / "Aligned.usda")
    assert "map_Kd material_0.png" in (output / "material_0.mtl").read_text(encoding="utf-8")
    assert "Kd 0.0 1.0 0.0" in (output / "material_1.mtl").read_text(encoding="utf-8")
    generated_xml = ET.parse(output / "object.xml").getroot()
    np.testing.assert_allclose(
        [float(value) for value in generated_xml.find("asset/material").get("rgba").split()],
        [0.5, 0.4, 0.3, 1],
    )
    generated_urdf = ET.parse(output / "object.urdf").getroot()
    np.testing.assert_allclose(
        [float(value) for value in generated_urdf.find(".//visual[1]/material/color").get("rgba").split()],
        [1, 1, 1, 1],
    )
    np.testing.assert_allclose(
        [float(value) for value in generated_urdf.find(".//visual[2]/material/color").get("rgba").split()],
        [0, 1, 0, 1],
    )

    generated_urdf.find(".//visual[1]/material/color").set("rgba", "0.5 0.4 0.3 1")
    ET.ElementTree(generated_urdf).write(output / "object.urdf", encoding="utf-8", xml_declaration=True)
    from tools.icl.prepare_geniesim_objects import _upgrade_prepared_object

    _upgrade_prepared_object(source, output, "toy_object")
    upgraded_provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert upgraded_provenance["urdf_textured_material_tint"] == "neutral_white"
    np.testing.assert_allclose(
        [
            float(value)
            for value in ET
            .parse(output / "object.urdf")
            .getroot()
            .find(".//visual[1]/material/color")
            .get("rgba")
            .split()
        ],
        [1, 1, 1, 1],
    )
    assert sum(line.startswith("vt ") for line in textured_obj.splitlines()) == 12
    assert sum(line.startswith("vt ") for line in solid_obj.splitlines()) == 12
    assert sum(line.startswith("f ") for line in textured_obj.splitlines()) == 2
    assert sum(line.startswith("f ") for line in solid_obj.splitlines()) == 2
    assert not ET.parse(output / "object.xml").getroot().findall(".//freejoint")
    textured_visual = ET.parse(output / "object.urdf").getroot().find(".//visual[1]/material/texture")
    assert textured_visual is not None and textured_visual.get("filename") == "material_0.png"
    provenance_path = output / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    # Upgrade an existing v2 conversion whose URDF material was tinted by the
    # texture image's average RGB and then multiplied by the texture again.
    import shutil

    legacy_v2 = tmp_path / "prepared_v2" / "toy_object"
    shutil.copytree(output, legacy_v2)
    legacy_provenance_path = legacy_v2 / "provenance.json"
    legacy_provenance = json.loads(legacy_provenance_path.read_text(encoding="utf-8"))
    legacy_provenance["format"] = "roboverse.icl.geniesim_rigid_object.v2"
    legacy_provenance_path.write_text(json.dumps(legacy_provenance), encoding="utf-8")
    legacy_color = " ".join(str(value) for value in (210 / 255, 50 / 255, 30 / 255, 1))
    legacy_urdf_path = legacy_v2 / "object.urdf"
    legacy_urdf = ET.parse(legacy_urdf_path)
    legacy_urdf.getroot().find(".//visual[1]/material/color").set(
        "rgba", " ".join(str(value) for value in (210 / 255, 50 / 255, 30 / 255, 1))
    )
    legacy_urdf.write(legacy_urdf_path, encoding="utf-8", xml_declaration=True)
    legacy_mjcf_path = legacy_v2 / "object.xml"
    legacy_mjcf = ET.parse(legacy_mjcf_path)
    legacy_mjcf.getroot().find("asset/material[@name='material_0']").set("rgba", legacy_color)
    legacy_mjcf.write(legacy_mjcf_path, encoding="utf-8", xml_declaration=True)
    from tools.icl.prepare_geniesim_objects import _upgrade_prepared_object

    _upgrade_prepared_object(source, legacy_v2, "toy_object")
    corrected_urdf_color = ET.parse(legacy_urdf_path).getroot().find(".//visual[1]/material/color").get("rgba")
    np.testing.assert_allclose([float(value) for value in corrected_urdf_color.split()], [1, 1, 1, 1])
    corrected_mjcf_color = ET.parse(legacy_mjcf_path).getroot().find("asset/material[@name='material_0']").get("rgba")
    np.testing.assert_allclose([float(value) for value in corrected_mjcf_color.split()], [0.5, 0.4, 0.3, 1])
    assert json.loads(legacy_provenance_path.read_text(encoding="utf-8"))["format"] == (
        "roboverse.icl.geniesim_rigid_object.v3"
    )

    provenance["format"] = "roboverse.icl.geniesim_rigid_object.v1"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    urdf_path = output / "object.urdf"
    old_urdf = ET.parse(urdf_path)
    for material_node in old_urdf.getroot().findall(".//visual/material"):
        for texture_node in material_node.findall("texture"):
            material_node.remove(texture_node)
    old_urdf.write(urdf_path, encoding="utf-8", xml_declaration=True)
    from tools.icl.prepare_geniesim_objects import _upgrade_prepared_object

    _upgrade_prepared_object(source, output, "toy_object")
    assert json.loads(provenance_path.read_text(encoding="utf-8"))["format"] == (
        "roboverse.icl.geniesim_rigid_object.v3"
    )
    from roboverse_pack.tasks.icl.geniesim_scene import _instance_wrapper

    wrapper = _instance_wrapper(output.parent, "toy_object", "toy_instance", tmp_path / "instance_wrapper")
    wrapper_texture = ET.parse(wrapper / "object.urdf").getroot().find(".//visual[1]/material/texture")
    assert wrapper_texture is not None and Path(wrapper_texture.get("filename")).is_file()
    (asset / "item.py").write_text('{"has_joint": True, "has_articulation": True}', encoding="utf-8")
    manifest = json.loads((source / "roboverse_source_manifest.json").read_text(encoding="utf-8"))
    item_entry = next(entry for entry in manifest["files"] if entry["rfilename"].endswith("/item.py"))
    item_entry["size"] = (asset / "item.py").stat().st_size
    item_entry["lfs"]["sha256"] = _sha256(asset / "item.py")
    (source / "roboverse_source_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Articulated source asset"):
        prepare_object(source, tmp_path / "articulated", "toy_object")


def test_geniesim_object_lookup_prefers_benchmark_asset_over_background_duplicate(tmp_path):
    from tools.icl.prepare_geniesim_g1 import REVISION
    from tools.icl.prepare_geniesim_objects import _find_asset

    source = tmp_path / "source"
    source.mkdir()
    asset_id = "benchmark_beverage_bottle_029"
    entries = [
        f"background/common/beverage_bottle/{asset_id}/Aligned.usda",
        f"objects/benchmark/beverage_bottle/{asset_id}/Aligned.usda",
    ]
    (source / "roboverse_source_manifest.json").write_text(
        json.dumps({
            "repo": "agibot-world/GenieSimAssets",
            "revision": REVISION,
            "files": [{"rfilename": path} for path in entries],
        }),
        encoding="utf-8",
    )
    assert _find_asset(source, asset_id) == f"objects/benchmark/beverage_bottle/{asset_id}"


def test_geniesim_object_lookup_prefers_library_asset_over_scene_duplicate(tmp_path):
    from tools.icl.prepare_geniesim_g1 import REVISION
    from tools.icl.prepare_geniesim_objects import _find_asset

    source = tmp_path / "source"
    source.mkdir()
    asset_id = "genie_beverage_bottle_004"
    entries = [
        f"scenes/GDK/{asset_id}/Aligned.usda",
        f"objects/genie/beverage_bottle/{asset_id}/Aligned.usda",
    ]
    (source / "roboverse_source_manifest.json").write_text(
        json.dumps({
            "repo": "agibot-world/GenieSimAssets",
            "revision": REVISION,
            "files": [{"rfilename": path} for path in entries],
        }),
        encoding="utf-8",
    )
    assert _find_asset(source, asset_id) == f"objects/genie/beverage_bottle/{asset_id}"


def test_geniesim_g1_inertia_floor_is_explicit_and_minimal():
    from xml.etree import ElementTree as ET

    from tools.icl.prepare_geniesim_g1 import _MIN_BACKEND_INERTIA_EIGENVALUE, _sanitize_urdf_inertias

    root = ET.fromstring(
        '<robot name="test"><link name="tiny"><inertial><inertia '
        'ixx="1e-8" ixy="0" ixz="0" iyy="2e-8" iyz="0" izz="3e-8"/>'
        '</inertial></link><link name="stable"><inertial><inertia '
        'ixx="0.1" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.3"/>'
        "</inertial></link></robot>"
    )

    corrections = _sanitize_urdf_inertias(root)

    assert [item["link"] for item in corrections] == ["tiny"]
    corrected = root.find("link[@name='tiny']/inertial/inertia")
    assert corrected is not None
    moments = np.linalg.eigvalsh(
        np.asarray([
            [float(corrected.get("ixx")), 0, 0],
            [0, float(corrected.get("iyy")), 0],
            [0, 0, float(corrected.get("izz"))],
        ])
    )
    assert moments[0] == pytest.approx(_MIN_BACKEND_INERTIA_EIGENVALUE)


@pytest.mark.examples
@pytest.mark.parametrize("simulator", ["mujoco", "genesis", "newton"])
def test_geniesim_textured_rigid_object_builds_on_selected_backends(tmp_path, simulator):
    requested = set(os.environ.get("ROBOVERSE_ICL_TEST_BACKENDS", "").split(","))
    if simulator != "mujoco" and simulator not in requested:
        pytest.skip(f"Set ROBOVERSE_ICL_TEST_BACKENDS to include {simulator}")
    if importlib.util.find_spec(simulator) is None:
        pytest.skip(f"{simulator} is not installed")
    asset = os.environ.get("ROBOVERSE_GENIESIM_OBJECT_ASSET")
    if not asset:
        pytest.skip("Set ROBOVERSE_GENIESIM_OBJECT_ASSET")
    if simulator == "newton":
        import torch

        if not torch.cuda.is_available():
            pytest.skip("Newton integration test requires a CUDA GPU")
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "object.png"
    result = subprocess.run(
        [
            sys.executable,
            "tools/icl/probe_geniesim_object.py",
            "--sim",
            simulator,
            "--asset",
            asset,
            "--width",
            "64",
            "--height",
            "64",
            "--output",
            str(output),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, "\n".join((result.stdout + result.stderr).splitlines()[-30:])
    from PIL import Image

    with Image.open(output) as image:
        frame = np.asarray(image)
        assert image.size == (64, 64) and image.mode == "RGB"
        assert frame.std() > 5


def test_g1_visual_uses_usd_attachment_not_original_mesh_origin():
    pytest.importorskip("pxr")
    import xml.etree.ElementTree as ET

    from pxr import Usd, UsdGeom

    from tools.icl.prepare_geniesim_g1 import _set_origin, _usd_visual_origin

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/genie")
    UsdGeom.Xform.Define(stage, "/genie/body_link2")
    visual = UsdGeom.Xform.Define(stage, "/genie/body_link2/visuals")
    visual.AddTranslateOp().Set((0.1, -0.2, 0.3))
    visual.AddRotateZOp().Set(30.0)
    translation, rpy = _usd_visual_origin(stage, "body_link2")
    original = ET.fromstring('<visual><origin xyz="0 0 0" rpy="0 0 -1.57079632679"/></visual>')
    _set_origin(original, translation, rpy)
    origin = original.find("origin")
    np.testing.assert_allclose(np.fromstring(origin.get("xyz"), sep=" "), [0.1, -0.2, 0.3], atol=1e-8)
    np.testing.assert_allclose(
        Rotation.from_euler("xyz", np.fromstring(origin.get("rpy"), sep=" ")).as_matrix(),
        Rotation.from_euler("z", 30, degrees=True).as_matrix(),
        atol=1e-8,
    )

    # robot.usda's composed G1 root is /G01 and carries a base visual-only
    # lift; reading only the standalone geometry layer loses this transform.
    UsdGeom.Xform.Define(stage, "/G01")
    UsdGeom.Xform.Define(stage, "/G01/base_link")
    base_visual = UsdGeom.Xform.Define(stage, "/G01/base_link/visuals")
    base_visual.AddTranslateOp().Set((0.0, 0.0, 0.21113469724309647))
    translation, rpy = _usd_visual_origin(stage, "base_link", root_path="/G01")
    np.testing.assert_allclose(translation, [0.0, 0.0, 0.21113469724309647], atol=1e-8)
    np.testing.assert_allclose(rpy, [0.0, 0.0, 0.0], atol=1e-8)


def test_prepared_geniesim_apple_has_source_texture_and_uvs():
    source = os.environ.get("ROBOVERSE_GENIESIM_SOURCE")
    apple = os.environ.get("ROBOVERSE_GENIESIM_APPLE_ASSET")
    if not source or not apple:
        pytest.skip("Set ROBOVERSE_GENIESIM_SOURCE and ROBOVERSE_GENIESIM_APPLE_ASSET")
    from tools.icl.prepare_assets import _sha256
    from tools.icl.prepare_geniesim_apple import SOURCE
    from tools.icl.prepare_geniesim_g1 import _verify

    source, apple = Path(source), Path(apple)
    info = json.loads((apple / "provenance.json").read_text(encoding="utf-8"))
    original = _verify(source, f"{SOURCE}/textures/Color.PNG")
    assert info["origin_files_sha256"]["textures/Color.PNG"] == _sha256(original)
    assert info["source_files_sha256"]["material_0.png"] == _sha256(apple / "material_0.png")
    obj = (apple / "textured.obj").read_text(encoding="utf-8")
    assert "map_Kd material_0.png" in (apple / "material_0.mtl").read_text(encoding="utf-8")
    assert sum(line.startswith("vt ") for line in obj.splitlines()) == 3 * sum(
        line.startswith("f ") for line in obj.splitlines()
    )


def test_prepared_g1_attachment_matches_source_usd():
    pytest.importorskip("pxr")
    source = os.environ.get("ROBOVERSE_GENIESIM_SOURCE")
    asset = os.environ.get("ROBOVERSE_GENIESIM_G1_ASSET")
    if not source or not asset:
        pytest.skip("Set ROBOVERSE_GENIESIM_SOURCE and ROBOVERSE_GENIESIM_G1_ASSET")
    import xml.etree.ElementTree as ET

    from pxr import Usd

    from tools.icl.prepare_geniesim_g1 import _usd_visual_origin, _verify

    usd = _verify(Path(source), "robot/G1_120s/configuration/G1_120s_base.usd")
    stage = Usd.Stage.Open(str(usd))
    robot = ET.parse(Path(asset) / "G1_120s.urdf").getroot()
    for name in ("base_link", "body_link1", "body_link2", "head_link2", "arm_l_link1"):
        position, rpy = _usd_visual_origin(stage, name)
        visual = robot.find(f"link[@name='{name}']/visual")
        origin = visual.find("origin")
        np.testing.assert_allclose(np.fromstring(origin.get("xyz"), sep=" "), position, atol=1e-8)
        np.testing.assert_allclose(
            Rotation.from_euler("xyz", np.fromstring(origin.get("rpy"), sep=" ")).as_matrix(),
            Rotation.from_euler("xyz", rpy).as_matrix(),
            atol=1e-8,
        )


@pytest.mark.examples
@pytest.mark.parametrize("simulator", ["mujoco", "genesis", "newton"])
def test_geniesim_g1_tabletop_preview_applies_apple_pose(tmp_path, simulator):
    """The preview must set and verify the same apple pose on every backend."""
    requested = set(os.environ.get("ROBOVERSE_ICL_TEST_BACKENDS", "").split(","))
    if simulator != "mujoco" and simulator not in requested:
        pytest.skip(f"Set ROBOVERSE_ICL_TEST_BACKENDS to include {simulator}")
    if importlib.util.find_spec(simulator) is None:
        pytest.skip(f"{simulator} is not installed")
    asset = os.environ.get("ROBOVERSE_GENIESIM_G1_ASSET")
    apple = os.environ.get("ROBOVERSE_GENIESIM_APPLE_ASSET")
    if not asset or not apple:
        pytest.skip("Set ROBOVERSE_GENIESIM_G1_ASSET and ROBOVERSE_GENIESIM_APPLE_ASSET")
    if simulator == "newton":
        import torch

        if not torch.cuda.is_available():
            pytest.skip("Newton integration test requires a CUDA GPU")
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "overview.png"
    result = subprocess.run(
        [
            sys.executable,
            "tools/icl/probe_geniesim_g1.py",
            "--sim",
            simulator,
            "--asset",
            asset,
            "--apple",
            apple,
            "--image-size",
            "64",
            "--output",
            str(output),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, "\n".join((result.stdout + result.stderr).splitlines()[-25:])
    from PIL import Image

    with Image.open(output) as image:
        assert image.size == (64, 64) and image.mode == "RGB"
