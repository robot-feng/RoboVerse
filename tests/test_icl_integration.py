"""Exercise physical ICL collection through the repository's simulator handlers."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.examples
def test_genesis_mjcf_handler_preserves_explicit_and_default_armature(tmp_path):
    """Genesis must use MJCF joint inertias rather than inject its own default."""
    if "genesis" not in set(os.environ.get("ROBOVERSE_ICL_TEST_BACKENDS", "").split(",")):
        pytest.skip("Set ROBOVERSE_ICL_TEST_BACKENDS to include genesis")
    if importlib.util.find_spec("genesis") is None:
        pytest.skip("Genesis is not installed")
    mjcf = tmp_path / "armature.xml"
    mjcf.write_text(
        '<mujoco model="armature_probe"><compiler angle="radian"/><worldbody><body name="base">'
        '<body name="explicit_link" pos=".1 0 0"><joint name="explicit" type="hinge" '
        'axis="0 1 0" armature=".123"/><geom type="sphere" size=".03" density="1000"/></body>'
        '<body name="implicit_link" pos="-.1 0 0"><joint name="implicit" type="hinge" '
        'axis="0 1 0"/><geom type="sphere" size=".03" density="1000"/></body>'
        '</body></worldbody></mujoco>',
        encoding="utf-8",
    )
    script = """
import json, sys
from metasim.scenario.robot import RobotCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils.setup_util import get_handler
robot = RobotCfg(name='test_robot', mjcf_path=sys.argv[1], num_joints=2,
                 fix_base_link=True, default_joint_positions={'explicit': 0., 'implicit': 0.})
robot.file_type['genesis'] = 'mjcf'
handler = get_handler(ScenarioCfg(robots=[robot], simulator='genesis', headless=True, add_default_ground=False))
try:
    entity = handler.object_inst_dict['test_robot']
    result = {}
    for name in ('explicit', 'implicit'):
        joint = next(joint for joint in entity.joints if joint.name == name)
        result[name] = float(entity.get_dofs_armature(dofs_idx_local=list(joint.dofs_idx_local)).reshape(-1)[0])
    print('ARMATURE_RESULT=' + json.dumps(result), flush=True)
finally:
    handler.close()
"""
    env = {**os.environ, "METASIM_GENESIS_BACKEND": "cpu"}
    process = subprocess.run(
        [sys.executable, "-c", script, str(mjcf)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert process.returncode == 0, process.stdout[-4000:] + process.stderr[-4000:]
    line = next(line for line in process.stdout.splitlines() if line.startswith("ARMATURE_RESULT="))
    result = json.loads(line.removeprefix("ARMATURE_RESULT="))
    assert result["explicit"] == pytest.approx(0.123)
    assert result["implicit"] == pytest.approx(0.0)


@pytest.mark.examples
@pytest.mark.requires_optional("mujoco", extra="icl,mujoco")
@pytest.mark.parametrize("simulator", ["mujoco", "genesis", "newton"])
def test_geniesim_fruit_scene_camera_and_joint_control(tmp_path, simulator):
    """Check mounted imagery and the physically coupled gripper on each backend."""
    requested = set(os.environ.get("ROBOVERSE_ICL_TEST_BACKENDS", "").split(","))
    if simulator != "mujoco" and simulator not in requested:
        pytest.skip(f"Set ROBOVERSE_ICL_TEST_BACKENDS to include {simulator}")
    if importlib.util.find_spec(simulator) is None:
        pytest.skip(f"{simulator} is not installed")
    if simulator == "newton":
        import torch

        if not torch.cuda.is_available():
            pytest.skip("Newton integration test requires a CUDA GPU")
    asset_root = Path(os.environ.get("ROBOVERSE_DATA_DIR", ROOT / "roboverse_data"))
    if (
        not (asset_root / "icl/scenes/fruit_sorting_1/scene.json").is_file()
        or not (asset_root / "icl/robots/geniesim_g1_omnipicker_closed_v5/provenance.json").is_file()
    ):
        pytest.skip("Prepare the Genie Sim fruit scene and closed-loop G1 asset v5 first")
    if os.environ.get("MUJOCO_GL", "").strip().lower() in ("disable", "disabled", "off", "false", "0"):
        pytest.skip("Rendering is disabled by MUJOCO_GL")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "METASIM_LOG_LEVEL": "WARNING"}
    if sys.platform.startswith("linux"):
        env["MUJOCO_GL"] = "egl"
    else:
        env.pop("MUJOCO_GL", None)
    result = subprocess.run(
        [
            sys.executable,
            "tools/icl/probe_fruit_task.py",
            "--sim",
            simulator,
            "--output",
            str(tmp_path),
            "--image-size",
            "64",
            "--steps",
            "60",
            "--delta-x",
            "0.01",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, "\n".join((result.stdout + result.stderr).splitlines()[-25:])
    report = json.loads((tmp_path / f"{simulator}_result.json").read_text(encoding="utf-8"))
    assert report["head_changed_pixels"] > 0
    camera_frames = report["camera_frames"]
    assert set(camera_frames) == {"head_rgb", "head_fisheye_left", "head_fisheye_right"}
    assert all(
        frame["shape"][0] > 0 and frame["shape"][1] > 0 and frame["shape"][2] == 3 and frame["pixel_std"] > 0
        for frame in camera_frames.values()
    )
    assert report["fisheye_projection"] == "pinhole_approximation"
    assert report["right_gripper_leader"] == pytest.approx(0.3, abs=0.05)
    assert report["right_gripper_leader"] + report["right_gripper_follower"] == pytest.approx(0, abs=0.02)
    assert np.linalg.norm(np.asarray(report["tcp_delta_m"]) - [0.01, 0, 0]) < 0.005
    assert not report["failure"]


@pytest.mark.examples
@pytest.mark.requires_optional("mujoco", "h5py", extra="icl,mujoco")
@pytest.mark.requires_asset("robots/franka/mjcf/panda.xml")
@pytest.mark.parametrize("simulator", ["mujoco", "genesis", "newton"])
def test_icl_pseudo_pair_has_aligned_states_and_shared_goal(tmp_path, simulator):
    """Exercise the generated pair, not just the transform helper."""
    import h5py

    requested = set(os.environ.get("ROBOVERSE_ICL_TEST_BACKENDS", "").split(","))
    if simulator != "mujoco" and simulator not in requested:
        pytest.skip(f"Set ROBOVERSE_ICL_TEST_BACKENDS to include {simulator}")
    if importlib.util.find_spec(simulator) is None:
        pytest.skip(f"{simulator} is not installed")
    if simulator == "newton":
        import torch

        if not torch.cuda.is_available():
            pytest.skip("Newton integration test requires a CUDA GPU")
    if os.environ.get("MUJOCO_GL", "").strip().lower() in ("disable", "disabled", "off", "false", "0"):
        pytest.skip("Rendering is disabled by MUJOCO_GL")
    asset_root = Path(os.environ.get("ROBOVERSE_DATA_DIR", ROOT / "roboverse_data"))
    if not (asset_root / "icl/objects/synthicl_tea_box_base2/provenance.json").is_file():
        pytest.skip("Prepare the SynthICL tea-box asset first")
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if sys.platform.startswith("linux"):
        env["MUJOCO_GL"] = "egl"
    else:
        env.pop("MUJOCO_GL", None)
    output = tmp_path / "episode_0.h5"
    result = subprocess.run(
        [
            sys.executable,
            "tools/icl/collect_pseudo.py",
            "--sim",
            simulator,
            "--image-size",
            "64",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, "\n".join((result.stdout + result.stderr).splitlines()[-25:])
    with h5py.File(output) as h5:
        metadata = json.loads(h5.attrs["metadata_json"])
        assert metadata["demonstration_kind"] == "pseudo_kinematic"
        assert metadata["physics_simulated"] is False
        first, second = metadata["trajectories"]
        assert first["source_pose"] != second["source_pose"]
        np.testing.assert_allclose(first["destination_from_placed_source"], second["destination_from_placed_source"])
        for index in range(2):
            group = h5[f"trajectory_{index}"]
            state, action = group["robot_states"][:], group["actions"][:]
            points, assigned = group["waypoints_idx"][:], group["assigned_wps"][:]
            assert len(state) == len(action) == len(assigned) == len(group["object_poses"])
            assert len(points) == 6 and points[0] == 0 and points[-1] == len(state) - 1
            assert np.all(np.diff(points) > 0) and np.all((assigned >= 1) & (assigned <= 5))
            np.testing.assert_allclose(state[1:, :3], state[:-1, :3] + action[:-1, :3], atol=0.0011, rtol=0)
            predicted = Rotation.from_rotvec(action[:-1, 3:6]) * Rotation.from_rotvec(state[:-1, 3:6])
            error = (predicted.inv() * Rotation.from_rotvec(state[1:, 3:6])).magnitude()
            assert error.max() < 0.002
            assert metadata["trajectories"][index]["position_error_m"] < 0.004
            for key in ("images_front", "images_side", "images_wrist"):
                images = group[key]
                assert images.shape == (len(state), 64, 64, 3)
                assert np.any(images[0] != images[-1])


@pytest.mark.examples
@pytest.mark.requires_optional("mujoco", "av", extra="icl,mujoco")
@pytest.mark.requires_asset("robots/franka/mjcf/panda.xml")
@pytest.mark.parametrize("simulator", ["mujoco", "genesis", "newton"])
def test_icl_object_choice_keeps_query_fixed_while_switching_target(tmp_path, simulator):
    """Both targets must be achievable from the same observed initial scene."""
    requested = set(os.environ.get("ROBOVERSE_ICL_TEST_BACKENDS", "").split(","))
    if simulator != "mujoco" and simulator not in requested:
        pytest.skip(f"Set ROBOVERSE_ICL_TEST_BACKENDS to include {simulator}")
    if importlib.util.find_spec(simulator) is None:
        pytest.skip(f"{simulator} is not installed")
    if simulator == "newton":
        import torch

        if not torch.cuda.is_available():
            pytest.skip("Newton integration test requires a CUDA GPU")
    if os.environ.get("MUJOCO_GL", "").strip().lower() in ("disable", "disabled", "off", "false", "0"):
        pytest.skip("Rendering is disabled by MUJOCO_GL")
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if sys.platform.startswith("linux"):
        env["MUJOCO_GL"] = "egl"
    else:
        env.pop("MUJOCO_GL", None)
    results = []
    for target in ("red", "blue"):
        process = subprocess.run(
            [
                sys.executable,
                "tools/icl/probe_object_choice.py",
                "--sim",
                simulator,
                "--target",
                target,
                "--layout",
                "0",
                "--image-size",
                "64",
                "--output",
                str(tmp_path),
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        tail = "\n".join((process.stdout + process.stderr).splitlines()[-25:])
        assert process.returncode == 0, f"{simulator} {target} probe failed:\n{tail}"
        results.append(json.loads((tmp_path / f"{simulator}_{target}_layout0.json").read_text(encoding="utf-8")))
    assert all(result["success"] for result in results)
    for name in ("red", "blue"):
        np.testing.assert_allclose(results[0]["initial"][name], results[1]["initial"][name], atol=1e-6, rtol=0)
    if simulator == "mujoco":
        assert results[0]["initial_rgb_sha256"] == results[1]["initial_rgb_sha256"]


@pytest.mark.examples
@pytest.mark.requires_optional("mujoco", "h5py", extra="icl,mujoco")
@pytest.mark.requires_asset("robots/franka/mjcf/panda.xml")
@pytest.mark.parametrize(
    ("simulator", "task", "object_id", "goal"),
    [
        ("mujoco", "push", "008_pudding_box", "left"),
        ("mujoco", "pick_place", "061_foam_brick", "right"),
        ("genesis", "pick_place", "061_foam_brick", "left"),
        ("newton", "pick_place", "061_foam_brick", "left"),
    ],
)
def test_icl_collects_a_physical_episode(tmp_path, simulator, task, object_id, goal):
    """Keep GPU-backed engines opt-in while exercising MuJoCo in the examples lane."""
    requested = set(os.environ.get("ROBOVERSE_ICL_TEST_BACKENDS", "").split(","))
    if simulator != "mujoco" and simulator not in requested:
        pytest.skip(f"Set ROBOVERSE_ICL_TEST_BACKENDS to include {simulator} to run its integration test")
    module = {"mujoco": "mujoco", "genesis": "genesis", "newton": "newton"}[simulator]
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"{simulator} is not installed")
    if simulator == "newton":
        import torch

        if not torch.cuda.is_available():
            pytest.skip("Newton integration test requires a CUDA GPU")
    if os.environ.get("MUJOCO_GL", "").strip().lower() in ("disable", "disabled", "off", "false", "0"):
        pytest.skip("Rendering is disabled by MUJOCO_GL")
    asset_root = Path(os.environ.get("ROBOVERSE_DATA_DIR", ROOT / "roboverse_data"))
    for filename in ("object.xml", "object.urdf", "provenance.json"):
        if not (asset_root / "icl" / "objects" / object_id / filename).is_file():
            pytest.skip(f"Prepare the {object_id} ICL asset with tools/icl/prepare_assets.py")
    command = [
        sys.executable,
        "tools/icl/collect.py",
        "collect",
        "--task",
        task,
        "--sim",
        simulator,
        "--goal",
        goal,
        "--object-id",
        object_id,
        "--split",
        "train",
        "--seed",
        "13",
        "--image-size",
        "64",
        "--output",
        str(tmp_path),
    ]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if sys.platform.startswith("linux"):
        env["MUJOCO_GL"] = "egl"
    else:
        env.pop("MUJOCO_GL", None)
    process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600, check=False)
    tail = "\n".join((process.stdout + process.stderr).splitlines()[-25:])
    assert process.returncode == 0, f"{simulator} {task} collection failed:\n{tail}"
    results = list((tmp_path / "raw").glob("*.json"))
    assert len(results) == 1 and json.loads(results[0].read_text())["success"] is True
    records = list((tmp_path / "raw").glob("*.npz"))
    assert len(records) == 1 and len(list((tmp_path / "raw").glob("*.hdf5"))) == 1
    with np.load(records[0], allow_pickle=False) as record:
        roots = record["objects/object/root_state"][:, 0]
        if task == "pick_place":
            names = json.loads(str(record["header"]))["joint_names"]["franka"]
            indices = [
                next(i for i, name in enumerate(names) if name.endswith(f"panda_finger_joint{j}")) for j in (1, 2)
            ]
            targets = record["actions"][:, 0, indices]
            assert np.any((roots[1:, 2] > 0.10) & (targets < 0.01).all(axis=1))
            assert (targets[-1] > 0.03).all()
