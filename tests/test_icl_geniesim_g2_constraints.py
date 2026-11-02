"""Check the converted G2 gripper against its pinned source constraints."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "roboverse_data/icl/robots/geniesim_g2_omnipicker_v8"
ASSET_V10 = ROOT / "roboverse_data/icl/robots/geniesim_g2_omnipicker_v10"
ASSET_V11 = ROOT / "roboverse_data/icl/robots/geniesim_g2_omnipicker_v11"


@pytest.fixture
def converted():
    if not (ASSET / "provenance.json").exists():
        pytest.skip("Prepare the pinned G2 v8 asset first")
    provenance = json.loads((ASSET / "provenance.json").read_text(encoding="utf-8"))
    urdf = ET.parse(ASSET / "G2_crs_omnipicker.urdf").getroot()
    mjcf = ET.parse(ASSET / "G2_crs_omnipicker.xml").getroot()
    return provenance["gripper_constraint_source"], urdf, mjcf


def test_g2_gripper_constraints_and_limits_follow_source_usd(converted):
    source, urdf, mjcf = converted
    joints = {item.get("name"): item for item in urdf.findall("joint")}
    mj_joints = {item.get("name"): item for item in mjcf.findall(".//worldbody//joint")}
    assert len(source["layers_sha256"]) >= 2
    assert len(source["joints"]) == 16
    assert sum(item["mimic"] is not None for item in source["joints"].values()) == 2
    for name, spec in source["joints"].items():
        joint = joints[name]
        limit = joint.find("limit")
        np.testing.assert_allclose([float(limit.get("lower")), float(limit.get("upper"))], spec["limits_rad"])
        np.testing.assert_allclose(
            [float(value) for value in mj_joints[name].get("range").split()], spec["limits_rad"], atol=1e-6
        )
        assert float(mj_joints[name].get("armature")) == pytest.approx(spec["armature"])
        mimic = joint.find("mimic")
        if spec["mimic"] is None:
            assert mimic is None
        else:
            assert mimic.get("joint") == spec["mimic"]["joint"]
            assert float(mimic.get("multiplier")) == pytest.approx(spec["mimic"]["multiplier"])
    loops = {item.get("name"): item for item in urdf.findall("loop_joint")}
    for name, spec in source["loops"].items():
        loop = loops[name]
        for index in (0, 1):
            link = loop.find(f"link{index + 1}")
            assert link.get("link") == spec[f"body{index}"]
            np.testing.assert_allclose([float(x) for x in link.get("xyz").split()], spec[f"local_pos{index}"])


def test_g2_benchmark_collision_enables_follow_source_usd():
    """Disabled source chassis/arm colliders must not penetrate the ground."""
    if not (ASSET_V10 / "provenance.json").is_file():
        pytest.skip("Prepare the pinned G2 v10 asset first")
    pytest.importorskip("pxr")
    from tools.icl.prepare_geniesim_g2 import DEFAULT_DRIVE_USD, _source_collision_policy

    provenance = json.loads((ASSET_V10 / "provenance.json").read_text(encoding="utf-8"))
    source = _source_collision_policy(DEFAULT_DRIVE_USD)
    recorded = provenance["collision_source"]
    assert source["links"] == recorded["links"]
    assert source["layers_sha256"] == recorded["layers_sha256"]
    urdf = ET.parse(ASSET_V10 / "G2_crs_omnipicker.urdf").getroot()
    links = {link.get("name"): link for link in urdf.findall("link")}
    for name, spec in source["links"].items():
        if name in links and spec["enabled"] == 0:
            assert not links[name].findall("collision"), name
    for side in ("l", "r"):
        assert links[f"gripper_{side}_base_link"].find("collision") is not None
        for part in ("inner_link1", "inner_link2", "inner_link3", "inner_link4", "outer_link1", "outer_link2", "outer_link3", "outer_link4"):
            assert links[f"gripper_{side}_{part}"].find("collision") is not None


def test_g2_mixed_source_colliders_are_not_silently_discarded():
    from tools.icl.prepare_geniesim_g2 import _apply_source_collision_policy

    root = ET.fromstring('<robot><link name="link"><collision/></link></robot>')
    with pytest.raises(ValueError, match="Mixed enabled/disabled"):
        _apply_source_collision_policy(root, {"links": {"link": {"enabled": 1, "disabled": 1}}})


def test_g2_benchmark_drive_values_and_passive_drives_follow_robot_fix_usd():
    if not (ASSET_V10 / "provenance.json").is_file():
        pytest.skip("Prepare the pinned G2 v10 asset first")
    pytest.importorskip("pxr")
    from tools.icl.prepare_geniesim_g2 import DEFAULT_DRIVE_USD, _source_drive_parameters, _source_gripper_parameters

    source = _source_drive_parameters(DEFAULT_DRIVE_USD, benchmark=True)
    base_gripper = _source_gripper_parameters(DEFAULT_DRIVE_USD)
    benchmark_gripper = _source_gripper_parameters(DEFAULT_DRIVE_USD, benchmark=True)
    assert all(base_gripper[key] == benchmark_gripper[key] for key in ("joints", "loops"))
    provenance = json.loads((ASSET_V10 / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["drive_source"]["robot_usd"].endswith("robot_fix.usda")
    actuator_root = ET.parse(ASSET_V10 / "G2_crs_omnipicker.xml").getroot().find("actuator")
    actuators = {item.get("joint"): item for item in actuator_root}
    assert set(source) == set(actuators)
    for name in source:
        assert float(actuators[name].get("kp")) == pytest.approx(source[name]["stiffness"])
        assert float(actuators[name].get("kv")) == pytest.approx(source[name]["damping"])
    assert source["idx41_gripper_l_outer_joint1"]["stiffness"] == pytest.approx(100 * 180 / np.pi)


def test_g2_v11_control_limits_follow_benchmark_usd_and_reset():
    if not (ASSET_V11 / "provenance.json").is_file():
        pytest.skip("Prepare the pinned G2 v11 asset first")
    pytest.importorskip("pxr")
    mujoco = pytest.importorskip("mujoco")
    from roboverse_pack.tasks.icl.control import prepare_geniesim_reset_pose
    from roboverse_pack.tasks.icl.geniesim_benchmark import SPEC_PATH, build_geniesim_g2_robot
    from roboverse_pack.tasks.icl.geniesim_scene import load_run_instance
    from tools.icl.prepare_geniesim_g2 import DEFAULT_DRIVE_USD, _source_control_limits

    limits = _source_control_limits(DEFAULT_DRIVE_USD)
    provenance = json.loads((ASSET_V11 / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["control_limit_source"]["limits"] == limits
    robot = build_geniesim_g2_robot(ASSET_V11)
    model = mujoco.MjModel.from_xml_path(robot.mjcf_path)
    urdf = ET.parse(ASSET_V11 / "G2_crs_omnipicker.urdf").getroot()
    joints = {joint.get("name"): joint for joint in urdf.findall("joint")}
    assert len(limits) == 24
    for name, source in limits.items():
        limit = joints[name].find("limit")
        for attr, key in (("lower", "lower"), ("upper", "upper"), ("velocity", "velocity"), ("effort", "drive_force")):
            assert float(limit.get(attr)) == pytest.approx(source[key])
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        np.testing.assert_allclose(model.jnt_range[jid], [source["lower"], source["upper"]], rtol=1e-6)
        np.testing.assert_allclose(model.jnt_actfrcrange[jid], [-source["drive_force"], source["drive_force"]])
        np.testing.assert_allclose(model.actuator_forcerange[aid], [-source["drive_force"], source["drive_force"]])
        np.testing.assert_allclose(model.actuator_gear[aid], [1, 0, 0, 0, 0, 0])
        assert sum(model.actuator_trnid[:, 0] == jid) == 1
    run, _, _ = load_run_instance(SPEC_PATH, "g2op_if_pick_object_type")
    reset = prepare_geniesim_reset_pose(robot, run["source_initial_positions"])
    assert reset.max_equality_residual < 1e-4
    assert not reset.clipped_source_positions


@pytest.mark.parametrize("side", ["l", "r"])
@pytest.mark.parametrize("angle", [0.0, 0.2, 0.4, 0.6, 0.785])
def test_g2_source_gripper_loops_can_close_through_opening_range(converted, side, angle):
    mujoco = pytest.importorskip("mujoco")
    from scipy.optimize import least_squares

    model = mujoco.MjModel.from_xml_path(str(ASSET / "G2_crs_omnipicker.xml"))
    data = mujoco.MjData(model)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    names = [name for name in names if name and f"_gripper_{side}_" in name]
    leader = next(name for name in names if "_outer_joint1" in name)
    follower = next(name for name in names if "_inner_joint1" in name)
    data.qpos[model.jnt_qposadr[model.joint(leader).id]] = angle
    data.qpos[model.jnt_qposadr[model.joint(follower).id]] = -angle
    joint_ids = [model.joint(name).id for name in names if name not in (leader, follower)]
    qpos_ids = model.jnt_qposadr[joint_ids]
    site_ids = sorted(
        (
            i
            for i in range(model.nsite)
            if f"_gripper_{side}_" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) or "")
            and "joint2" in mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
        ),
        key=lambda i: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i),
    )
    assert len(site_ids) == 4

    def residual(values):
        data.qpos[qpos_ids] = values
        mujoco.mj_kinematics(model, data)
        return np.concatenate([data.site_xpos[site_ids[i]] - data.site_xpos[site_ids[i + 1]] for i in (0, 2)])

    solution = least_squares(
        residual, np.zeros(len(joint_ids)), bounds=(model.jnt_range[joint_ids, 0], model.jnt_range[joint_ids, 1])
    )
    assert np.max(np.linalg.norm(residual(solution.x).reshape(-1, 3), axis=1)) < 1e-4
