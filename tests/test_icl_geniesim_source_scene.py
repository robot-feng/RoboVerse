"""Source-scene payload, visibility and transform reconciliation checks."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


def test_scene_snapshot_records_authored_entity_physics_and_transform_overrides(tmp_path):
    pytest.importorskip("pxr")
    from tools.icl.import_geniesim_specs import _scene_snapshot_from_usd

    scene = tmp_path / "scene.usda"
    scene.write_text(
        '''#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def "World" {
    def "Objects" {
        def Xform "pot" (prepend payload = @/geniesim_assets/objects/benchmark/pot/pot_001/Aligned.usda@) {
            double3 xformOp:translate = (0, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            over "entity" {
                float physics:mass = 1
                quatf xformOp:orient = (0.7071, 0.7071, 0, 0)
                double3 xformOp:translate = (0.1, 0.2, 0.3)
                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
            }
        }
    }
}
''',
        encoding="utf-8",
    )
    snapshot = _scene_snapshot_from_usd(scene)
    overrides = snapshot["objects"]["pot"]["scene_authored_overrides"]
    attrs = {(item["prim"], item["attribute"]): item["value"] for item in overrides["attributes"]}
    assert attrs[("entity", "physics:mass")] == 1.0
    np.testing.assert_allclose(attrs[("entity", "xformOp:translate")], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(attrs[("entity", "xformOp:orient")], [0.7071, 0.7071, 0.0, 0.0], atol=1e-6)
    assert (".", "xformOp:translate") not in attrs  # parent world pose is already recorded separately


def test_asset_usd_physics_reads_rigid_mass_and_bound_material(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdPhysics

    from tools.icl.prepare_geniesim_objects import _single_usd_rigid_physics, _source_rigid_physics

    source = tmp_path / "object.usda"
    source.write_text(
        '''#usda 1.0
def Xform "World" {
    def Xform "entity" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
        float physics:mass = 0.1
        def Cube "collision" (prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"]) {
            rel material:binding:physics = </World/entity/PhysicsMaterial>
        }
        def Material "PhysicsMaterial" (prepend apiSchemas = ["PhysicsMaterialAPI"]) {
            float physics:staticFriction = 0.5
            float physics:dynamicFriction = 0.3
        }
    }
}
''',
        encoding="utf-8",
    )
    stage = Usd.Stage.Open(str(source))
    physics = _source_rigid_physics(stage)
    body = physics["rigid_bodies"][0]
    assert body["prim_path"] == "/World/entity"
    assert body["authored_mass_kg"] == pytest.approx(0.1)
    collider = body["colliders"][0]
    assert collider["material_path"] == "/World/entity/PhysicsMaterial"
    assert collider["static_friction"] == pytest.approx(0.5)
    assert collider["dynamic_friction"] == pytest.approx(0.3)
    UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath("/World")).CreateMassAttr(0.2)
    body = _source_rigid_physics(stage)["rigid_bodies"][0]
    assert body["authored_mass_kg"] == pytest.approx(0.2)
    assert body["mass_authored_prim_path"] == "/World"
    UsdPhysics.MassAPI(stage.GetPrimAtPath("/World/entity")).CreateDiagonalInertiaAttr(Gf.Vec3f(0.01, 0.02, 0.03))
    with pytest.raises(ValueError, match="authored inertial properties"):
        _single_usd_rigid_physics(stage, tmp_path, "object")


def test_usd_friction_proxy_is_authored_on_mjcf_without_changing_urdf_mass(tmp_path):
    import xml.etree.ElementTree as ET

    from tools.icl.prepare_geniesim_objects import _write_wrappers

    _write_wrappers(tmp_path, "block", 0.1, np.asarray([[-0.02] * 3, [0.02] * 3]), [], friction_proxy=0.3)
    mjcf = ET.parse(tmp_path / "object.xml").getroot()
    urdf = ET.parse(tmp_path / "object.urdf").getroot()
    assert float(mjcf.find("worldbody/body/inertial").get("mass")) == pytest.approx(0.1)
    assert float(mjcf.find("worldbody/body/geom[@name='collision']").get("friction").split()[0]) == pytest.approx(0.3)
    assert float(urdf.find("link/inertial/mass").get("value")) == pytest.approx(0.1)


def test_usd_physics_preparation_creates_separate_version_with_verified_source_values(tmp_path):
    pytest.importorskip("pxr")
    import xml.etree.ElementTree as ET

    from tools.icl.prepare_geniesim_objects import prepare_object

    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / "datasets/icl/sources/geniesim-assets"
    layer = source / "objects/benchmark/building_blocks/benchmark_building_blocks_073/Aligned.usda"
    if not layer.is_file() or not (source / "roboverse_source_manifest.json").is_file():
        pytest.skip("Prepare the pinned GenieSim asset snapshot first")
    output = tmp_path / "usd_physics_v4"
    provenance = prepare_object(source, output, "benchmark_building_blocks_073", physics_source="usd")
    assert provenance["format"] == "roboverse.icl.geniesim_rigid_object.v4"
    assert provenance["mass_kg"] == pytest.approx(0.1)
    physics = provenance["source_physics"]
    assert physics["source_static_friction"] == pytest.approx(0.5)
    assert physics["source_dynamic_friction"] == pytest.approx(0.3)
    assert physics["scene_overrides_applied"] is False
    assert len(physics["used_layer_sha256"]) >= 2
    mjcf = ET.parse(output / "object.xml").getroot()
    assert float(mjcf.find("worldbody/body/inertial").get("mass")) == pytest.approx(0.1)
    assert float(mjcf.find("worldbody/body/geom[@name='collision']").get("friction").split()[0]) == pytest.approx(0.3)
    assert "not applied" in provenance["physics_mapping"]["newton"]


@pytest.fixture
def source_instance(tmp_path):
    pytest.importorskip("pxr")
    from tools.icl.import_geniesim_specs import _scene_snapshot_from_usd

    scene = tmp_path / "scene.usda"
    scene.write_text(
        """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def "World" {
    def "Objects" {
        def Xform "bottle" (
            delete payload = @/geniesim_assets/objects/old/old_bottle/Aligned.usda@
            prepend payload = @/geniesim_assets/objects/genie/beverage_bottle/new_bottle/Aligned.usda@
        ) {
            double3 xformOp:translate = (0.1, 0.2, 0.3)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
        def Xform "hidden" (
            active = false
            prepend payload = @/geniesim_assets/objects/genie/bottle/hidden_bottle/Aligned.usda@
        ) {}
        def Xform "extra" (
            prepend payload = @/geniesim_assets/objects/genie/pen/new_pen/Aligned.usd@
        ) {}
    }
}
""",
        encoding="utf-8",
    )
    snapshot = _scene_snapshot_from_usd(scene)
    return {
        "task": "pick_object_type",
        "instance_id": "1",
        "source_files": {
            "scene.usda": {"path": "benchmark/pick_object_type/1/scene.usda", "sha256": snapshot["sha256"]}
        },
        "source_scene": snapshot,
        "scene_info": {
            "layout": {
                "bottle": {
                    "usd": "old_bottle",
                    "xyz": [9, 9, 9],
                    "xyzw": [0, 0, 0, 1],
                    "description": {"color": "black"},
                },
                "hidden": {"usd": "hidden_bottle"},
                "ghost": {"usd": "missing_bottle"},
            }
        },
    }, scene


def test_source_scene_snapshot_preserves_payload_visibility_and_sha(source_instance):
    from roboverse_pack.tasks.icl.geniesim_scene import _instance_scene_layout
    from tools.icl.import_geniesim_specs import _scene_reconciliation

    instance, path = source_instance
    snapshot = instance["source_scene"]
    assert snapshot["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert snapshot["objects"]["bottle"]["asset_id"] == "new_bottle"
    assert snapshot["objects"]["hidden"]["active"] is False
    assert _scene_reconciliation(instance["scene_info"], snapshot) == {
        "metadata_only": ["ghost"],
        "source_only": ["extra"],
        "inactive": ["hidden"],
        "asset_id_mismatches": {"bottle": {"metadata": "old_bottle", "source_payload": "new_bottle"}},
    }
    layout = _instance_scene_layout(instance)
    assert set(layout) == {"bottle", "extra"}
    assert layout["bottle"]["usd"] == "new_bottle"
    np.testing.assert_allclose(layout["bottle"]["xyz"], [0.1, 0.2, 0.3])
    assert layout["bottle"]["description"]["color"] == "black"
    assert layout["bottle"]["_source_scene_payload"]["metadata_asset_id"] == "old_bottle"


def test_source_scene_rejects_wrong_sha_and_unsupported_scale(source_instance):
    from roboverse_pack.tasks.icl.geniesim_scene import _instance_scene_layout

    instance, _ = source_instance
    instance["source_scene"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not match the recorded source file"):
        _instance_scene_layout(instance)
    instance["source_scene"]["sha256"] = instance["source_files"]["scene.usda"]["sha256"]
    instance["source_scene"]["objects"]["bottle"]["scale"] = [2.0, 1.0, 2.0]
    with pytest.raises(NotImplementedError, match=r"nonuniform scale.*cross-backend"):
        _instance_scene_layout(instance)


def test_source_scene_layer_and_prepared_provenance_must_match(source_instance, tmp_path):
    from roboverse_pack.tasks.icl.geniesim_scene import (
        _instance_scene_layout,
        _instance_wrapper,
        _payload_asset_root,
        _validate_payload_provenance,
    )

    instance, _ = source_instance
    layout = _instance_scene_layout(instance)
    entry = layout["extra"]
    root = _payload_asset_root(tmp_path, entry)
    assert root == tmp_path / "_source_layers" / "Aligned_usd"
    asset = root / "new_pen"
    asset.mkdir(parents=True)
    provenance = {
        "format": "roboverse.icl.geniesim_rigid_object.v3",
        "asset_id": "new_pen",
        "source_revision": "a0813aad7c16165daffbe6c2737754e0344809ee",
        "source_asset_directory": "objects/genie/pen/new_pen",
        "source_layer": "Aligned.usd",
        "source_files_sha256": {"Aligned.usd": "1" * 64},
    }
    _validate_payload_provenance(entry, provenance)
    wrong_layer = {**provenance, "source_layer": "Aligned.usda"}
    with pytest.raises(ValueError, match="does not match source-scene payload"):
        _validate_payload_provenance(entry, wrong_layer)
    (asset / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    (asset / "object.xml").write_text('<mujoco model="new_pen"><worldbody><body name="new_pen"/></worldbody></mujoco>')
    (asset / "object.urdf").write_text('<robot name="new_pen"><link name="new_pen"/></robot>')
    wrapper = _instance_wrapper(
        root, "new_pen", "extra", tmp_path / "wrapper", source_scene_payload=entry["_source_scene_payload"]
    )
    manifest = json.loads((wrapper / "instance.json").read_text(encoding="utf-8"))
    assert manifest["source_scene_payload"] == entry["_source_scene_payload"]


@pytest.mark.parametrize("version", ["v1", "v2", "v3"])
def test_object_probe_accepts_prepared_asset_versions_and_collision_resources(tmp_path, version):
    from tools.icl.probe_geniesim_object import _prepared_asset_resources

    (tmp_path / "provenance.json").write_text(
        json.dumps({"format": f"roboverse.icl.geniesim_rigid_object.{version}", "asset_id": "sample"})
    )
    for filename in ("object.urdf", "object.xml", "visual_0.obj", "collision.obj", "material_0.png"):
        (tmp_path / filename).write_bytes(b"fixture")
    provenance, resources = _prepared_asset_resources(tmp_path)
    assert provenance["format"].endswith(version)
    assert {path.name for path in resources} == {"visual_0.obj", "collision.obj", "material_0.png"}


def test_uniform_source_scale_changes_geometry_and_inertia_with_constant_mass(source_instance, tmp_path):
    import xml.etree.ElementTree as ET

    from roboverse_pack.tasks.icl.geniesim_scene import _instance_scene_layout, _instance_wrapper

    instance, _ = source_instance
    instance["source_scene"]["objects"]["bottle"]["scale"] = [2.0, 2.0, 2.0]
    entry = _instance_scene_layout(instance)["bottle"]
    asset = tmp_path / "assets/new_bottle"
    asset.mkdir(parents=True)
    (asset / "provenance.json").write_text(
        json.dumps({"format": "roboverse.icl.geniesim_rigid_object.v3", "asset_id": "new_bottle"})
    )
    (asset / "object.xml").write_text(
        '<mujoco model="new_bottle"><asset><mesh name="mesh" file="mesh.obj"/></asset><worldbody>'
        '<body name="new_bottle"><inertial pos="0.1 0 0" mass="0.3" diaginertia="1 2 3"/>'
        '<geom name="visual" type="mesh" mesh="mesh" pos="0 0.2 0"/></body></worldbody></mujoco>'
    )
    (asset / "object.urdf").write_text(
        '<robot name="new_bottle"><link name="new_bottle"><inertial><origin xyz="0.1 0 0"/>'
        '<mass value="0.3"/><inertia ixx="1" ixy="0" ixz="0" iyy="2" iyz="0" izz="3"/>'
        '</inertial><visual><origin xyz="0 0.2 0"/><geometry><mesh filename="mesh.obj"/>'
        "</geometry></visual></link></robot>"
    )
    wrapper = _instance_wrapper(
        asset.parent,
        "new_bottle",
        "bottle",
        tmp_path / "scaled_wrapper",
        source_scene_payload=entry["_source_scene_payload"],
    )
    mjcf = ET.parse(wrapper / "object.xml").getroot()
    urdf = ET.parse(wrapper / "object.urdf").getroot()
    assert mjcf.find("asset/mesh").get("scale") == "2 2 2"
    assert mjcf.find("worldbody/body/inertial").get("diaginertia") == "4 8 12"
    assert mjcf.find("worldbody/body/inertial").get("mass") == "0.3"
    assert urdf.find("link/visual/geometry/mesh").get("scale") == "2 2 2"
    assert urdf.find("link/inertial/mass").get("value") == "0.3"
    assert float(urdf.find("link/inertial/inertia").get("izz")) == pytest.approx(12)
    np.testing.assert_allclose(np.fromstring(urdf.find("link/inertial/origin").get("xyz"), sep=" "), [0.2, 0, 0])
    manifest = json.loads((wrapper / "instance.json").read_text(encoding="utf-8"))
    assert manifest["uniform_scale"] == 2.0


@pytest.mark.parametrize("has_source_scene, expected_offset", [(True, (0.0, 0.0)), (False, (-4.03, -0.032))])
def test_run_scene_uses_authored_world_frame_without_workspace_offset(
    monkeypatch, tmp_path, has_source_scene, expected_offset
):
    from types import SimpleNamespace

    from roboverse_pack.tasks.icl import control, geniesim_scene

    instance = {"source_scene": {"format": "snapshot"} if has_source_scene else None}
    run = {"source_initial_positions": []}
    evaluation = {
        "robot": {
            "robot_id": "G2",
            "robot_init_pose": {"workspace_00": {"position": [-0.66, 0.0, 0.0], "quaternion": [1, 0, 0, 0]}},
        },
        "scene": {"function_space_objects": {"workspace_00": {"position": [-4.03, -0.032, 0.8]}}},
    }
    monkeypatch.setattr(geniesim_scene, "load_run_instance", lambda *_: (run, instance, evaluation))
    monkeypatch.setattr(geniesim_scene, "resolve_geniesim_environment_hdr", lambda _: None)
    monkeypatch.setattr(
        control,
        "prepare_geniesim_reset_pose",
        lambda *_: SimpleNamespace(joint_positions={}, clipped_source_positions={}, max_equality_residual=0.0),
    )
    captured = {}

    def build_scene(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(geniesim_scene, "build_scene_scenario", build_scene)
    robot = SimpleNamespace(name="agibot_g2", default_joint_positions={}, joint_limits={})
    geniesim_scene.build_run_scenario(
        spec_path=tmp_path / "spec.json",
        run_id="g2_test",
        simulator="mujoco",
        robot=robot,
        assets_root=tmp_path,
        cache_root=tmp_path,
    )
    assert captured["workspace_offset_xy"] == expected_offset
    assert captured["robot_position"] == [-0.66, 0.0, 0.0]


@pytest.mark.examples
def test_raw_usd_layer_conversion_preserves_authored_geometry(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    from tools.icl.prepare_geniesim_objects import prepare_object

    root = Path(__file__).resolve().parents[1]
    source = root / "datasets/icl/sources/geniesim-assets"
    usd_path = source / "objects/benchmark/pen/benchmark_pen_007/Aligned.usd"
    if not usd_path.is_file() or not (source / "roboverse_source_manifest.json").is_file():
        pytest.skip("Prepare the pinned GenieSimAssets benchmark_pen_007 source first")
    info = prepare_object(source, tmp_path / "raw", "benchmark_pen_007", source_layer="Aligned.usd")
    stage = Usd.Stage.Open(str(usd_path))
    prim = stage.GetPrimAtPath("/World/body/visual")
    points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
    transform = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim), dtype=np.float64)
    points = points @ transform[:3, :3] + transform[3, :3]
    np.testing.assert_allclose(info["bounds_m"], [points.min(axis=0), points.max(axis=0)], atol=1e-7)
    assert info["source_layer"] == "Aligned.usd"
    assert info["source_files_sha256"]["Aligned.usd"] == hashlib.sha256(usd_path.read_bytes()).hexdigest()


def test_historical_asset_requires_exact_revision_and_hashes():
    from copy import deepcopy

    from roboverse_pack.tasks.icl.asset_sources import HISTORICAL_ASSETS
    from roboverse_pack.tasks.icl.geniesim_scene import _validate_payload_provenance

    asset_id = "benchmark_garage_kit_064"
    source = HISTORICAL_ASSETS[asset_id]
    entry = {"_source_scene_payload": {
        "asset_path": f"/geniesim_assets/{source['directory']}/Aligned.usda",
        "resolved_asset_id": asset_id,
    }}
    provenance = {
        "asset_id": asset_id,
        "source_revision": source["revision"],
        "source_asset_directory": source["directory"],
        "source_layer": "Aligned.usda",
        "source_files_sha256": dict(source["sha256"]),
    }
    _validate_payload_provenance(entry, provenance)
    for mutation in ("revision", "hash", "asset", "directory"):
        bad = deepcopy(provenance)
        if mutation == "revision":
            bad["source_revision"] = "a0813aad7c16165daffbe6c2737754e0344809ee"
        elif mutation == "hash":
            bad["source_files_sha256"]["_Aligned.usd"] = "0" * 64
        elif mutation == "asset":
            bad["asset_id"] = "another_asset"
        else:
            bad["source_asset_directory"] = "objects/wrong/path"
        with pytest.raises(ValueError, match="does not match"):
            _validate_payload_provenance(entry, bad)


def test_warehouse_conversion_keeps_source_convex_colliders_and_filters_noncolliders(tmp_path):
    pytest.importorskip("pxr")
    pytest.importorskip("coacd")
    pytest.importorskip("mujoco")
    import xml.etree.ElementTree as ET

    import mujoco
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from roboverse_pack.tasks.icl.geniesim_background_collision import prepare_warehouse_collisions

    source = tmp_path / "source"
    scene = "background/warehouse/warehouse_1/background.usda"
    source_path = source / scene
    source_path.parent.mkdir(parents=True)
    stage = Usd.Stage.CreateNew(str(source_path))
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    vertices = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    triangles = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
                 1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7]
    for name, enabled in (("active", True), ("disabled", False)):
        mesh = UsdGeom.Mesh.Define(stage, f"/World/background/{name}")
        mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in vertices])
        mesh.CreateFaceVertexCountsAttr([3] * 12)
        mesh.CreateFaceVertexIndicesAttr(triangles)
        mesh.AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(enabled)
    stage.GetRootLayer().Save()
    result = prepare_warehouse_collisions(tmp_path / "cache", source, scene)
    assert result is not None
    urdf, mjcf, resources = result
    provenance = json.loads((urdf.parent / "provenance.json").read_text(encoding="utf-8"))
    assert len(provenance["source_meshes"]) == 1
    assert provenance["source_meshes"][0]["prim"] == "/World/background/active"
    assert provenance["source_meshes"][0]["method"] == "source_convex_mesh"
    assert len(resources) == 1
    assert len(ET.parse(urdf).findall("link/collision")) == 1
    assert not ET.parse(urdf).findall("link/visual")
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    assert model.ngeom == 1
    assert model.nbody == 2  # a separate fixed link prevents Genesis merging collision SDFs
    np.testing.assert_allclose(model.geom_pos[0], [1, 2, 3], atol=1e-7)
    assert prepare_warehouse_collisions(tmp_path / "cache", source, scene) == result
    assert prepare_warehouse_collisions(tmp_path / "cache", source, "background/elsewhere.usda") is None
    mjcf.write_text(mjcf.read_text(encoding="utf-8") + "<!-- modified -->", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid cached warehouse collisions"):
        prepare_warehouse_collisions(tmp_path / "cache", source, scene)


@pytest.mark.examples
@pytest.mark.parametrize("simulator", ["mujoco", "genesis", "newton"])
def test_warehouse_source_tray_physically_supports_falling_block(tmp_path, simulator):
    import os

    pytest.importorskip("pxr")
    pytest.importorskip("coacd")
    requested = set(os.environ.get("ROBOVERSE_ICL_TEST_BACKENDS", "").split(","))
    if simulator != "mujoco" and simulator not in requested:
        pytest.skip(f"Set ROBOVERSE_ICL_TEST_BACKENDS to include {simulator}")
    pytest.importorskip(simulator)
    import torch

    from metasim.constants import PhysicStateType
    from metasim.scenario.objects import RigidObjCfg
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.scenario.simulator_params import SimParamCfg
    from metasim.utils.setup_util import get_handler
    from roboverse_pack.tasks.icl.geniesim_background_collision import prepare_warehouse_collisions
    from roboverse_pack.tasks.icl.geniesim_scene import _instance_wrapper

    if simulator == "newton" and not torch.cuda.is_available():
        pytest.skip("Newton contact validation requires CUDA")
    root = Path(__file__).resolve().parents[1]
    source = root / "datasets/icl/sources/geniesim-assets"
    scene = "background/warehouse/warehouse_1/background.usda"
    asset_root = root / "roboverse_data/icl/objects/geniesim"
    block_dir = asset_root / "benchmark_building_blocks_073"
    if not (source / scene).is_file() or not (block_dir / "provenance.json").is_file():
        pytest.skip("Prepare the warehouse background and source block asset first")
    urdf, mjcf, resources = prepare_warehouse_collisions(
        root / "roboverse_data/icl/cache/geniesim_scene_wrappers_v6", source, scene
    )
    collision = RigidObjCfg(name="warehouse_collision", urdf_path=str(urdf), mjcf_path=str(mjcf),
                            extra_resources=resources, physics=PhysicStateType.GEOM, fix_base_link=True,
                            newton_hide_collision_shapes=True)
    collision.file_type.update({"mujoco": "mjcf", "genesis": "mjcf", "newton": "urdf"})
    block_path = _instance_wrapper(asset_root, "benchmark_building_blocks_073", "block", tmp_path / "wrapper")
    block_resources = [str(p) for p in block_dir.iterdir() if p.suffix.lower() in {".obj", ".mtl", ".png"}]
    block = RigidObjCfg(name="block", urdf_path=str(block_path / "object.urdf"),
                        mjcf_path=str(block_path / "object.xml"), extra_resources=block_resources,
                        physics=PhysicStateType.RIGIDBODY, default_position=(0.3, -0.9, 1.1))
    block.file_type.update({"mujoco": "mjcf", "genesis": "mjcf", "newton": "urdf"})
    scenario = ScenarioCfg(objects=[collision, block], simulator=simulator, headless=True, num_envs=1,
                           sim_params=SimParamCfg(dt=1 / 600, njmax=512, newton_mujoco_enable_multiccd=True), decimation=20)
    handler = get_handler(scenario)
    try:
        handler.set_states([{"objects": {obj.name: {"pos": torch.tensor(obj.default_position),
                            "rot": torch.tensor(obj.default_orientation)} for obj in (collision, block)}, "robots": {}}])
        for _ in range(60):
            handler.simulate()
        final = handler.get_states(mode="tensor").objects["block"].root_state[0, :3].cpu().numpy()
        # Authored tray floor is at z=0.768 m; the block is 0.04 m high.
        np.testing.assert_allclose(final[:2], [0.3, -0.9], atol=0.025)
        assert final[2] == pytest.approx(0.788, abs=0.005)
    finally:
        handler.close()
