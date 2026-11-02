"""CPU-only conversion and provenance contracts using tiny generated GLBs."""

from __future__ import annotations

import json
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import pytest
from imitator_game_robotwin_fixtures import make_synthetic_robotwin_source

from roboverse_pack.tasks.imitator_game.clean_desk import (
    ASSET_ARCHIVE_SHA256,
    ASSET_LICENSE,
    ASSET_REVISION,
    validate_assets as validate_h19_assets,
)
from roboverse_pack.tasks.imitator_game.robotwin_assets import (
    ROBOTWIN_LICENSE,
    ROBOTWIN_PROFILES,
    validate_prepared_robotwin_asset,
)
from tools.icl.prepare_robotwin_imitator_assets import prepare_robotwin_asset


@pytest.mark.parametrize("profile_key", ("h7_breadbasket", "h19_tissue_box"))
def test_converter_preserves_materials_axes_scale_and_collision_parts(tmp_path, profile_key):
    """The converter keeps source visual texture/transform data and does not merge convex pieces."""
    source, profile = make_synthetic_robotwin_source(tmp_path, profile_key)
    output = tmp_path / "prepared"
    prepare_robotwin_asset(profile=profile, source_directory=source, output_directory=output)
    provenance = validate_prepared_robotwin_asset(output, profile)

    assert provenance["source_files_sha256"] == {key: value[0] for key, value in profile.source_files.items()}
    assert provenance["metadata_scale"] == list(profile.metadata_scale)
    assert provenance["baked_scale"] == list(profile.baked_scale)
    assert provenance["collision_component_count"] == 2
    assert provenance["visual_component_count"] == 1
    assert provenance["bounds_m"][0] == pytest.approx([
        -0.6 * profile.baked_scale[0],
        -0.15 * profile.baked_scale[1],
        -0.1 * profile.baked_scale[2],
    ])
    assert provenance["bounds_m"][1] == pytest.approx([
        0.6 * profile.baked_scale[0],
        0.15 * profile.baked_scale[1],
        0.1 * profile.baked_scale[2],
    ])
    assert "glTF metallic/roughness channels" in provenance["visual_materials"]

    urdf = ET.parse(output / "object.urdf").getroot()
    mjcf = ET.parse(output / "object.xml").getroot()
    assert len(urdf.findall(".//visual")) == 1
    assert len(urdf.findall(".//collision")) == 2
    assert len(mjcf.findall(".//geom[@group='0']")) == 2
    assert len(list((output / "visual").rglob("*.png"))) == 1
    assert mjcf.find(".//texture") is not None


def test_converter_checks_source_hash_and_never_overwrites(tmp_path):
    """Changed inputs fail before output and an existing output remains untouched."""
    source, profile = make_synthetic_robotwin_source(tmp_path, "h7_breadbasket")
    (source / "visual" / "base3.glb").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="pinned RoboTwin revision payload"):
        prepare_robotwin_asset(profile=profile, source_directory=source, output_directory=tmp_path / "bad")

    protected = tmp_path / "existing"
    protected.mkdir()
    marker = protected / "keep.txt"
    marker.write_text("untouched", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_robotwin_asset(profile=profile, source_directory=source, output_directory=protected)
    assert marker.read_text(encoding="utf-8") == "untouched"


def test_converter_builds_provenance_when_download_has_no_sidecar(tmp_path):
    """Pinned per-file hashes and model metadata suffice for a fresh HF download folder."""
    source, profile = make_synthetic_robotwin_source(tmp_path, "h7_breadbasket")
    (source / "provenance.json").unlink()
    output = tmp_path / "prepared"
    prepare_robotwin_asset(profile=profile, source_directory=source, output_directory=output)
    prepared = validate_prepared_robotwin_asset(output, profile)
    assert prepared["asset_license"] == ROBOTWIN_LICENSE
    assert (output / "source_provenance.json").is_file()


def test_prepared_asset_strictly_checks_generated_mesh_checksums(tmp_path):
    """A changed wrapper resource fails provenance validation before a task can load it."""
    source, profile = make_synthetic_robotwin_source(tmp_path, "h19_tissue_box")
    output = tmp_path / "prepared"
    prepare_robotwin_asset(profile=profile, source_directory=source, output_directory=output)
    material = next((output / "visual").rglob("*.png"))
    material.write_bytes(material.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Generated wrapper checksum mismatch"):
        validate_prepared_robotwin_asset(output, profile)


def test_converted_tissue_box_passes_h19_task_asset_validation(tmp_path, monkeypatch):
    """The task adapter resolves every generated URDF/MJCF mesh and texture reference."""
    source, synthetic_profile = make_synthetic_robotwin_source(tmp_path, "h19_tissue_box")
    monkeypatch.setitem(ROBOTWIN_PROFILES, "h19_tissue_box", synthetic_profile)

    asset_root = tmp_path / "h19"
    prepared_directory = asset_root / "robotwin_prepared" / "023_tissue-box" / "model_3"
    prepare_robotwin_asset(
        profile=synthetic_profile,
        source_directory=source,
        output_directory=prepared_directory,
    )

    brick_directory = asset_root / "ycb" / "061_foam_brick"
    brick_directory.mkdir(parents=True)
    (brick_directory / "object.urdf").write_text(
        '<robot name="brick"><link name="body"/></robot>',
        encoding="utf-8",
    )
    (brick_directory / "object.xml").write_text(
        '<mujoco model="brick"><worldbody><body name="brick"/></worldbody></mujoco>',
        encoding="utf-8",
    )
    (brick_directory / "provenance.json").write_text(
        json.dumps(
            {
                "source_object": "061_foam_brick",
                "source_archive_sha256": ASSET_ARCHIVE_SHA256,
                "source_revision": ASSET_REVISION,
                "asset_license": ASSET_LICENSE,
                "task_scale_baked_in": 0.4,
                "bounds_m": [[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]],
            }
        ),
        encoding="utf-8",
    )

    prepared = validate_h19_assets("L3", asset_root=asset_root)

    assert prepared["tissue_box"].directory == prepared_directory
    assert prepared["brick"].directory == brick_directory


def test_h19_l3_geometric_metric_has_positive_and_negative_cases():
    """L3 task success retains the source distance/step/travel thresholds but omits grasp parity."""
    from roboverse_pack.tasks.imitator_game.clean_desk import LEVEL_SPECS, ImitatorGameCleanDeskL3

    def make_env():
        env = object.__new__(ImitatorGameCleanDeskL3)
        env._level_spec = LEVEL_SPECS["L3"]
        env._clean_steps = torch.zeros(1, dtype=torch.int32)
        env._clean_distance = torch.zeros(1, dtype=torch.float32)
        env._last_task_object_xy = torch.zeros((1, 2), dtype=torch.float32)
        env._has_last_task_object = torch.zeros(1, dtype=torch.bool)
        return env

    import torch

    def states(x):
        tissue = SimpleNamespace(root_state=torch.tensor([[x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]))
        brick = SimpleNamespace(root_state=torch.zeros((1, 7)))
        return SimpleNamespace(objects={"tissue_box": tissue, "brick": brick})

    success_env = make_env()
    assert not success_env._terminated(states(0.0)).item()
    for index in range(1, 91):
        x = 0.1 if index % 2 else -0.1
        done = success_env._terminated(states(x))
    assert done.item()

    stationary_env = make_env()
    done = torch.tensor([False])
    for _ in range(91):
        done = stationary_env._terminated(states(0.0))
    assert not done.item()
