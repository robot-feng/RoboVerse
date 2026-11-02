# Copyright (c) 2026 RoboVerse contributors
# SPDX-License-Identifier: Apache-2.0
"""Pinned RoboTwin asset provenance contracts used by Imitator Game adapters."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROBOTWIN_REPOSITORY = "https://huggingface.co/datasets/TianxingChen/RoboTwin2.0"
ROBOTWIN_REVISION = "39b4a57384f28b6ed0ea1118871ae302e67eae38"
ROBOTWIN_COMMIT_DATE = "2025-06-16T17:14:26Z"
ROBOTWIN_LICENSE = "MIT License (declared in repository dataset card and objects/README.md)"
ROBOTWIN_LICENSE_SOURCE = (
    f"https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/blob/{ROBOTWIN_REVISION}/objects/README.md"
)
ROBOTWIN_DATASET_LICENSE_SOURCE = f"https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/{ROBOTWIN_REVISION}"


@dataclass(frozen=True)
class RobotwinAssetProfile:
    """One pinned source model and the task-faithful scale rule for its prepared wrapper."""

    key: str
    source_object: str
    model_id: int
    metadata_scale: tuple[float, float, float]
    baked_scale: tuple[float, float, float]
    task_scale_semantics: str
    source_files: dict[str, tuple[str, int]]
    mass_kg: float | None
    static: bool


ROBOTWIN_PROFILES: dict[str, RobotwinAssetProfile] = {
    "h7_breadbasket": RobotwinAssetProfile(
        key="h7_breadbasket",
        source_object="076_breadbasket",
        model_id=3,
        metadata_scale=(0.1, 0.1, 0.1),
        baked_scale=(0.1, 0.1, 0.1),
        task_scale_semantics=(
            "Bake RoboTwin model_data3 metadata scale; retain H7 task actor scale 1.8 in the scene config."
        ),
        source_files={
            "visual/base3.glb": (
                "1cec728919add43f6130f3468fe0d89bacc2bb11c95446d4fe10282843e9d0ff",
                8644060,
            ),
            "collision/base3.glb": (
                "c8834678940f48c24b7f33046896ecac1d7e0a6b8cbb12584b469cdb66b3fa1f",
                331780,
            ),
            "model_data3.json": (
                "cf87ecaa1fabe1c5d89c5089ca7fefcd6408b6b9609caaabdc7d5e5d6e5bfdb7",
                1509,
            ),
        },
        mass_kg=None,
        static=True,
    ),
    "h19_tissue_box": RobotwinAssetProfile(
        key="h19_tissue_box",
        source_object="023_tissue-box",
        model_id=3,
        metadata_scale=(0.06, 0.06, 0.06),
        baked_scale=(0.2, 0.13, 0.06),
        task_scale_semantics=("Bake H19's replace_scale=True task scale (0.2, 0.13, 0.06); ignore model_data3 scale."),
        source_files={
            "visual/base3.glb": (
                "c467b09fb192667ea2d4240a19d7872e351007c95fc7d8b917348772d758eccb",
                1463836,
            ),
            "collision/base3.glb": (
                "4da5338215006b5e6dd79cc6278e47b535823814a2b4b9eedc2805f50dbbad56",
                288844,
            ),
            "model_data3.json": (
                "aaa0ae51c2cdc8ca07ec5dd6c5671fdb18e5d26dc9691e2bad6bcbcbe8f882de",
                2260,
            ),
        },
        mass_kg=0.5,
        static=False,
    ),
}


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _float_tuple(value: Any, *, field: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError(f"{field} must be a non-empty numeric array")
    try:
        numbers = tuple(float(component) for component in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a non-empty numeric array") from exc
    if not all(math.isfinite(number) for number in numbers):
        raise ValueError(f"{field} must contain only finite values")
    return numbers


def validate_robotwin_source(source_dir: Path, profile: RobotwinAssetProfile) -> tuple[dict, dict]:
    """Verify pinned upstream metadata, license facts and byte-identical source payloads."""
    source_dir = Path(source_dir)
    provenance_path = source_dir / "provenance.json"
    has_source_provenance = provenance_path.is_file()
    try:
        if has_source_provenance:
            source_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            if not isinstance(source_provenance, dict):
                raise ValueError("source provenance must be a JSON object")
        else:
            source_provenance = {}
        metadata = json.loads((source_dir / "model_data3.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read pinned RoboTwin source metadata under {source_dir}: {exc}") from exc

    expected_identity = {
        "source_repository": ROBOTWIN_REPOSITORY,
        "source_revision": ROBOTWIN_REVISION,
        "source_commit_date": ROBOTWIN_COMMIT_DATE,
        "source_object": profile.source_object,
        "model_id": profile.model_id,
        "upstream_variant": "base3",
        "asset_license": ROBOTWIN_LICENSE,
        "license_source": ROBOTWIN_LICENSE_SOURCE,
        "dataset_license_source": ROBOTWIN_DATASET_LICENSE_SOURCE,
    }
    if has_source_provenance:
        for key, expected in expected_identity.items():
            if source_provenance.get(key) != expected:
                raise ValueError(f"source provenance {key} must match pinned value {expected!r}")
    else:
        source_provenance.update(expected_identity)

    listed_files = {
        item.get("path"): item
        for item in source_provenance.get("files", [])
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    observed_hashes: dict[str, str] = {}
    observed_files = []
    for relative_path, (expected_hash, expected_size) in profile.source_files.items():
        asset_path = source_dir / Path(relative_path)
        if not asset_path.is_file():
            raise FileNotFoundError(f"Pinned RoboTwin source file is missing: {asset_path}")
        source_file = listed_files.get(relative_path)
        actual_size = asset_path.stat().st_size
        actual_hash = sha256_file(asset_path)
        if actual_size != expected_size or actual_hash != expected_hash:
            raise ValueError(f"{relative_path} does not match the pinned RoboTwin revision payload")
        if source_provenance.get("files") and not isinstance(source_file, dict):
            raise ValueError(f"source provenance does not list {relative_path}")
        if isinstance(source_file, dict):
            if source_file.get("size_bytes") != expected_size or source_file.get("sha256") != expected_hash:
                raise ValueError(f"source provenance checksum/size for {relative_path} is incorrect")
            if relative_path.endswith(".glb") and source_file.get("upstream_lfs_sha256") != expected_hash:
                raise ValueError(f"source provenance LFS hash for {relative_path} is incorrect")
            if source_file.get("source_path") != f"objects/{profile.source_object}/{relative_path}":
                raise ValueError(f"source provenance path for {relative_path} is incorrect")
        observed_hashes[relative_path] = actual_hash
        observed_files.append({
            "path": relative_path,
            "source_path": f"objects/{profile.source_object}/{relative_path}",
            "size_bytes": actual_size,
            "sha256": actual_hash,
            "upstream_lfs_sha256": actual_hash if relative_path.endswith(".glb") else None,
        })
    if not source_provenance.get("files"):
        source_provenance["files"] = observed_files
    if source_provenance.get("metadata_scale") is not None:
        recorded_scale = _float_tuple(source_provenance["metadata_scale"], field="provenance.metadata_scale")
        if recorded_scale != profile.metadata_scale:
            raise ValueError(f"source provenance metadata_scale must be {profile.metadata_scale}")
    source_provenance["metadata_scale"] = list(profile.metadata_scale)
    source_provenance["official_object_documentation"] = (
        f"https://robotwin-platform.github.io/doc/objects/{profile.source_object}.html"
    )

    metadata_scale = _float_tuple(metadata.get("scale"), field="model_data3.scale")
    if len(metadata_scale) != 3 or any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
        for actual, expected in zip(metadata_scale, profile.metadata_scale, strict=True)
    ):
        raise ValueError(f"model_data3.scale must be {profile.metadata_scale}")
    provenance_scale = _float_tuple(source_provenance.get("metadata_scale"), field="provenance.metadata_scale")
    if provenance_scale != profile.metadata_scale:
        raise ValueError(f"source provenance metadata_scale must be {profile.metadata_scale}")
    matrix = metadata.get("transform_matrix")
    if matrix != [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]:
        raise ValueError("source model_data3 transform_matrix must be identity for this pinned conversion")
    return source_provenance, {"model_data": metadata, "source_files_sha256": observed_hashes}


def validate_prepared_robotwin_asset(asset_dir: Path, profile: RobotwinAssetProfile) -> dict:
    """Check generated wrappers and all source/generated hashes before a task loads the asset."""
    asset_dir = Path(asset_dir)
    provenance_path = asset_dir / "provenance.json"
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read prepared Robotwin provenance at {provenance_path}: {exc}") from exc
    expected = {
        "asset_schema": "roboverse.imitator_game.robotwin_prepared.v1",
        "source_repository": ROBOTWIN_REPOSITORY,
        "source_revision": ROBOTWIN_REVISION,
        "source_commit_date": ROBOTWIN_COMMIT_DATE,
        "source_object": profile.source_object,
        "model_id": profile.model_id,
        "upstream_variant": "base3",
        "asset_license": ROBOTWIN_LICENSE,
        "license_source": ROBOTWIN_LICENSE_SOURCE,
        "dataset_license_source": ROBOTWIN_DATASET_LICENSE_SOURCE,
        "metadata_scale": list(profile.metadata_scale),
        "baked_scale": list(profile.baked_scale),
        "task_scale_semantics": profile.task_scale_semantics,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"prepared provenance {key} must match pinned value {value!r}")
    if provenance.get("source_files_sha256") != {name: values[0] for name, values in profile.source_files.items()}:
        raise ValueError("source_files_sha256 does not match the pinned RoboTwin files")
    try:
        source_provenance = json.loads((asset_dir / "source_provenance.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read copied upstream source provenance: {exc}") from exc
    for key, value in {
        "source_repository": ROBOTWIN_REPOSITORY,
        "source_revision": ROBOTWIN_REVISION,
        "source_object": profile.source_object,
        "model_id": profile.model_id,
        "upstream_variant": "base3",
        "asset_license": ROBOTWIN_LICENSE,
        "license_source": ROBOTWIN_LICENSE_SOURCE,
        "dataset_license_source": ROBOTWIN_DATASET_LICENSE_SOURCE,
    }.items():
        if source_provenance.get(key) != value:
            raise ValueError(f"copied source provenance {key} does not match the pinned source")
    source_file_list = {
        item.get("path"): item
        for item in source_provenance.get("files", [])
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    copied_paths = {
        "visual/base3.glb": "source/visual/base3.glb",
        "collision/base3.glb": "source/collision/base3.glb",
        "model_data3.json": "model_data3.json",
    }
    for relative_path, (expected_hash, expected_size) in profile.source_files.items():
        copied_path = asset_dir / copied_paths[relative_path]
        listed = source_file_list.get(relative_path)
        if not copied_path.is_file() or copied_path.stat().st_size != expected_size:
            raise ValueError(f"copied pinned source file has wrong size or is missing: {relative_path}")
        if sha256_file(copied_path) != expected_hash:
            raise ValueError(f"copied pinned source file checksum mismatch: {relative_path}")
        if (
            not isinstance(listed, dict)
            or listed.get("sha256") != expected_hash
            or listed.get("size_bytes") != expected_size
        ):
            raise ValueError(f"copied source provenance checksum/size mismatch: {relative_path}")
    generated = provenance.get("generated_files_sha256")
    if not isinstance(generated, dict) or not generated:
        raise ValueError("generated_files_sha256 must list generated wrapper/resources")
    for relative_path, digest in generated.items():
        file_path = asset_dir / relative_path
        if not file_path.is_file():
            raise FileNotFoundError(f"Generated wrapper resource is missing: {file_path}")
        if sha256_file(file_path) != digest:
            raise ValueError(f"Generated wrapper checksum mismatch: {relative_path}")
    for required in (
        "object.urdf",
        "object.xml",
        "model_data3.json",
        "source_provenance.json",
        "source/visual/base3.glb",
        "source/collision/base3.glb",
    ):
        if not (asset_dir / required).is_file():
            raise FileNotFoundError(f"Prepared Robotwin asset is missing required file: {asset_dir / required}")
    recorded_scale = _float_tuple(provenance.get("baked_scale"), field="baked_scale")
    if len(recorded_scale) != 3 or recorded_scale != profile.baked_scale:
        raise ValueError(f"baked_scale must be {profile.baked_scale}")
    if profile.mass_kg is not None and not math.isclose(
        float(provenance.get("mass_kg", -1)), profile.mass_kg, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError(f"mass_kg must be {profile.mass_kg} kg")
    return provenance
