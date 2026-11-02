"""Shared tiny RoboTwin GLB fixtures for simulator-free task contract tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from roboverse_pack.tasks.imitator_game.robotwin_assets import (
    ROBOTWIN_COMMIT_DATE,
    ROBOTWIN_DATASET_LICENSE_SOURCE,
    ROBOTWIN_LICENSE,
    ROBOTWIN_LICENSE_SOURCE,
    ROBOTWIN_PROFILES,
    ROBOTWIN_REPOSITORY,
    ROBOTWIN_REVISION,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_synthetic_robotwin_source(root: Path, profile_key: str):
    """Write small textured visual and multi-part collision GLBs with pinned profile metadata."""
    profile = ROBOTWIN_PROFILES[profile_key]
    source = root / "source"
    (source / "visual").mkdir(parents=True)
    (source / "collision").mkdir(parents=True)

    image = Image.new("RGB", (4, 4), (220, 40, 15))
    visual_mesh = trimesh.creation.box(extents=(2.0, 1.0, 0.5))
    visual_mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(visual_mesh.vertices), 2), dtype=np.float64),
        material=trimesh.visual.material.PBRMaterial(baseColorTexture=image),
    )
    visual_scene = trimesh.Scene()
    visual_scene.add_geometry(visual_mesh, geom_name="textured_box", node_name="textured_box")
    visual_scene.export(source / "visual" / "base3.glb", file_type="glb")

    collision_scene = trimesh.Scene()
    for index, x in enumerate((-0.4, 0.4)):
        component = trimesh.creation.box(extents=(0.4, 0.3, 0.2))
        transform = trimesh.transformations.translation_matrix((x, 0.0, 0.0))
        collision_scene.add_geometry(
            component,
            geom_name=f"convex_{index}",
            node_name=f"convex_{index}",
            transform=transform,
        )
    collision_scene.export(source / "collision" / "base3.glb", file_type="glb")

    model_data = {
        "center": [0.0, 0.0, 0.0],
        "extents": [2.0, 1.0, 0.5],
        "scale": list(profile.metadata_scale),
        "transform_matrix": np.eye(4).tolist(),
        "stable": True,
    }
    (source / "model_data3.json").write_text(json.dumps(model_data), encoding="utf-8")

    checksums = {}
    files = []
    for relative_path in ("visual/base3.glb", "collision/base3.glb", "model_data3.json"):
        path = source / relative_path
        digest = _sha256(path)
        checksums[relative_path] = (digest, path.stat().st_size)
        files.append({
            "path": relative_path,
            "source_path": f"objects/{profile.source_object}/{relative_path}",
            "size_bytes": path.stat().st_size,
            "sha256": digest,
            "upstream_lfs_sha256": digest if relative_path.endswith(".glb") else None,
        })
    source_provenance = {
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
        "official_object_documentation": "https://robotwin-platform.github.io/doc/objects/test.html",
        "files": files,
    }
    (source / "provenance.json").write_text(json.dumps(source_provenance, indent=2), encoding="utf-8")
    synthetic_profile = replace(profile, source_files=checksums)
    return source, synthetic_profile
