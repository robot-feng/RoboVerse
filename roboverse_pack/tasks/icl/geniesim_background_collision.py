# Copyright (c) 2026 RoboVerse contributors.
# SPDX-License-Identifier: MPL-2.0
"""Portable collision meshes for the authored Genie Sim warehouse background."""

from __future__ import annotations

import hashlib
import json
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_cached_collisions(destination: Path, collision_hash: str) -> tuple[Path, Path, list[str]]:
    provenance = json.loads((destination / "provenance.json").read_text(encoding="utf-8"))
    filenames = provenance.get("files", [])
    if not isinstance(filenames, list) or any(not isinstance(n, str) or Path(n).name != n for n in filenames):
        raise ValueError(f"Invalid cached warehouse collision file list: {destination}")
    required = [destination / filename for filename in filenames]
    hashes = provenance.get("files_sha256", {})
    if (
        provenance.get("content_sha256") != collision_hash
        or len(required) < 3
        or filenames[:2] != ["collision.urdf", "collision.xml"]
        or not isinstance(hashes, dict)
        or set(hashes) != set(filenames)
        or not all(path.is_file() and _file_sha256(path) == hashes[path.name] for path in required)
    ):
        raise ValueError(f"Invalid cached warehouse collisions: {destination}")
    return required[0], required[1], [str(path) for path in required[2:]]


def prepare_warehouse_collisions(
    cache_root: Path, source_root: Path, scene_usd: str | None, *, exclude_asset_ids=()
) -> tuple[Path, Path, list[str]] | None:
    """Convert enabled source warehouse mesh colliders into cached static geometry.

    PhysX's non-convex decomposition is not portable. CoACD produces an
    explicitly approximate convex decomposition for those meshes; individual
    source meshes that are already convex are retained without decomposition.
    The cache key includes source mesh vertices, transforms, parameters and
    CoACD version, rather than assuming a background layer hash covers payloads.
    """
    if not scene_usd or scene_usd.lstrip("/\\").replace("\\", "/") != (
        "background/warehouse/warehouse_1/background.usda"
    ):
        return None
    import coacd
    import trimesh
    from pxr import Usd, UsdGeom, UsdPhysics

    from roboverse_pack.tasks.icl.geniesim_scene import _source_mesh_triangles

    source_root = Path(source_root).resolve()
    scene_path = (source_root / scene_usd.lstrip("/\\")).resolve()
    if not scene_path.is_relative_to(source_root) or not scene_path.is_file():
        raise FileNotFoundError(f"Warehouse collision conversion needs the source USD: {scene_path}")
    stage = Usd.Stage.Open(str(scene_path))
    if not stage or UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise ValueError(f"Cannot read Z-up warehouse background: {scene_path}")
    meters = float(UsdGeom.GetStageMetersPerUnit(stage))
    cache = UsdGeom.XformCache()
    selected = []
    digest = hashlib.sha256(b"geniesim-warehouse-colliders-v5-coacd-threshold-0.03-resolution-300-iterations-30")
    digest.update(version("coacd").encode("ascii"))
    excluded = set(exclude_asset_ids)
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if (
            not path.startswith("/World/background/")
            or excluded.intersection(path.split("/"))
            or not prim.IsA(UsdGeom.Mesh)
            or not prim.HasAPI(UsdPhysics.CollisionAPI)
            or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False
        ):
            continue
        vertices, triangles, _ = _source_mesh_triangles(prim)
        transform = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64)
        if not len(vertices) or not len(triangles) or not np.isfinite(vertices).all():
            raise ValueError(f"Invalid source warehouse collision mesh: {path}")
        vertices = np.asarray(vertices, dtype=np.float64)
        triangles = np.asarray(triangles, dtype=np.int32)
        digest.update(path.encode("utf-8"))
        digest.update(vertices.tobytes())
        digest.update(triangles.tobytes())
        digest.update(transform.tobytes())
        selected.append((path, vertices, triangles, transform))
    digest.update(np.asarray([meters], dtype=np.float64).tobytes())
    if not selected:
        raise ValueError(f"No enabled source warehouse collisions found in {scene_path}")

    collision_hash = digest.hexdigest()
    destination = Path(cache_root).resolve() / f"geniesim_warehouse_collisions_v5_{collision_hash[:16]}"
    provenance_path = destination / "provenance.json"
    if provenance_path.is_file():
        return _load_cached_collisions(destination, collision_hash)
    if destination.exists():
        raise FileExistsError(f"Incomplete cached warehouse collisions: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    decomposition_cache = {}
    with tempfile.TemporaryDirectory(prefix="warehouse-collision-", dir=destination.parent) as temp:
        staging = Path(temp) / destination.name
        staging.mkdir()
        geometries = []
        source_records = []
        for path, vertices, triangles, transform in selected:
            mesh_hash = hashlib.sha256(vertices.tobytes() + triangles.tobytes()).hexdigest()
            if mesh_hash not in decomposition_cache:
                mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=True)
                if mesh.is_convex:
                    parts = [(vertices, triangles)]
                    method = "source_convex_mesh"
                else:
                    parts = coacd.run_coacd(
                        coacd.Mesh(vertices, triangles),
                        threshold=0.03,
                        resolution=300,
                        preprocess_resolution=30,
                        mcts_nodes=8,
                        mcts_iterations=30,
                        mcts_max_depth=3,
                        seed=0,
                    )
                    method = "coacd_approximate_convex_decomposition"
                    if not parts:
                        raise RuntimeError(f"Convex decomposition yielded no meshes for {path}")
                decomposition_cache[mesh_hash] = (parts, method)
            parts, method = decomposition_cache[mesh_hash]
            for vertices_part, faces_part in parts:
                world_vertices = (np.asarray(vertices_part) @ transform[:3, :3] + transform[3, :3]) * meters
                name = f"collision_{len(geometries):04d}.obj"
                trimesh.Trimesh(vertices=world_vertices, faces=faces_part, process=False).export(staging / name)
                geometries.append(name)
            source_records.append({
                "prim": path,
                "source_mesh_sha256": mesh_hash,
                "method": method,
                "parts": len(parts),
            })

        urdf = ['<?xml version="1.0"?><robot name="warehouse_collision"><link name="warehouse_collision"/>']
        mjcf = ['<mujoco model="warehouse_collision"><asset>']
        for index, name in enumerate(geometries):
            path = (destination / name).as_posix()
            # Keep far-apart shapes on separate fixed links: Newton's exported
            # MuJoCo midphase may miss contacts within a single large body BVH.
            urdf.append(
                f'<link name="collision_body_{index}"><collision name="collision_{index}">'
                f'<geometry><mesh filename="{path}"/></geometry></collision></link>'
                f'<joint name="collision_joint_{index}" type="fixed"><parent link="warehouse_collision"/>'
                f'<child link="collision_body_{index}"/></joint>'
            )
            mjcf.append(f'<mesh name="collision_{index}" file="{path}" inertia="shell"/>')
        urdf.append("</robot>")
        mjcf.append("</asset><worldbody>")
        for index in range(len(geometries)):
            # Genesis merges geoms on the same link into one SDF. Separate fixed
            # bodies retain each source convex piece instead of closing an
            # entire 14 m warehouse into a single spurious collision volume.
            mjcf.append(
                f'<body name="collision_body_{index}"><geom name="collision_{index}" '
                f'type="mesh" mesh="collision_{index}" group="3"/></body>'
            )
        mjcf.append("</worldbody></mujoco>")
        (staging / "collision.urdf").write_text("".join(urdf), encoding="utf-8")
        (staging / "collision.xml").write_text("".join(mjcf), encoding="utf-8")
        provenance = {
            "format": "roboverse.icl.geniesim_warehouse_collisions.v5",
            "source_scene": scene_usd,
            "content_sha256": collision_hash,
            "source_meshes": source_records,
            "coacd_version": version("coacd"),
            "coacd_parameters": {
                "threshold": 0.03,
                "resolution": 300,
                "preprocess_resolution": 30,
                "mcts_nodes": 8,
                "mcts_iterations": 30,
                "mcts_max_depth": 3,
                "seed": 0,
            },
            "limitations": "CoACD approximates PhysX convex decomposition; friction is not calibrated",
            "files": ["collision.urdf", "collision.xml", *geometries],
        }
        provenance["files_sha256"] = {name: _file_sha256(staging / name) for name in provenance["files"]}
        (staging / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        try:
            staging.rename(destination)
        except FileExistsError:
            # Another process may have completed the same deterministic cache.
            return _load_cached_collisions(destination, collision_hash)
    return _load_cached_collisions(destination, collision_hash)
