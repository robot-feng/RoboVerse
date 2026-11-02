"""Prepare selected SynthICL mesh assets for MetaSim without changing source files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh
import yaml
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.icl.download_dtd import MD5 as DTD_MD5
from tools.icl.prepare_assets import _sha256, _write_wrappers

SOURCE_REPOSITORY = "https://huggingface.co/datasets/sniper-qian/SynthICL-PseudoDemo-Objects"
SOURCE_REVISION = "5f3124eff350ea0e1014253c700842aa94345607"


def write_source_manifest(source: Path) -> Path:
    """Save the official pinned-revision file list for offline integrity checks."""
    from huggingface_hub import HfApi

    repo = SOURCE_REPOSITORY.rsplit("datasets/", 1)[-1]
    info = HfApi(endpoint="https://huggingface.co", token=False).dataset_info(
        repo, revision=SOURCE_REVISION, files_metadata=True, token=False
    )
    if info.sha != SOURCE_REVISION or not info.siblings:
        raise ValueError("Official source did not return the pinned dataset revision")
    files = {}
    for item in info.siblings:
        if item.size is None or not item.blob_id:
            raise ValueError(f"Official source omitted checksum or size: {item.rfilename}")
        files[item.rfilename] = {
            "size": item.size,
            "git_blob_sha1": item.blob_id,
            "lfs_sha256": getattr(item.lfs, "sha256", None) if item.lfs is not None else None,
        }
    source.mkdir(parents=True, exist_ok=True)
    path = source / "roboverse_source_manifest.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump({"repository": repo, "revision": SOURCE_REVISION, "files": files}, stream, indent=2)
        stream.write("\n")
    return path


def _manifest(source: Path) -> dict:
    manifest = json.loads((source / "roboverse_source_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("repository") != SOURCE_REPOSITORY.rsplit("datasets/", 1)[-1]
        or manifest.get("revision") != SOURCE_REVISION
    ):
        raise ValueError("Source manifest does not identify the pinned SynthICL dataset")
    if not isinstance(manifest.get("files"), dict) or not manifest["files"]:
        raise ValueError("Source manifest contains no files")
    return manifest


def _source_file(root: Path, relative: str, manifest: dict) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or unsafe source path: {relative}")
    key = path.relative_to(root.resolve()).as_posix()
    info = manifest.get("files", {}).get(key)
    if not info or path.stat().st_size != info.get("size"):
        raise ValueError(f"Source file is missing from the pinned manifest or has the wrong size: {key}")
    if info.get("lfs_sha256"):
        digest = _sha256(path)
        expected = info["lfs_sha256"]
    else:
        sha = hashlib.sha1(usedforsecurity=False)
        sha.update(f"blob {path.stat().st_size}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(block)
        digest, expected = sha.hexdigest(), info.get("git_blob_sha1")
    if not expected or digest != expected:
        raise ValueError(f"Source file checksum disagrees with pinned manifest: {key}")
    return path


def verify_source(source: Path) -> dict:
    """Check all downloaded files against the official pinned-revision manifest."""
    source = source.resolve()
    manifest = _manifest(source)
    missing, invalid = [], []
    for name in manifest["files"]:
        if not (source / name).is_file():
            missing.append(name)
            continue
        try:
            _source_file(source, name, manifest)
        except ValueError:
            invalid.append(name)
    return {
        "expected": len(manifest["files"]),
        "verified": len(manifest["files"]) - len(missing) - len(invalid),
        "missing": missing,
        "invalid": invalid,
    }


def prepare(
    source: Path, output: Path, names: list[str], *, dtd_root: Path | None = None, texture: Path | None = None
) -> None:
    """Prepare graspable source objects, retaining original mesh scale and provenance."""
    if not names or len(set(names)) != len(names) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in names):
        raise ValueError("Provide unique names containing only letters, digits, underscores or hyphens")
    source = source.resolve()
    source_manifest_path = source / "roboverse_source_manifest.json"
    source_manifest = _manifest(source)
    if (dtd_root is None) != (texture is None):
        raise ValueError("--dtd-root and --texture must be provided together")
    texture_info = None
    if texture is not None:
        dtd_root = dtd_root.resolve()
        dtd_manifest = json.loads((dtd_root / "roboverse_source_manifest.json").read_text(encoding="utf-8"))
        if dtd_manifest.get("archive_md5") != DTD_MD5:
            raise ValueError("DTD source manifest does not identify the verified Oxford archive")
        image_root = (dtd_root / "extracted/dtd/images").resolve()
        texture = texture.resolve()
        if not texture.is_relative_to(image_root) or not texture.is_file() or texture.suffix.lower() != ".jpg":
            raise ValueError("Texture must be a JPEG inside the extracted DTD images directory")
        texture_info = {
            "source": "DTD r1.0.1",
            "path": texture.relative_to(image_root).as_posix(),
            "sha256": _sha256(texture),
            "source_manifest_sha256": _sha256(dtd_root / "roboverse_source_manifest.json"),
        }
    catalog = yaml.safe_load(
        _source_file(source, "graspable_configuration.yaml", source_manifest).read_text(encoding="utf-8")
    )
    entries = catalog.get("objects", [])
    if not isinstance(entries, list):
        raise ValueError("Invalid graspable object catalog")
    chosen = {}
    for name in names:
        matches = [entry for entry in entries if entry.get("name") == name]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one catalog entry for {name}")
        chosen[name] = matches[0]
    output.mkdir(parents=True, exist_ok=True)
    prefix = {
        name: f"synthicl_{name}" + (f"_dtd_{texture_info['sha256'][:10]}" if texture_info else "") for name in names
    }
    for name in names:
        if (output / prefix[name]).exists():
            raise FileExistsError(f"Refusing to replace existing asset: {name}")

    with tempfile.TemporaryDirectory(prefix="synthicl-assets-", dir=output) as temp:
        staging = Path(temp)
        for name, entry in chosen.items():
            bbox_half_extents = np.asarray(entry.get("bbox_size"), dtype=np.float64)
            grasp_angle = entry.get("grasp_angle")
            base_orientation = entry.get("base_orientation")
            if (
                bbox_half_extents.shape != (3,)
                or not np.isfinite(bbox_half_extents).all()
                or np.any(bbox_half_extents <= 0)
                or isinstance(grasp_angle, bool)
                or not isinstance(grasp_angle, (int, float))
                or not np.isfinite(grasp_angle)
                or not isinstance(base_orientation, dict)
                or not isinstance(base_orientation.get("z_negative"), bool)
            ):
                raise ValueError(
                    f"{name}: SynthICL task preparation requires bbox_size half extents, a finite grasp_angle, "
                    "and boolean base_orientation.z_negative"
                )
            mesh = _source_file(source, entry["path"], source_manifest)
            grasp_file = _source_file(source, entry["graspnet_pose_path"], source_manifest)
            grasps = np.load(grasp_file, allow_pickle=False)
            if grasps.ndim != 2 or grasps.shape[1] != 17 or len(grasps) == 0 or not np.isfinite(grasps).all():
                raise ValueError(f"{name}: expected nonempty finite GraspNet poses of shape (N, 17)")
            rotations = grasps[:, 4:13].reshape(-1, 3, 3)
            if not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-4):
                raise ValueError(f"{name}: GraspNet rotations are not orthonormal")
            if not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-4) or np.any(grasps[:, 3] < 0):
                raise ValueError(f"{name}: invalid GraspNet rotation determinant or grasp depth")
            material = _source_file(source, str(mesh.relative_to(source).parent / "material.mtl"), source_manifest)
            source_texture = _source_file(
                source, str(mesh.relative_to(source).parent / "material_0.png"), source_manifest
            )
            if "mtllib material.mtl" not in mesh.read_text(encoding="utf-8", errors="strict")[:4096]:
                raise ValueError(f"{name}: unsupported OBJ material reference")
            if "map_Kd material_0.png" not in material.read_text(encoding="utf-8"):
                raise ValueError(f"{name}: unsupported material texture reference")
            dest = staging / prefix[name]
            dest.mkdir()
            shutil.copy2(mesh, dest / "textured.obj")
            shutil.copy2(material, dest / "material.mtl")
            shutil.copy2(material, dest / "material_0.mtl")
            shutil.copy2(source_texture, dest / "material_0.png")
            if texture_info:
                with Image.open(texture) as image:
                    image.convert("RGB").save(dest / "material_0.png", format="PNG")
            shutil.copy2(grasp_file, dest / "grasps.npy")
            visual_mesh = trimesh.load(mesh, force="mesh", process=False)
            if visual_mesh.is_empty:
                raise ValueError(f"{name}: empty mesh")
            visual_mesh.export(dest / "collision.ply")
            _write_wrappers(dest, source_name=name, archive_sha256="derived-from-synthicl-obj")
            manifest_path = dest / "provenance.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.update({
                "source_repository": SOURCE_REPOSITORY,
                "source_revision": SOURCE_REVISION,
                "source_manifest_sha256": _sha256(source_manifest_path),
                "source_archive_sha256": None,
                "origin_files_sha256": {
                    str(path.relative_to(source).as_posix()): _sha256(path)
                    for path in (mesh, material, source_texture, grasp_file)
                },
                "source_files_note": "Adapter inputs; collision.ply is derived from the original visual mesh.",
                "original_scale": 1.0,
                "graspnet_count": len(grasps),
                "graspnet_path": "grasps.npy",
                "source_base_orientation": entry.get("base_orientation"),
                "bbox_half_extents_m": bbox_half_extents.tolist(),
                "grasp_angle_rad": float(grasp_angle),
                "appearance_texture": texture_info,
            })
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for name in names:
            (staging / prefix[name]).rename(output / prefix[name])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--objects", nargs="+")
    parser.add_argument("--dtd-root", type=Path, help="Verified DTD source directory")
    parser.add_argument("--texture", type=Path, help="JPEG inside the extracted DTD images directory")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-only", action="store_true", help="Verify all source files against the pinned manifest")
    mode.add_argument("--write-manifest", action="store_true", help="Fetch the official pinned-revision file manifest")
    args = parser.parse_args()
    if args.write_manifest:
        if args.output or args.objects or args.dtd_root or args.texture:
            parser.error("--write-manifest cannot be combined with --output or --objects")
        print(f"Saved pinned source manifest: {write_source_manifest(args.source)}")
        return
    if args.verify_only:
        if args.output or args.objects or args.dtd_root or args.texture:
            parser.error("--verify-only cannot be combined with --output or --objects")
        report = verify_source(args.source)
        print(json.dumps(report, indent=2))
        if report["missing"] or report["invalid"]:
            raise SystemExit(1)
        return
    if not args.output or not args.objects:
        parser.error("--output and --objects are required when preparing assets")
    prepare(args.source, args.output, args.objects, dtd_root=args.dtd_root, texture=args.texture)
    print(f"Prepared {len(args.objects)} SynthICL assets in {args.output}")


if __name__ == "__main__":
    main()
