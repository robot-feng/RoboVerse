"""Build the Phase 1 QA record and publish its review media in one command.

The manifest is the small hand-authored input. This tool hashes referenced
assets and previews, decodes preview media, builds a cross-backend overview,
then passes the generated QA record through the strict gallery publisher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from tools.icl.publish_phase1_preview import (
    REQUIRED_CHECKS,
    SUPPORTED_SIMULATORS,
    PreviewPublishError,
    publish,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SAFE_ROLE = re.compile(r"^[a-z][a-z0-9_-]*$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PreviewPublishError(f"manifest is missing required text field {key!r}")
    return value.strip()


def _relative_file(root: Path, relative: Any, *, label: str) -> tuple[Path, str]:
    if not isinstance(relative, str) or not relative.strip():
        raise PreviewPublishError(f"{label} must be a relative file path")
    path_part = Path(relative)
    if path_part.is_absolute() or any(part in {"", ".", ".."} for part in path_part.parts):
        raise PreviewPublishError(f"{label} has an unsafe relative path: {relative!r}")
    path = (root / path_part).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise PreviewPublishError(f"{label} escapes its allowed directory") from exc
    if not path.is_file() or path.stat().st_size == 0:
        raise PreviewPublishError(f"{label} is missing or empty: {relative}")
    return path, path_part.as_posix()


def _inspect_preview(path: Path, *, label: str) -> tuple[dict[str, Any], Any]:
    """Return preview metadata and a decoded first frame for the montage."""
    suffix = path.suffix.lower()
    if suffix == ".png":
        try:
            from PIL import Image

            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                frame = image.convert("RGB")
                width, height = frame.size
        except Exception as exc:
            raise PreviewPublishError(f"{label} is not a decodable PNG: {exc}") from exc
        if width < 1 or height < 1:
            raise PreviewPublishError(f"{label} has invalid PNG dimensions")
        return {"width": width, "height": height}, frame

    if suffix == ".mp4":
        try:
            import av
            from PIL import Image

            with av.open(str(path), mode="r") as container:
                streams = [stream for stream in container.streams if stream.type == "video"]
                if not streams:
                    raise ValueError("no video stream")
                first_frame = None
                width = height = 0
                decoded_frames = 0
                for frame in container.decode(streams[0]):
                    decoded_frames += 1
                    if first_frame is None:
                        array = frame.to_ndarray(format="rgb24")
                        height, width = array.shape[:2]
                        first_frame = Image.fromarray(array)
        except Exception as exc:
            raise PreviewPublishError(f"{label} is not a decodable MP4: {exc}") from exc
        if first_frame is None or width < 1 or height < 1 or decoded_frames < 2:
            raise PreviewPublishError(f"{label} must decode to at least two video frames")
        return {"width": width, "height": height, "decoded_frames": decoded_frames}, first_frame

    raise PreviewPublishError(f"{label} must be a PNG or MP4")


def _write_overview(frames: list[tuple[str, Any]], target: Path) -> tuple[int, int]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise PreviewPublishError("overview generation requires Pillow; install the RoboVerse `icl` extra") from exc
    if not frames:
        raise PreviewPublishError("manifest has no decoded backend previews for its overview")

    tile_width, tile_height, header_height = 640, 480, 42
    columns = min(3, len(frames))
    rows = (len(frames) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_width, rows * (tile_height + header_height)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (simulator, frame) in enumerate(frames):
        row, column = divmod(index, columns)
        x, y = column * tile_width, row * (tile_height + header_height)
        draw.rectangle((x, y, x + tile_width - 1, y + header_height - 1), fill=(245, 246, 248))
        draw.text((x + 12, y + 13), simulator, fill=(20, 26, 36), font=font)
        tile = Image.new("RGB", (tile_width, tile_height), (235, 235, 235))
        preview = frame.copy()
        preview.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
        offset = ((tile_width - preview.width) // 2, (tile_height - preview.height) // 2)
        tile.paste(preview, offset)
        canvas.paste(tile, (x, y + header_height))
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, format="PNG", optimize=True)
    return canvas.width, canvas.height


def _load_manifest(path: Path, qa_root: Path) -> dict[str, Any]:
    path = path.resolve()
    qa_root = qa_root.resolve()
    try:
        path.relative_to(qa_root)
    except ValueError as exc:
        raise PreviewPublishError("manifest JSON must be stored under the QA run root") from exc
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreviewPublishError(f"could not read Phase 1 manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != "roboverse.icl.phase1_environment_manifest.v1":
        raise PreviewPublishError("manifest format must be roboverse.icl.phase1_environment_manifest.v1")
    return manifest


def build_candidate(
    manifest_path: Path,
    *,
    qa_root: Path,
    asset_root: Path,
) -> Path:
    """Generate a hash-bound candidate and overview from a compact QA manifest."""
    manifest_path = manifest_path.resolve()
    qa_root = qa_root.resolve()
    asset_root = asset_root.resolve()
    manifest = _load_manifest(manifest_path, qa_root)
    environment_id = _required_text(manifest, "environment_id")
    if not _SAFE_ID.fullmatch(environment_id):
        raise PreviewPublishError(f"unsafe environment_id: {environment_id!r}")
    for key in (
        "source",
        "upstream_revision",
        "source_task_id",
        "source_config",
        "task_variant",
        "scene_id",
        "robot_id",
        "migrated_task_id",
    ):
        _required_text(manifest, key)
    roboverse_commit = _required_text(manifest, "roboverse_commit")
    if not _GIT_SHA.fullmatch(roboverse_commit):
        raise PreviewPublishError("roboverse_commit must be a full 40-character Git SHA")

    required_backends = manifest.get("required_backends")
    if not isinstance(required_backends, list) or not required_backends:
        raise PreviewPublishError("manifest must declare required_backends")
    backends = [value.lower() if isinstance(value, str) else "" for value in required_backends]
    if len(set(backends)) != len(backends) or any(value not in SUPPORTED_SIMULATORS for value in backends):
        raise PreviewPublishError("required_backends must be unique supported simulator names")

    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        raise PreviewPublishError("manifest must contain backend runs")
    seen: set[str] = set()
    candidate_runs = []
    montage_frames = []
    manifest_dir = manifest_path.parent
    for index, run in enumerate(runs):
        label = f"runs[{index}]"
        if not isinstance(run, dict):
            raise PreviewPublishError(f"{label} must be an object")
        simulator = _required_text(run, "simulator").lower()
        if simulator not in SUPPORTED_SIMULATORS or simulator in seen:
            raise PreviewPublishError(f"{label} has an unsupported or duplicate simulator")
        seen.add(simulator)
        simulator_version = _required_text(run, "simulator_version")
        seed = run.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise PreviewPublishError(f"{label}.seed must be an integer")
        checks = run.get("checks")
        if not isinstance(checks, dict):
            raise PreviewPublishError(f"{label} is missing Phase 1 checks")
        missing_or_failed = [check for check in REQUIRED_CHECKS if checks.get(check) is not True]
        if missing_or_failed:
            raise PreviewPublishError(f"{label} has incomplete or failed checks: {', '.join(missing_or_failed)}")

        asset_paths = run.get("asset_paths")
        if not isinstance(asset_paths, list) or not asset_paths:
            raise PreviewPublishError(f"{label}.asset_paths must list the assets used by this backend")
        asset_hashes: dict[str, str] = {}
        for relative in asset_paths:
            asset_path, normalized = _relative_file(asset_root, relative, label=f"{label}.asset_paths")
            if normalized in asset_hashes:
                raise PreviewPublishError(f"{label} repeats asset path {normalized!r}")
            asset_hashes[normalized] = _sha256(asset_path)

        evidence_paths = run.get("evidence_paths")
        if not isinstance(evidence_paths, list) or not evidence_paths:
            raise PreviewPublishError(f"{label}.evidence_paths must list QA reports or test logs")
        evidence_files = []
        for evidence_index, relative in enumerate(evidence_paths):
            evidence_path, normalized = _relative_file(
                manifest_dir,
                relative,
                label=f"{label}.evidence_paths[{evidence_index}]",
            )
            evidence_files.append({"path": normalized, "sha256": _sha256(evidence_path)})

        preview_specs = run.get("previews")
        if not isinstance(preview_specs, list) or not preview_specs:
            raise PreviewPublishError(f"{label}.previews must contain at least one PNG or MP4")
        previews = []
        roles: set[str] = set()
        chosen_overview = None
        for preview_index, spec in enumerate(preview_specs):
            preview_label = f"{label}.previews[{preview_index}]"
            if not isinstance(spec, dict):
                raise PreviewPublishError(f"{preview_label} must be an object")
            role = _required_text(spec, "role")
            if not _SAFE_ROLE.fullmatch(role) or role in roles:
                raise PreviewPublishError(f"{preview_label} has an unsafe or duplicate role")
            roles.add(role)
            preview_path, relative = _relative_file(manifest_dir, spec.get("path"), label=preview_label)
            media, frame = _inspect_preview(preview_path, label=preview_label)
            preview = {
                "role": role,
                "path": relative,
                "sha256": _sha256(preview_path),
                **media,
            }
            previews.append(preview)
            if role == "overview":
                chosen_overview = frame
            elif chosen_overview is None:
                chosen_overview = frame
        montage_frames.append((simulator, chosen_overview))
        candidate_runs.append({
            "simulator": simulator,
            "status": "VERIFIED",
            "seed": seed,
            "simulator_version": simulator_version,
            "roboverse_commit": roboverse_commit,
            "asset_hashes": asset_hashes,
            "checks": {check: checks[check] for check in REQUIRED_CHECKS},
            "evidence_files": evidence_files,
            "previews": previews,
        })

    if seen != set(backends):
        missing = sorted(set(backends) - seen)
        extra = sorted(seen - set(backends))
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected: {', '.join(extra)}")
        raise PreviewPublishError("backend runs do not match required_backends (" + "; ".join(details) + ")")

    overview_path = manifest_dir / "overview_first_frames.generated.png"
    width, height = _write_overview(montage_frames, overview_path)
    candidate = {
        "format": "roboverse.icl.phase1_environment_candidate.v1",
        "environment_id": environment_id,
        "source": manifest["source"],
        "upstream_revision": manifest["upstream_revision"],
        "source_task_id": manifest["source_task_id"],
        "source_config": manifest["source_config"],
        "task_variant": manifest["task_variant"],
        "scene_id": manifest["scene_id"],
        "robot_id": manifest["robot_id"],
        "migrated_task_id": manifest["migrated_task_id"],
        "roboverse_commit": roboverse_commit,
        "required_backends": backends,
        "source_manifest": {
            "path": manifest_path.relative_to(qa_root).as_posix(),
            "sha256": _sha256(manifest_path),
        },
        "combined_overview": {
            "path": overview_path.name,
            "sha256": _sha256(overview_path),
            "width": width,
            "height": height,
        },
        "runs": candidate_runs,
    }
    candidate_path = manifest_dir / "phase1_candidate.generated.json"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=manifest_dir, suffix=".tmp", delete=False
    ) as stream:
        temporary_path = Path(stream.name)
        json.dump(candidate, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    os.replace(temporary_path, candidate_path)
    return candidate_path


def publish_manifest(
    manifest_path: Path,
    *,
    qa_root: Path,
    asset_root: Path,
    output_root: Path,
) -> tuple[Path, Path]:
    """Generate the QA candidate and publish review media from one manifest."""
    candidate = build_candidate(manifest_path, qa_root=qa_root, asset_root=asset_root)
    destination = publish(candidate, qa_root=qa_root, output_root=output_root, asset_root=asset_root)
    return candidate, destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Phase 1 manifest JSON stored under qa_runs")
    parser.add_argument("--qa-root", type=Path, default=ROOT / "examples" / "output" / "qa_runs")
    parser.add_argument("--asset-root", type=Path, default=ROOT / "roboverse_data")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "examples" / "output" / "ICL_Benchmark" / "environments",
    )
    args = parser.parse_args()
    try:
        candidate, output = publish_manifest(
            args.manifest,
            qa_root=args.qa_root,
            asset_root=args.asset_root,
            output_root=args.output_root,
        )
    except PreviewPublishError as exc:
        parser.error(str(exc))
    print(f"QA candidate: {candidate}")
    print(f"Review materials: {output}")


if __name__ == "__main__":
    main()
