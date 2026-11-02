"""Publish preview media for a scene that passed the Phase 1 environment gate.

The QA candidate JSON and detailed logs stay under ``examples/output/qa_runs``.
The human-review gallery receives only PNG/GIF/MP4 previews and a short README.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from tools.icl.video_utils import write_preview_gif

SUPPORTED_SIMULATORS = ("mujoco", "genesis", "newton")
REQUIRED_CHECKS = (
    "asset_resolution",
    "create",
    "repeated_reset",
    "step",
    "finite_state",
    "task_predicate_positive",
    "task_predicate_negative",
    "viewer_interactive",
    "rendered_preview",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SAFE_ROLE = re.compile(r"^[a-z][a-z0-9_-]*$")


class PreviewPublishError(ValueError):
    """Raised when a candidate is not safe to include in the review gallery."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PreviewPublishError(f"candidate is missing required text field {key!r}")
    return value.strip()


def _candidate_file(candidate_dir: Path, record: dict[str, Any], *, label: str) -> Path:
    relative = record.get("path")
    digest = record.get("sha256")
    if not isinstance(relative, str) or not relative.strip():
        raise PreviewPublishError(f"{label} is missing a relative path")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise PreviewPublishError(f"{label} is missing a lowercase SHA-256")
    path_part = Path(relative)
    if path_part.is_absolute() or any(part in {"", ".", ".."} for part in path_part.parts):
        raise PreviewPublishError(f"{label} has an unsafe relative path: {relative!r}")
    path = (candidate_dir / path_part).resolve()
    try:
        path.relative_to(candidate_dir)
    except ValueError as exc:
        raise PreviewPublishError(f"{label} escapes the QA candidate directory") from exc
    if not path.is_file() or path.stat().st_size == 0:
        raise PreviewPublishError(f"{label} is missing or empty: {relative}")
    if _sha256(path) != digest:
        raise PreviewPublishError(f"{label} SHA-256 does not match its validation record")
    return path


def _asset_file(asset_root: Path, relative: str, digest: str, *, label: str) -> Path:
    asset_path = Path(relative)
    if (
        asset_path.is_absolute()
        or any(part in {"", ".", ".."} for part in asset_path.parts)
        or not isinstance(digest, str)
        or not _SHA256.fullmatch(digest)
    ):
        raise PreviewPublishError(f"{label} has an unsafe path or invalid SHA-256")
    path = (asset_root / asset_path).resolve()
    try:
        path.relative_to(asset_root)
    except ValueError as exc:
        raise PreviewPublishError(f"{label} escapes the canonical asset directory") from exc
    if not path.is_file() or _sha256(path) != digest:
        raise PreviewPublishError(f"{label} is missing or its SHA-256 does not match")
    return path


def _validate_png(path: Path, record: dict[str, Any], *, label: str) -> None:
    if path.suffix.lower() != ".png":
        raise PreviewPublishError(f"{label} must be a PNG")
    header = path.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise PreviewPublishError(f"{label} is not a valid PNG image")
    width, height = struct.unpack(">II", header[16:24])
    if width < 1 or height < 1:
        raise PreviewPublishError(f"{label} has invalid PNG dimensions")
    if record.get("width") != width or record.get("height") != height:
        raise PreviewPublishError(f"{label} dimensions do not match the validation record")


def _validate_mp4(path: Path, record: dict[str, Any], *, label: str) -> None:
    if path.suffix.lower() != ".mp4":
        raise PreviewPublishError(f"{label} must be an MP4")
    with path.open("rb") as stream:
        header = stream.read(4096)
    if b"ftyp" not in header:
        raise PreviewPublishError(f"{label} has no MP4 ftyp box")
    width, height, frame_count = (record.get(key) for key in ("width", "height", "decoded_frames"))
    if not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in (width, height)):
        raise PreviewPublishError(f"{label} has no positive decoded dimensions")
    if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count < 2:
        raise PreviewPublishError(f"{label} must record at least two decoded frames")


def _validate_candidate(
    candidate_path: Path,
    qa_root: Path,
    asset_root: Path,
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], list[tuple[dict[str, Any], Path]]]], Path]:
    candidate_path = candidate_path.resolve()
    qa_root = qa_root.resolve()
    if not candidate_path.is_file():
        raise PreviewPublishError(f"candidate JSON does not exist: {candidate_path}")
    try:
        candidate_path.relative_to(qa_root)
    except ValueError as exc:
        raise PreviewPublishError("candidate JSON must be stored under the QA run root") from exc
    try:
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreviewPublishError(f"could not read Phase 1 candidate JSON: {exc}") from exc
    if not isinstance(candidate, dict) or candidate.get("format") != "roboverse.icl.phase1_environment_candidate.v1":
        raise PreviewPublishError("candidate format must be roboverse.icl.phase1_environment_candidate.v1")

    environment_id = _required_text(candidate, "environment_id")
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
        _required_text(candidate, key)
    roboverse_commit = _required_text(candidate, "roboverse_commit")
    if not _GIT_SHA.fullmatch(roboverse_commit):
        raise PreviewPublishError("roboverse_commit must be a full 40-character Git SHA")

    source_manifest = candidate.get("source_manifest")
    if source_manifest is not None:
        if not isinstance(source_manifest, dict):
            raise PreviewPublishError("source_manifest must include a relative path and SHA-256")
        relative = source_manifest.get("path")
        digest = source_manifest.get("sha256")
        source_path = Path(relative) if isinstance(relative, str) else Path()
        if (
            not isinstance(relative, str)
            or source_path.is_absolute()
            or any(part in {"", ".", ".."} for part in source_path.parts)
            or not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
        ):
            raise PreviewPublishError("source_manifest has an unsafe path or invalid SHA-256")
        source_path = (qa_root / source_path).resolve()
        try:
            source_path.relative_to(qa_root)
        except ValueError as exc:
            raise PreviewPublishError("source_manifest escapes the QA run root") from exc
        if not source_path.is_file() or _sha256(source_path) != digest:
            raise PreviewPublishError("source_manifest is missing or its SHA-256 does not match")

    required_backends = candidate.get("required_backends")
    if not isinstance(required_backends, list) or not required_backends:
        raise PreviewPublishError("candidate must declare its required_backends")
    normalized_backends = [backend.lower() if isinstance(backend, str) else "" for backend in required_backends]
    if len(set(normalized_backends)) != len(normalized_backends) or any(
        backend not in SUPPORTED_SIMULATORS for backend in normalized_backends
    ):
        raise PreviewPublishError("required_backends must be unique supported simulator names")

    combined_overview = candidate.get("combined_overview")
    if not isinstance(combined_overview, dict):
        raise PreviewPublishError("candidate must include a combined_overview PNG")
    overview_path = _candidate_file(candidate_path.parent, combined_overview, label="combined_overview")
    _validate_png(overview_path, combined_overview, label="combined_overview")

    runs = candidate.get("runs")
    if not isinstance(runs, list) or not runs:
        raise PreviewPublishError("candidate must include at least one verified backend run")
    seen_simulators: set[str] = set()
    validated_runs = []
    for index, run in enumerate(runs):
        label = f"runs[{index}]"
        if not isinstance(run, dict) or run.get("status") != "VERIFIED":
            raise PreviewPublishError(f"{label} is not marked VERIFIED")
        simulator = _required_text(run, "simulator").lower()
        if simulator not in SUPPORTED_SIMULATORS:
            raise PreviewPublishError(f"{label} has unsupported simulator {simulator!r}")
        if simulator in seen_simulators:
            raise PreviewPublishError(f"candidate repeats simulator {simulator!r}")
        seen_simulators.add(simulator)
        if run.get("roboverse_commit") != roboverse_commit:
            raise PreviewPublishError(f"{label} commit differs from the candidate commit")
        _required_text(run, "simulator_version")
        seed = run.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise PreviewPublishError(f"{label} seed must be an integer")
        asset_hashes = run.get("asset_hashes")
        if not isinstance(asset_hashes, dict) or not asset_hashes:
            raise PreviewPublishError(f"{label} must bind at least one asset hash")
        for relative_path, digest in asset_hashes.items():
            if not isinstance(relative_path, str):
                raise PreviewPublishError(f"{label} asset paths must be strings")
            _asset_file(asset_root, relative_path, digest, label=f"{label}.asset_hashes[{relative_path!r}]")
        checks = run.get("checks")
        if not isinstance(checks, dict):
            raise PreviewPublishError(f"{label} is missing Phase 1 checks")
        missing_or_failed = [check for check in REQUIRED_CHECKS if checks.get(check) is not True]
        if missing_or_failed:
            raise PreviewPublishError(f"{label} has incomplete or failed checks: {', '.join(missing_or_failed)}")

        evidence_files = run.get("evidence_files")
        if not isinstance(evidence_files, list) or not evidence_files:
            raise PreviewPublishError(f"{label} must bind at least one QA evidence file")
        for evidence_index, evidence in enumerate(evidence_files):
            if not isinstance(evidence, dict):
                raise PreviewPublishError(f"{label}.evidence_files[{evidence_index}] must be an object")
            _candidate_file(
                candidate_path.parent,
                evidence,
                label=f"{label}.evidence_files[{evidence_index}]",
            )

        previews = run.get("previews")
        if not isinstance(previews, list) or not previews:
            raise PreviewPublishError(f"{label} has no rendered previews")
        seen_roles: set[str] = set()
        validated_previews = []
        for preview_index, preview in enumerate(previews):
            preview_label = f"{label}.previews[{preview_index}]"
            if not isinstance(preview, dict):
                raise PreviewPublishError(f"{preview_label} must be an object")
            role = _required_text(preview, "role")
            if not _SAFE_ROLE.fullmatch(role) or role in seen_roles:
                raise PreviewPublishError(f"{preview_label} has an unsafe or duplicate camera role")
            seen_roles.add(role)
            preview_path = _candidate_file(candidate_path.parent, preview, label=preview_label)
            suffix = preview_path.suffix.lower()
            if suffix == ".png":
                _validate_png(preview_path, preview, label=preview_label)
            elif suffix == ".mp4":
                _validate_mp4(preview_path, preview, label=preview_label)
            else:
                raise PreviewPublishError(f"{preview_label} must be a PNG or MP4")
            validated_previews.append((preview, preview_path))
        validated_runs.append((run, validated_previews))

    if seen_simulators != set(normalized_backends):
        missing = sorted(set(normalized_backends) - seen_simulators)
        extra = sorted(seen_simulators - set(normalized_backends))
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected: {', '.join(extra)}")
        raise PreviewPublishError("backend runs do not match required_backends (" + "; ".join(details) + ")")

    return candidate, validated_runs, overview_path


def _readme(
    candidate: dict[str, Any],
    runs: list[tuple[dict[str, Any], list[tuple[dict[str, Any], Path]]]],
    candidate_path: Path,
    qa_root: Path,
    gif_previews: dict[str, str],
) -> str:
    lines = [
        f"# {candidate['environment_id']}",
        "",
        "**Phase 1 environment: VERIFIED.** The scene runtime, task predicate positive/negative cases, and Viewer interaction were checked. These previews do not claim a successful manipulation rollout.",
        "",
        f"- Source: `{candidate['source']}` (`{candidate['upstream_revision']}`)",
        f"- Source task/config: `{candidate['source_task_id']}` / `{candidate['source_config']}`",
        f"- Variant and scene: `{candidate['task_variant']}` / `{candidate['scene_id']}`",
        f"- RoboVerse task and robot: `{candidate['migrated_task_id']}` / `{candidate['robot_id']}`",
        f"- RoboVerse commit: `{candidate['roboverse_commit']}`",
        "",
    ]
    for run, previews in runs:
        simulator = run["simulator"]
        lines.extend([f"## {simulator}", ""])
        evidence_links = [f"`{item['path']}`" for item in run["evidence_files"]]
        lines.extend(["QA evidence: " + " · ".join(evidence_links), ""])
        gif_role = gif_previews.get(simulator)
        if gif_role is not None:
            lines.extend([f"![{simulator} scene preview]({simulator}/{simulator}_{gif_role}.gif)", ""])
        else:
            overview_still = next(
                (
                    (item, source)
                    for item, source in previews
                    if item["role"] == "overview" and source.suffix.lower() == ".png"
                ),
                None,
            )
            if overview_still is not None:
                item, _ = overview_still
                lines.extend([f"![{simulator} scene preview]({simulator}/{simulator}_{item['role']}.png)", ""])
        links = [
            f"[{item['role']} {source.suffix[1:].upper()}]({simulator}/{simulator}_{item['role']}{source.suffix.lower()})"
            for item, source in previews
        ]
        lines.extend(["Preview streams: " + " · ".join(links), ""])
        lines.append(f"Version: `{run['simulator_version']}`; seed: `{run['seed']}`.")
        lines.append("")
    qa_relative = candidate_path.resolve().relative_to(qa_root.resolve()).as_posix()
    lines.extend([
        "",
        "Assets, validation details, reports, and logs remain in the QA run record:",
        f"`examples/output/qa_runs/{qa_relative}`.",
    ])
    if candidate.get("source_manifest") is not None:
        lines.extend([
            f"Source manifest: `examples/output/qa_runs/{candidate['source_manifest']['path']}`.",
        ])
    lines.append("")
    return "\n".join(lines)


def publish(candidate_path: Path, *, qa_root: Path, output_root: Path, asset_root: Path | None = None) -> Path:
    """Copy verified media and generate compact Markdown-friendly scene GIFs."""
    candidate, runs, overview_path = _validate_candidate(
        candidate_path,
        qa_root,
        (asset_root or ROOT / "roboverse_data").resolve(),
    )
    output_root = output_root.resolve()
    environment_id = candidate["environment_id"]
    destination = output_root / environment_id
    if destination.exists():
        raise PreviewPublishError(f"refusing to overwrite existing review previews: {destination}")
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{environment_id}.", dir=output_root) as temporary:
        stage = Path(temporary) / environment_id
        stage.mkdir()
        shutil.copy2(overview_path, stage / "overview_first_frames.png")
        gif_previews: dict[str, str] = {}
        for run, previews in runs:
            simulator = run["simulator"]
            simulator_dir = stage / simulator
            simulator_dir.mkdir()
            for preview, source in previews:
                shutil.copy2(source, simulator_dir / f"{simulator}_{preview['role']}{source.suffix.lower()}")
            overview_video = next(
                (
                    (preview, source)
                    for preview, source in previews
                    if preview["role"] == "overview" and source.suffix.lower() == ".mp4"
                ),
                None,
            )
            if overview_video is None:
                overview_video = next(
                    ((preview, source) for preview, source in previews if source.suffix.lower() == ".mp4"),
                    None,
                )
            if overview_video is not None:
                preview, source = overview_video
                gif_path = simulator_dir / f"{simulator}_{preview['role']}.gif"
                try:
                    write_preview_gif(source, gif_path)
                except (OSError, ValueError, RuntimeError, ImportError) as exc:
                    raise PreviewPublishError(
                        f"could not make {simulator} Markdown GIF from {preview['role']}: {exc}"
                    ) from exc
                gif_previews[simulator] = preview["role"]
        readme = _readme(candidate, runs, Path(candidate_path), qa_root, gif_previews)
        (stage / "README.md").write_text(readme, encoding="utf-8")
        os.replace(stage, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate", type=Path, required=True, help="Verified Phase 1 environment JSON stored under qa_runs"
    )
    parser.add_argument("--qa-root", type=Path, default=ROOT / "examples" / "output" / "qa_runs")
    parser.add_argument("--asset-root", type=Path, default=ROOT / "roboverse_data")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "examples" / "output" / "ICL_Benchmark" / "environments",
    )
    args = parser.parse_args()
    print(publish(args.candidate, qa_root=args.qa_root, output_root=args.output_root, asset_root=args.asset_root))


if __name__ == "__main__":
    main()
