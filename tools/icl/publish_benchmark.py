"""Publish only ICL runs with task, replay, video and backend-parity evidence.

The SQLite queue remains the source of verification state. This module is the
single release gate that turns a fully verified demo into the clean
``examples/output/ICL_Benchmark`` gallery. Reports and failed runs stay in the
QA run tree and are never copied into the published directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from tools.icl.benchmark_queue import KEY_FIELDS, list_jobs

_GATE_PHASES = ("smoke", "semantics", "demo", "parity")
_GATE_STATUS = {
    "smoke": "smoke_passed",
    "semantics": "semantics_passed",
    "demo": "demo_verified",
    "parity": "parity_passed",
}
_SOURCE_DIRECTORY = {
    "imitator_game": "The-Imitator-Game",
    "imitator-game": "The-Imitator-Game",
    "the-imitator-game": "The-Imitator-Game",
    "the_imitator_game": "The-Imitator-Game",
    "synthicl": "SynthICL",
    "geniesim": "GenieSim",
}
_SOURCE_LICENSES = {
    "imitator_game": {"MIT"},
    "the-imitator-game": {"MIT"},
    "the_imitator_game": {"MIT"},
    "imitator-game": {"MIT"},
    "synthicl": {"Apache-2.0"},
    "geniesim": {"MPL-2.0"},
}
_SOURCE_URLS = {
    "imitator_game": {"https://github.com/imitator-game/The-Imitator-Game"},
    "the-imitator-game": {"https://github.com/imitator-game/The-Imitator-Game"},
    "the_imitator_game": {"https://github.com/imitator-game/The-Imitator-Game"},
    "imitator-game": {"https://github.com/imitator-game/The-Imitator-Game"},
    "synthicl": {"https://github.com/sNiper-Qian/SynthICL"},
    "geniesim": {"https://github.com/AgibotTech/genie_sim"},
}
_SUPPORTED_SPDX_LICENSES = {
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "CC-BY-4.0",
    "CC-BY-NC-4.0",
    "CC-BY-NC-ND-4.0",
    "CC-BY-NC-SA-4.0",
    "CC-BY-SA-4.0",
    "CC0-1.0",
    "MIT",
    "MPL-2.0",
    "Unlicense",
}


class PublishError(ValueError):
    """Raised when a candidate cannot be safely released."""


def _read_result(job: dict[str, Any]) -> dict[str, Any]:
    """Load a queue result only when its recorded hash still matches."""
    result_path = Path(job.get("result_path") or "")
    if not result_path.is_file() or result_path.stat().st_size == 0:
        raise PublishError("registered result JSON is missing or empty")
    payload = result_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != job.get("result_sha256"):
        raise PublishError("registered result JSON hash does not match the queue")
    try:
        result = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublishError("registered result JSON is invalid") from exc
    if not isinstance(result, dict):
        raise PublishError("registered result must be a JSON object")
    if any(result.get(key) != job.get(key) for key in KEY_FIELDS):
        raise PublishError("registered result provenance differs from the queue")
    if result.get("job_id") != job.get("job_id") or result.get("artifact_dir") != job.get("artifact_dir"):
        raise PublishError("registered result job or artifact directory differs from the queue")
    result["_queue_result_sha256"] = job["result_sha256"]
    return result


def _gate_key(job: dict[str, Any]) -> tuple[Any, ...]:
    """Identify the same seed/embodiment/backend across verification phases."""
    return tuple(job[field] for field in KEY_FIELDS if field != "phase")


def _safe_components(value: str) -> tuple[str, ...]:
    """Split a manifest path component into safe output-directory names."""
    if not value or "\\" in value or value.startswith("/") or re.match(r"^[A-Za-z]:", value):
        raise PublishError(f"unsafe output path component: {value!r}")
    parts = tuple(value.split("/"))
    if any(part in {"", ".", ".."} or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", part) for part in parts):
        raise PublishError(f"unsafe output path component: {value!r}")
    return parts


def _artifact_path(job: dict[str, Any], relative: Any, *, role: str) -> Path:
    """Resolve one required artifact and prove it stays inside its job directory."""
    if not isinstance(relative, str) or not relative.strip():
        raise PublishError(f"demo result is missing its {role} artifact path")
    relative_path = Path(relative)
    if relative_path.is_absolute() or any(part in {"..", ""} for part in relative_path.parts):
        raise PublishError(f"unsafe {role} artifact path: {relative!r}")
    root = Path(job["artifact_dir"]).resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PublishError(f"{role} artifact escapes the job directory") from exc
    if not candidate.is_file() or candidate.stat().st_size == 0:
        raise PublishError(f"{role} artifact is missing or empty: {relative}")
    return candidate


def _verified_demo_artifacts(job: dict[str, Any], result: dict[str, Any]) -> tuple[dict[str, Path], dict[str, str]]:
    """Verify artifact hashes before files can support a release or parity claim."""
    artifacts = result.get("artifacts")
    hashes = result.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not isinstance(hashes, dict):
        raise PublishError("demo result must bind video, thumbnail and trajectory to SHA-256 hashes")
    paths = {}
    verified = {}
    for role in ("video", "thumbnail", "trajectory"):
        path = _artifact_path(job, artifacts.get(role), role=role)
        digest = hashes.get(role)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise PublishError(f"demo result has no SHA-256 for its {role}")
        actual = _sha256(path)
        if actual != digest:
            raise PublishError(f"demo {role} changed after its verification result was recorded")
        paths[role] = path
        verified[role] = actual
    return paths, verified


def _inspect_video(path: Path) -> dict[str, Any]:
    """Decode an MP4 stream and return its dimensions and decoded frame count."""
    if path.suffix.lower() != ".mp4":
        raise PublishError("demo video must be an MP4 file")
    try:
        import av

        with av.open(str(path), mode="r") as container:
            streams = [stream for stream in container.streams if stream.type == "video"]
            if len(streams) != 1:
                raise PublishError(f"expected one video stream, found {len(streams)}")
            stream = streams[0]
            width, height = int(stream.width), int(stream.height)
            frame_count = sum(1 for _ in container.decode(stream))
            duration = (
                float(stream.duration * stream.time_base)
                if stream.duration is not None and stream.time_base is not None
                else float(container.duration / av.time_base)
                if container.duration is not None
                else 0.0
            )
    except PublishError:
        raise
    except Exception as exc:
        raise PublishError(f"could not decode demo video: {exc}") from exc
    if width < 1 or height < 1 or frame_count < 2 or not math.isfinite(duration) or duration <= 0:
        raise PublishError("demo video has invalid dimensions, duration, or fewer than two decodable frames")
    return {"width": width, "height": height, "decoded_frames": frame_count, "duration_seconds": duration}


def _inspect_thumbnail(path: Path) -> dict[str, int]:
    """Decode a PNG thumbnail and return its dimensions."""
    if path.suffix.lower() != ".png":
        raise PublishError("thumbnail must be a PNG file")
    try:
        from PIL import Image

        with Image.open(path) as image:
            if image.format != "PNG":
                raise PublishError("thumbnail must contain PNG encoded data")
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
    except Exception as exc:
        raise PublishError(f"could not decode thumbnail: {exc}") from exc
    if width < 1 or height < 1:
        raise PublishError("thumbnail dimensions must be positive")
    return {"width": width, "height": height}


def _inspect_trajectory(path: Path) -> dict[str, Any]:
    """Validate finite actions and time-aligned state/observation records."""
    suffix = path.suffix.lower()
    action_count = 0
    state_count = 0
    try:
        if suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            actions = value.get("actions") if isinstance(value, dict) else None
            action_count = len(actions) if isinstance(actions, list) else 0
            states = (
                next(
                    (
                        value[key]
                        for key in ("trace", "states", "robot_states", "observations")
                        if isinstance(value.get(key), list)
                    ),
                    None,
                )
                if isinstance(value, dict)
                else None
            )
            state_count = len(states) if states is not None else 0
            action_values = []
            state_values = []

            def collect_numbers(item: Any, target: list[float]) -> None:
                if isinstance(item, dict):
                    for nested in item.values():
                        collect_numbers(nested, target)
                elif isinstance(item, list):
                    for nested in item:
                        collect_numbers(nested, target)
                elif isinstance(item, (int, float)) and not isinstance(item, bool):
                    target.append(float(item))

            collect_numbers(actions, action_values)
            collect_numbers(states, state_values)
            if not action_values or not all(math.isfinite(number) for number in action_values):
                raise PublishError("JSON trajectory actions must contain finite numeric values")
            if not state_values or not all(math.isfinite(number) for number in state_values):
                raise PublishError("JSON trajectory state/observation records must contain finite numeric values")
        elif suffix == ".npz":
            import numpy as np

            with np.load(path, allow_pickle=False) as archive:
                action_key = next((key for key in ("actions", "action", "joint_actions") if key in archive), None)
                if action_key is None:
                    raise PublishError("trajectory NPZ has no actions array")
                actions = archive[action_key]
                action_count = int(actions.shape[0]) if actions.ndim else 0
                if not np.issubdtype(actions.dtype, np.number) or not np.isfinite(actions).all():
                    raise PublishError("trajectory NPZ actions must be finite numeric values")
                state_key = next(
                    (key for key in ("states", "robot_states", "observations", "images_front") if key in archive), None
                )
                if state_key is None:
                    raise PublishError("trajectory NPZ has no state or observation sequence")
                states = archive[state_key]
                state_count = int(states.shape[0]) if states.ndim else 0
                if state_count < action_count:
                    raise PublishError("trajectory NPZ state/observation sequence is shorter than its actions")
                if np.issubdtype(states.dtype, np.number) and not np.isfinite(states).all():
                    raise PublishError("trajectory NPZ states contain non-finite numeric values")
        elif suffix in {".h5", ".hdf5"}:
            import h5py
            import numpy as np

            with h5py.File(path, "r") as archive:
                action_datasets = []
                state_datasets = []

                def collect_datasets(name: str, item: Any) -> None:
                    if not isinstance(item, h5py.Dataset):
                        return
                    tail = name.rsplit("/", 1)[-1].lower()
                    if tail in {"actions", "action", "joint_actions"}:
                        action_datasets.append(item)
                    elif tail in {"states", "robot_states", "observations", "images_front"}:
                        state_datasets.append(item)

                archive.visititems(collect_datasets)
                action_count = max((int(item.shape[0]) if item.ndim else 0 for item in action_datasets), default=0)
                state_count = max((int(item.shape[0]) if item.ndim else 0 for item in state_datasets), default=0)
                if not action_datasets or not state_datasets:
                    raise PublishError("trajectory HDF5 must contain action and state/observation datasets")
                if state_count < action_count:
                    raise PublishError("trajectory HDF5 state/observation sequence is shorter than its actions")
                for dataset in action_datasets:
                    if not dataset.ndim or dataset.dtype.kind not in "biufc":
                        raise PublishError("trajectory HDF5 actions must be numeric")
                    if dataset.dtype.kind in "fc":
                        for start in range(0, dataset.shape[0], 256):
                            if not np.isfinite(dataset[start : start + 256]).all():
                                raise PublishError("trajectory HDF5 actions contain non-finite values")
                for dataset in state_datasets:
                    if dataset.dtype.kind in "fc":
                        for start in range(0, dataset.shape[0], 32):
                            if not np.isfinite(dataset[start : start + 32]).all():
                                raise PublishError("trajectory HDF5 state/observation contains non-finite values")
        else:
            raise PublishError("trajectory must be JSON, NPZ, H5 or HDF5")
    except PublishError:
        raise
    except Exception as exc:
        raise PublishError(f"could not validate trajectory: {exc}") from exc
    if action_count < 1 or state_count < action_count:
        raise PublishError("trajectory must contain actions and an aligned state/observation sequence")
    return {"format": suffix.lstrip("."), "action_count": action_count, "state_count": state_count}


def _validate_semantics(result: dict[str, Any]) -> dict[str, Any]:
    """Require task success or explicit trajectory validity for random exploration."""
    validation = result.get("validation")
    if not isinstance(validation, dict) or validation.get("passed") is not True:
        raise PublishError("semantics result has no passing validation record")
    kind = validation.get("kind")
    if kind == "task_success":
        if (
            validation.get("success") is not True
            or not isinstance(validation.get("criterion"), str)
            or not validation["criterion"].strip()
            or not isinstance(validation.get("observed"), dict)
            or not validation["observed"]
        ):
            raise PublishError("task-success validation must record success and its criterion")
    elif kind == "trajectory_validity":
        checks = validation.get("checks")
        if (
            validation.get("valid") is not True
            or not isinstance(checks, list)
            or not checks
            or any(not isinstance(check, dict) or check.get("passed") is not True for check in checks)
        ):
            raise PublishError("random-exploration validation must pass explicit trajectory checks")
        required_checks = {"finite_actions", "within_action_bounds", "timebase_aligned"}
        check_names = [check.get("name") for check in checks]
        if any(not isinstance(name, str) or not name.strip() for name in check_names):
            raise PublishError("random-exploration validity checks need nonempty names")
        if not required_checks.issubset(check_names):
            raise PublishError(
                f"random-exploration validation is missing checks: {sorted(required_checks - check_names)}"
            )
        if validation.get("success") is True:
            raise PublishError("random exploration cannot be labeled as task success")
    else:
        raise PublishError("semantics validation kind must be task_success or trajectory_validity")
    return validation


def _validate_parity(
    result: dict[str, Any], *, expected_result_hashes: dict[str, str], expected_trajectory_hashes: dict[str, str]
) -> dict[str, Any]:
    """Require measured comparison evidence across at least two backends."""
    evidence = result.get("parity_evidence")
    if not isinstance(evidence, dict) or evidence.get("passed") is not True:
        raise PublishError("parity result has no passing comparison evidence")
    backends = evidence.get("compared_backends")
    metrics = evidence.get("metrics")
    if (
        not isinstance(backends, list)
        or not all(isinstance(backend, str) for backend in backends)
        or len(set(backends)) < 2
        or any(backend not in {"mujoco", "genesis", "newton"} for backend in backends)
        or not isinstance(metrics, dict)
        or not metrics
    ):
        raise PublishError("parity evidence must compare two target backends and record measured metrics")
    if evidence.get("compared_demo_result_sha256") != expected_result_hashes:
        raise PublishError("parity evidence is not bound to the compared demo result hashes")
    if evidence.get("compared_trajectory_sha256") != expected_trajectory_hashes:
        raise PublishError("parity evidence is not bound to the compared trajectory hashes")
    for name, measurement in metrics.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(measurement, dict)
            or type(measurement.get("value")) not in {int, float}
            or type(measurement.get("tolerance")) not in {int, float}
            or not math.isfinite(measurement["value"])
            or not math.isfinite(measurement["tolerance"])
            or measurement["tolerance"] < 0
            or abs(measurement["value"]) > measurement["tolerance"]
        ):
            raise PublishError(f"parity metric {name!r} lacks a passing measured value/tolerance pair")
    return evidence


def _validate_source_provenance(job: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Validate pinned repository and per-asset provenance for a demo result."""
    source = job["source"].lower()
    source_directory = _SOURCE_DIRECTORY.get(source)
    if source_directory is None:
        raise PublishError(f"unknown benchmark source: {job['source']!r}")
    license_id = result.get("source_license")
    if not isinstance(license_id, str) or license_id not in _SOURCE_LICENSES.get(source, set()):
        raise PublishError("demo license must match the verified SPDX license for its benchmark source")
    source_url = result.get("source_url")
    if not isinstance(source_url, str) or source_url not in _SOURCE_URLS.get(source, set()):
        raise PublishError("demo source URL must match the pinned upstream repository")
    asset_hashes = result.get("asset_hashes")
    if (
        not isinstance(asset_hashes, dict)
        or not asset_hashes
        or any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in asset_hashes.values()
        )
    ):
        raise PublishError("demo result must record SHA-256 hashes for its source assets")
    asset_records = result.get("asset_records")
    if not isinstance(asset_records, dict) or set(asset_records) != set(asset_hashes):
        raise PublishError("demo result must record source URL and SPDX license for every hashed asset")
    for name, record in asset_records.items():
        if (
            not isinstance(record, dict)
            or record.get("sha256") != asset_hashes[name]
            or record.get("license") not in _SUPPORTED_SPDX_LICENSES
            or not isinstance(record.get("source_url"), str)
            or not record["source_url"].startswith("https://")
        ):
            raise PublishError(f"asset {name!r} has incomplete or unsupported source/license provenance")
    if any(not re.fullmatch(r"[0-9a-f]{40,64}", job[field]) for field in ("upstream_sha", "roboverse_sha")):
        raise PublishError("demo result must pin upstream and RoboVerse revisions")
    return {
        "source_directory": source_directory,
        "source_license": license_id,
        "source_url": source_url,
        "asset_hashes": asset_hashes,
        "asset_records": asset_records,
    }


def inspect_candidates(connection: sqlite3.Connection) -> dict[str, Any]:
    """Return fully checked demo candidates and explicit rejection reasons."""
    jobs = list_jobs(connection)
    by_key_and_phase = {
        (tuple(job[field] for field in KEY_FIELDS if field != "phase"), job["phase"]): job for job in jobs
    }
    grouped_counts: dict[tuple[str, str, str], int] = {}
    for job in jobs:
        key = (job["source"], job["phase"], job["status"])
        grouped_counts[key] = grouped_counts.get(key, 0) + 1
    eligible, rejected = [], []
    for demo in jobs:
        if demo["phase"] != "demo" or demo["status"] != "demo_verified":
            continue
        try:
            gates = {}
            for phase in _GATE_PHASES:
                gate = by_key_and_phase.get((_gate_key(demo), phase))
                if gate is None or gate["status"] != _GATE_STATUS[phase]:
                    raise PublishError(f"required {phase} gate is missing or not passed")
                gates[phase] = _read_result(gate)
            smoke_job = by_key_and_phase[(_gate_key(demo), "smoke")]
            semantics_job = by_key_and_phase[(_gate_key(demo), "semantics")]
            demo_job = by_key_and_phase[(_gate_key(demo), "demo")]
            parity_job = by_key_and_phase[(_gate_key(demo), "parity")]
            if smoke_job["job_id"] not in semantics_job["dependencies"]:
                raise PublishError("semantics gate does not depend on its smoke result")
            if semantics_job["job_id"] not in demo_job["dependencies"]:
                raise PublishError("demo gate does not depend on its semantics result")
            if demo_job["job_id"] not in parity_job["dependencies"]:
                raise PublishError("parity gate does not depend on this backend's demo")
            reference_demos = [
                job
                for job in jobs
                if job["phase"] == "demo"
                and job["status"] == "demo_verified"
                and job["job_id"] in parity_job["dependencies"]
                and job["backend"] != demo["backend"]
                and all(job[field] == demo[field] for field in KEY_FIELDS if field not in {"backend", "phase"})
            ]
            if not reference_demos:
                raise PublishError("parity gate has no dependency on a matching demo from another backend")
            result = gates["demo"]
            for phase in ("smoke", "semantics"):
                if gates[phase].get("asset_hashes") != result.get("asset_hashes"):
                    raise PublishError(f"{phase} and demo results do not use identical source asset hashes")
            artifact_paths, artifact_hashes = _verified_demo_artifacts(demo_job, result)
            semantics = _validate_semantics(gates["semantics"])
            success_evidence = result.get("success_evidence")
            if (
                not isinstance(success_evidence, dict)
                or success_evidence.get("passed") is not True
                or not isinstance(success_evidence.get("observed"), dict)
                or not success_evidence["observed"]
            ):
                raise PublishError("demo result has no passing outcome evidence")
            if success_evidence != semantics:
                raise PublishError("demo outcome differs from the semantics gate's checker result")
            video_info = _inspect_video(artifact_paths["video"])
            thumbnail_info = _inspect_thumbnail(artifact_paths["thumbnail"])
            trajectory_info = _inspect_trajectory(artifact_paths["trajectory"])
            source = demo["source"].lower()
            provenance = _validate_source_provenance(demo, result)
            license_id = provenance["source_license"]
            source_url = provenance["source_url"]
            asset_hashes = provenance["asset_hashes"]
            asset_records = provenance["asset_records"]
            source_directory = provenance["source_directory"]
            task_parts = _safe_components(demo["task_id"])
            variant_parts = _safe_components(demo["variant"])
            scene_parts = _safe_components(demo["scene_id"])
            embodiment_parts = _safe_components(demo["embodiment"])
            backend_parts = _safe_components(demo["backend"])
            expected_result_hashes = {demo_job["job_id"]: demo_job["result_sha256"]}
            expected_trajectory_hashes = {demo_job["job_id"]: artifact_hashes["trajectory"]}
            expected_backends = {demo["backend"]}
            for reference in reference_demos:
                reference_gate_key = _gate_key(reference)
                reference_smoke = by_key_and_phase.get((reference_gate_key, "smoke"))
                reference_semantics = by_key_and_phase.get((reference_gate_key, "semantics"))
                if (
                    reference_smoke is None
                    or reference_smoke["status"] != "smoke_passed"
                    or reference_semantics is None
                    or reference_semantics["status"] != "semantics_passed"
                ):
                    raise PublishError("parity reference demo is missing its smoke or semantics gate")
                reference_result = _read_result(reference)
                reference_smoke_result = _read_result(reference_smoke)
                reference_semantics_result = _read_result(reference_semantics)
                if reference_smoke["job_id"] not in reference_semantics["dependencies"]:
                    raise PublishError("parity reference semantics gate does not depend on its smoke result")
                if reference_semantics["job_id"] not in reference["dependencies"]:
                    raise PublishError("parity reference demo does not depend on its semantics result")
                if reference_smoke_result.get("asset_hashes") != reference_result.get(
                    "asset_hashes"
                ) or reference_semantics_result.get("asset_hashes") != reference_result.get("asset_hashes"):
                    raise PublishError("parity reference verification gates use different source asset hashes")
                _validate_source_provenance(reference, reference_result)
                if reference_result.get("success_evidence") != _validate_semantics(reference_semantics_result):
                    raise PublishError("parity reference demo differs from its semantics gate")
                reference_paths, reference_hashes = _verified_demo_artifacts(reference, reference_result)
                _inspect_video(reference_paths["video"])
                _inspect_thumbnail(reference_paths["thumbnail"])
                _inspect_trajectory(reference_paths["trajectory"])
                expected_result_hashes[reference["job_id"]] = reference["result_sha256"]
                expected_trajectory_hashes[reference["job_id"]] = reference_hashes["trajectory"]
                expected_backends.add(reference["backend"])
            parity = _validate_parity(
                gates["parity"],
                expected_result_hashes=expected_result_hashes,
                expected_trajectory_hashes=expected_trajectory_hashes,
            )
            if set(parity["compared_backends"]) != expected_backends:
                raise PublishError("parity evidence backend set differs from its verified demo dependencies")
            relative_parts = (
                source_directory,
                *task_parts,
                *variant_parts,
                *scene_parts,
                *embodiment_parts,
                *backend_parts,
            )
            eligible.append({
                "job": demo,
                "result": result,
                "gates": gates,
                "success_evidence": success_evidence,
                "source_license": license_id,
                "source_url": source_url,
                "asset_hashes": asset_hashes,
                "asset_records": asset_records,
                "destination_parts": relative_parts,
                "artifacts": artifact_paths,
                "artifact_sha256": artifact_hashes,
                "artifact_info": {
                    "video": video_info,
                    "thumbnail": thumbnail_info,
                    "trajectory": trajectory_info,
                },
                "parity_evidence": parity,
            })
        except (PublishError, OSError) as exc:
            rejected.append({"job_id": demo["job_id"], "task_id": demo["task_id"], "reason": str(exc)})
    destinations = [item["destination_parts"] for item in eligible]
    if len(set(destinations)) != len(destinations):
        raise PublishError("two eligible jobs map to the same published directory")
    queue_counts = [
        {"source": source, "phase": phase, "status": status, "count": count}
        for (source, phase, status), count in sorted(grouped_counts.items())
    ]
    return {
        "eligible": eligible,
        "rejected": rejected,
        "demo_candidate_count": sum(j["phase"] == "demo" for j in jobs),
        "queue_summary": {"enqueued_job_units": len(jobs), "counts": queue_counts},
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stage_publication(
    candidates: list[dict[str, Any]], output: Path, queue_summary: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    """Build a complete clean gallery in a sibling staging directory."""
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage.", dir=output.parent))
    index_entries = []
    try:
        for candidate in candidates:
            job = candidate["job"]
            target = stage.joinpath(*candidate["destination_parts"])
            target.mkdir(parents=True, exist_ok=False)
            files = {
                "demo_video": (candidate["artifacts"]["video"], "demo.mp4"),
                "thumbnail": (candidate["artifacts"]["thumbnail"], "thumbnail.png"),
                "trajectory": (
                    candidate["artifacts"]["trajectory"],
                    f"trajectory{candidate['artifacts']['trajectory'].suffix.lower()}",
                ),
            }
            output_artifacts = {}
            for role, (source_path, filename) in files.items():
                destination = target / filename
                shutil.copy2(source_path, destination)
                source_role = "video" if role == "demo_video" else role
                copied_sha256 = _sha256(destination)
                expected_sha256 = candidate["artifact_sha256"][source_role]
                if copied_sha256 != expected_sha256:
                    raise PublishError(f"{source_role} changed while the gallery was being staged")
                output_artifacts[role] = {
                    "path": filename,
                    "sha256": copied_sha256,
                    "size_bytes": destination.stat().st_size,
                    **candidate["artifact_info"][
                        "video" if role == "demo_video" else "thumbnail" if role == "thumbnail" else "trajectory"
                    ],
                }
            verification = {
                "format": "roboverse.icl.published_demo.v1",
                "status": "published",
                "test_unit": {field: job[field] for field in KEY_FIELDS},
                "job_id": job["job_id"],
                "source_license": candidate["source_license"],
                "source_url": candidate["source_url"],
                "asset_hashes": candidate["asset_hashes"],
                "asset_records": candidate["asset_records"],
                "source_artifact_sha256": candidate["artifact_sha256"],
                "execution_command": candidate["result"]["command"],
                "success_evidence": candidate["success_evidence"],
                "parity_evidence": candidate["parity_evidence"],
                "phase_results": {
                    phase: {
                        "job_id": candidate["gates"][phase]["job_id"],
                        "status": candidate["gates"][phase]["status"],
                        "result_sha256": candidate["gates"][phase]["_queue_result_sha256"],
                    }
                    for phase in _GATE_PHASES
                },
                "artifacts": output_artifacts,
            }
            _write_json(target / "verification.json", verification)
            index_entries.append({
                **verification["test_unit"],
                "job_id": job["job_id"],
                "path": "/".join(candidate["destination_parts"]),
                "artifacts": output_artifacts,
            })

        index = {
            "format": "roboverse.icl.published_index.v1",
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "published_count": len(index_entries),
            "queue_summary": queue_summary,
            "items": index_entries,
        }
        _write_json(stage / "index.json", index)
        source_counts: dict[str, int] = {}
        for entry in index_entries:
            source_counts[entry["source"]] = source_counts.get(entry["source"], 0) + 1
        lines = [
            "# RoboVerse ICL Benchmark — verified demos",
            "",
            "This directory contains only demos that passed recorded smoke, semantics, demo, and cross-backend parity gates.",
            "Random exploration is labeled as trajectory validity and is not counted as task success.",
            "",
            f"Published test-unit count: {len(index_entries)}",
            "",
            "| Source | Published units |",
            "| --- | ---: |",
        ]
        lines.extend(f"| {source} | {count} |" for source, count in sorted(source_counts.items()))
        lines.extend((
            "",
            f"Queue jobs enqueued: {queue_summary['enqueued_job_units']} (this is the QA work-unit denominator, not upstream task coverage).",
            "",
            "| Source | Phase | Status | Jobs |",
            "| --- | --- | --- | ---: |",
        ))
        lines.extend(
            f"| {item['source']} | {item['phase']} | {item['status']} | {item['count']} |"
            for item in queue_summary["counts"]
        )
        lines.extend((
            "",
            "Automated publication does not imply human approval; use the ICL Inspector to record human review separately.",
            "See `index.json` for source SHAs, backends, seeds, evidence, artifact hashes and replay paths.",
            "",
        ))
        (stage / "README.md").write_text("\n".join(lines), encoding="utf-8")
        return stage, index
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def publish(
    candidates: list[dict[str, Any]], output_root: Path, *, queue_summary: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Atomically replace the published gallery while preserving its previous generation."""
    if not candidates:
        raise PublishError("no demo passed all publish gates; output directory was left unchanged")
    output = Path(output_root).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage, index = _stage_publication(candidates, output, queue_summary or {"enqueued_job_units": 0, "counts": []})
    backup = None
    try:
        if output.exists():
            backup = output.parent / f".{output.name}.previous.{time.strftime('%Y%m%dT%H%M%S')}.{uuid.uuid4().hex[:8]}"
            os.replace(output, backup)
        try:
            os.replace(stage, output)
        except Exception:
            if backup is not None and backup.exists() and not output.exists():
                os.replace(backup, output)
            raise
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
    return {
        "output": str(output),
        "published_count": index["published_count"],
        "backup": str(backup) if backup is not None else None,
        "index": index,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("examples/output/qa_runs/icl_benchmark_queue.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("examples/output/ICL_Benchmark"))
    parser.add_argument(
        "--publish", action="store_true", help="write the clean gallery; default only reports eligibility"
    )
    args = parser.parse_args()
    database = args.db.resolve()
    if not database.is_file():
        raise SystemExit(f"Verification queue database does not exist: {database}")
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        audit = inspect_candidates(connection)
    finally:
        connection.close()
    summary = {
        "output": str(args.output.resolve()),
        "demo_candidate_count": audit["demo_candidate_count"],
        "eligible_count": len(audit["eligible"]),
        "queue_summary": audit["queue_summary"],
        "rejected": audit["rejected"],
        "mode": "publish" if args.publish else "dry_run",
    }
    if args.publish:
        summary["publication"] = publish(audit["eligible"], args.output, queue_summary=audit["queue_summary"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
