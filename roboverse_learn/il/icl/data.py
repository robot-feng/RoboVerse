"""Store independent demonstrations and expose pairs to the SynthICL loader.

Raw trajectories use ``.hdf5`` and pair files use ``.h5``. The latter contain
relative HDF5 links, so a recursively scanned training directory does not
accidentally treat raw episodes as paired tasks or duplicate their images.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

IMAGE_KEYS = ("images_front", "images_wrist", "images_side")
MASK_KEYS = {key: f"valid_mask_{key.removeprefix('images_')}" for key in IMAGE_KEYS}
ACTION_SPEC = "tcp_world_delta_xyz_rotvec_gripper_open_v1"
HEAD_CAMERA_ROLES = {
    "images_front": "head_rgb",
    "images_wrist": "head_fisheye_left",
    "images_side": "head_fisheye_right",
}
_METADATA_KEYS = {
    "origin_id",
    "task",
    "goal",
    "split",
    "object_id",
    "asset_sha256",
    "simulator",
    "control_period",
    "camera_roles",
    "success",
    "physical_record",
    "action_spec",
    "phase_schema",
}


def _metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(metadata)
    missing = _METADATA_KEYS - result.keys()
    if missing:
        raise ValueError(f"Episode metadata missing fields: {sorted(missing)}")
    for key in ("origin_id", "task", "goal", "object_id", "physical_record"):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f"Episode metadata {key} must be a non-empty string")
    if result["split"] not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation or test")
    if result["simulator"] not in {"mujoco", "genesis", "newton"}:
        raise ValueError("simulator must be mujoco, genesis or newton")
    if not re.fullmatch(r"[0-9a-f]{64}", str(result["asset_sha256"])):
        raise ValueError("asset_sha256 must be a lowercase SHA-256 digest")
    if result["action_spec"] != ACTION_SPEC:
        raise ValueError(f"Expected action_spec {ACTION_SPEC!r}")
    if not np.isfinite(result["control_period"]) or result["control_period"] <= 0:
        raise ValueError("control_period must be positive and finite")
    if result["success"] is not True:
        raise ValueError("Only successful physical rollouts can be exported as demonstrations")
    phases = result["phase_schema"]
    if (
        not isinstance(phases, list)
        or not 2 <= len(phases) <= 6
        or not all(isinstance(phase, str) and phase for phase in phases)
        or len(set(phases)) != len(phases)
    ):
        raise ValueError("phase_schema must contain two to six distinct phase names")
    roles = result["camera_roles"]
    if not isinstance(roles, dict) or set(roles) != set(IMAGE_KEYS) or not all(roles.values()):
        raise ValueError(f"camera_roles must describe each of {IMAGE_KEYS}")
    # Reject non-JSON values before creating a file.
    json.dumps(result, sort_keys=True, allow_nan=False)
    return result


def _validated_numeric_arrays(robot_states, actions, waypoints_idx, assigned_wps) -> dict[str, np.ndarray]:
    actions = np.asarray(actions, dtype=np.float32)
    states = np.asarray(robot_states, dtype=np.float32)
    waypoints = np.asarray(waypoints_idx)
    assigned = np.asarray(assigned_wps)
    if actions.ndim != 2 or actions.shape[1] != 7 or actions.shape[0] < 2:
        raise ValueError("actions must have shape (T, 7), including a terminal placeholder")
    count = len(actions)
    if states.shape != (count, 7) or not np.isfinite(actions).all() or not np.isfinite(states).all():
        raise ValueError("robot_states and actions must be finite arrays of shape (T, 7)")
    if not np.allclose(actions[-1, :6], 0) or not np.isclose(actions[-1, 6], actions[-2, 6]):
        raise ValueError("The terminal action must be a zero motion retaining the previous gripper command")
    if np.any(np.linalg.norm(actions[:, :3], axis=1) > 0.021001):
        raise ValueError("Translation action exceeds 0.021 m per step")
    if np.any(np.linalg.norm(actions[:, 3:6], axis=1) > 0.210001):
        raise ValueError("Rotation action exceeds 0.21 rad per step")
    if np.any((actions[:, 6] < 0) | (actions[:, 6] > 1)):
        raise ValueError("Gripper command must be in [0, 1]")
    if waypoints.ndim != 1 or not np.issubdtype(waypoints.dtype, np.integer) or not 2 <= len(waypoints) <= 6:
        raise ValueError("waypoints_idx must contain two to six integer frame indices")
    if waypoints[0] != 0 or waypoints[-1] != count - 1 or np.any(np.diff(waypoints) <= 0):
        raise ValueError("waypoints_idx must increase strictly from the first through the terminal frame")
    if assigned.shape != (count,) or not np.issubdtype(assigned.dtype, np.integer):
        raise ValueError("assigned_wps must contain one integer index per frame")
    if np.any((assigned < 0) | (assigned >= len(waypoints))) or np.any(np.diff(assigned) < 0):
        raise ValueError("assigned_wps must be nondecreasing indices into the waypoint list")
    return {"robot_states": states, "actions": actions, "waypoints_idx": waypoints, "assigned_wps": assigned}


def _validated_arrays(
    images: Mapping[str, np.ndarray], robot_states, actions, waypoints_idx, assigned_wps
) -> dict[str, np.ndarray]:
    arrays = _validated_numeric_arrays(robot_states, actions, waypoints_idx, assigned_wps)
    count = len(arrays["actions"])
    if set(images) != set(IMAGE_KEYS):
        raise ValueError(f"Expected images for {IMAGE_KEYS}")
    for key in IMAGE_KEYS:
        value = np.asarray(images[key])
        if (value.dtype != np.uint8 or value.ndim != 4 or value.shape[0] != count or value.shape[-1] != 3
                or min(value.shape[1:3]) < 1):
            raise ValueError(f"{key} must be uint8 RGB with shape (T, H, W, 3)")
        arrays[key] = value
    return arrays


def image_preprocessing_plan(
    source_hw: tuple[int, int], target_hw: tuple[int, int] = (256, 256), *, mode: str = "letterbox"
) -> dict[str, Any]:
    """Describe a camera resize using pixel-center coordinates and explicit padding/cropping.

    The 3x3 affine maps original pixel centers to output pixel centers, matching
    PyTorch bilinear interpolation with ``align_corners=False``. Sizes use
    ``(height, width)``; padding and cropping use left/top/right/bottom order.
    """
    if mode not in {"stretch", "letterbox", "center_crop"}:
        raise ValueError("mode must be stretch, letterbox or center_crop")
    for label, size in (("source_hw", source_hw), ("target_hw", target_hw)):
        if (not isinstance(size, (tuple, list)) or len(size) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1 for v in size)):
            raise ValueError(f"{label} must contain two positive integers")
    source_h, source_w = (int(v) for v in source_hw)
    target_h, target_w = (int(v) for v in target_hw)
    if mode == "stretch":
        resized_h, resized_w = target_h, target_w
    else:
        ratio = (min if mode == "letterbox" else max)(target_h / source_h, target_w / source_w)
        resized_h = max(1, math.floor(source_h * ratio + 0.5))
        resized_w = max(1, math.floor(source_w * ratio + 0.5))
        if mode == "letterbox":
            resized_h, resized_w = min(resized_h, target_h), min(resized_w, target_w)
        else:
            resized_h, resized_w = max(resized_h, target_h), max(resized_w, target_w)
    pad_left = max(target_w - resized_w, 0) // 2
    pad_top = max(target_h - resized_h, 0) // 2
    pad_right = max(target_w - resized_w, 0) - pad_left
    pad_bottom = max(target_h - resized_h, 0) - pad_top
    crop_left = max(resized_w - target_w, 0) // 2
    crop_top = max(resized_h - target_h, 0) // 2
    crop_right = max(resized_w - target_w, 0) - crop_left
    crop_bottom = max(resized_h - target_h, 0) - crop_top
    sx, sy = resized_w / source_w, resized_h / source_h
    affine = [[sx, 0.0, (sx - 1) / 2 + pad_left - crop_left],
              [0.0, sy, (sy - 1) / 2 + pad_top - crop_top], [0.0, 0.0, 1.0]]
    return {
        "mode": mode,
        "source_hw": [source_h, source_w],
        "target_hw": [target_h, target_w],
        "resized_hw": [resized_h, resized_w],
        "padding_ltrb": [pad_left, pad_top, pad_right, pad_bottom],
        "crop_ltrb": [crop_left, crop_top, crop_right, crop_bottom],
        "source_to_output_pixel_affine": affine,
        "interpolation": "pytorch_bilinear_antialias_align_corners_false",
        "padding_rgb": [0, 0, 0],
        "lens_distortion": "unchanged; no undistortion or fisheye correction",
        "output_projection": "apply the input projection and distortion first, then the recorded pixel affine",
    }


def preprocess_rgb_images(images: np.ndarray, plan: Mapping[str, Any]) -> np.ndarray:
    """Apply an image plan to a uint8 RGB batch and return uint8 model inputs."""
    import torch
    import torch.nn.functional as F

    value = np.asarray(images)
    if (value.dtype != np.uint8 or value.ndim != 4 or value.shape[-1] != 3
            or list(value.shape[1:3]) != plan.get("source_hw")):
        raise ValueError("RGB frames must match the plan's source (T, H, W, 3) uint8 shape")
    tensor = torch.from_numpy(np.ascontiguousarray(value)).permute(0, 3, 1, 2).float()
    resized = tuple(plan["resized_hw"])
    if tuple(tensor.shape[-2:]) != resized:
        tensor = F.interpolate(tensor, size=resized, mode="bilinear", align_corners=False, antialias=True)
    left, top, right, bottom = plan["crop_ltrb"]
    tensor = tensor[:, :, top : resized[0] - bottom, left : resized[1] - right]
    left, top, right, bottom = plan["padding_ltrb"]
    if any((left, top, right, bottom)):
        tensor = F.pad(tensor, (left, right, top, bottom), value=0)
    if tuple(tensor.shape[-2:]) != tuple(plan["target_hw"]):
        raise ValueError("Image plan does not produce the requested target dimensions")
    return tensor.round().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()


def preprocess_valid_mask(mask: np.ndarray, plan: Mapping[str, Any]) -> np.ndarray:
    """Conservatively keep pixels whose interpolation support is entirely valid."""
    import torch
    import torch.nn.functional as F

    value = np.asarray(mask)
    if value.dtype != np.bool_ or list(value.shape) != plan.get("source_hw"):
        raise ValueError("valid mask must be bool with the plan's source (H, W) shape")
    tensor = torch.from_numpy(np.ascontiguousarray(value)).float()[None, None]
    resized = tuple(plan["resized_hw"])
    if tuple(value.shape) != resized:
        tensor = F.interpolate(tensor, size=resized, mode="bilinear", align_corners=False, antialias=True)
    left, top, right, bottom = plan["crop_ltrb"]
    tensor = tensor[:, :, top : resized[0] - bottom, left : resized[1] - right]
    left, top, right, bottom = plan["padding_ltrb"]
    tensor = F.pad(tensor, (left, right, top, bottom), value=0)
    return (tensor[0, 0] >= 1.0 - 1e-6).numpy()


def _plan_calibration(plan: dict[str, Any], calibration: Any) -> None:
    if calibration is None:
        plan["calibration"] = {"status": "not_provided", "output_intrinsic": None}
        return
    if not isinstance(calibration, Mapping):
        raise ValueError("Each camera calibration must be a mapping")
    record = dict(calibration)
    if record.get("calibration_source") == "borrowed_g2" and record.get("intrinsic_provenance") == "physical_calibration":
        raise ValueError("G2 calibration borrowed for G1 is not physical G1 calibration")
    # Preserve projection type, distortion and source-specific fields exactly.
    json.dumps(record, allow_nan=False)
    intrinsic = record.get("intrinsic")
    if intrinsic is None:
        plan["calibration"] = {"status": "intrinsic_not_provided", "input": record, "output_intrinsic": None}
        return
    if record.get("intrinsic_provenance") not in {"renderer_reported", "source_nominal", "physical_calibration"}:
        raise ValueError("A supplied intrinsic needs explicit renderer_reported, source_nominal or physical_calibration provenance")
    matrix = np.asarray(intrinsic, dtype=np.float64)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1])):
        raise ValueError("camera intrinsic must be a finite 3x3 matrix with positive focal lengths")
    plan["calibration"] = {
        "status": "transformed_input_intrinsic",
        "input": record,
        "output_intrinsic": (np.asarray(plan["source_to_output_pixel_affine"]) @ matrix).tolist(),
    }


def _mask_sources(metadata: Mapping[str, Any]) -> dict[str, Any]:
    records = metadata.get("camera_valid_masks", {})
    if not isinstance(records, dict) or set(records) - set(IMAGE_KEYS):
        raise ValueError("camera_valid_masks must be keyed by image dataset names")
    for key, record in records.items():
        if not isinstance(record, dict) or not isinstance(record.get("source"), str) or not record["source"]:
            raise ValueError(f"A valid mask source must be specified for {key}")
    return records


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess_trajectory(
    source: Path,
    destination: Path,
    *,
    target_hw: tuple[int, int] = (256, 256),
    mode: str = "letterbox",
    batch_size: int = 8,
) -> Path:
    """Create a traceable SynthICL-size copy while retaining the original native-resolution file.

    This only resamples RGB pixels; it never rectifies lens distortion. The
    source SHA-256 and each camera's actual affine and optional calibration
    provenance are saved in the output metadata. Input files are not modified.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source.suffix != ".hdf5" or destination.suffix != ".hdf5":
        raise ValueError("Source and destination must be distinct .hdf5 trajectory files")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    initial_digest = _sha256_file(source)
    metadata = read_metadata(source)
    if "image_preprocessing" in metadata:
        raise ValueError("Preprocess the original raw trajectory rather than resampling an already processed copy")
    calibrations = metadata.get("camera_calibrations", {})
    if not isinstance(calibrations, dict) or set(calibrations) - set(IMAGE_KEYS):
        raise ValueError("camera_calibrations must be a mapping keyed by the image dataset names")
    mask_sources = _mask_sources(metadata)
    temporary = _temporary_file(destination)
    try:
        with h5py.File(source, "r") as src, h5py.File(temporary, "w") as out:
            trajectory = src["trajectory"]
            expected_keys = {*IMAGE_KEYS, *MASK_KEYS.values(), "robot_states", "actions", "waypoints_idx", "assigned_wps"}
            extras = set(trajectory.keys()) - expected_keys
            if extras:
                raise ValueError(f"Cannot preprocess unrecognized trajectory fields without a transform: {sorted(extras)}")
            if any(MASK_KEYS[key] not in trajectory for key in mask_sources):
                raise ValueError("camera_valid_masks metadata references a missing mask")
            numeric = _validated_numeric_arrays(*(trajectory[name][:] for name in (
                "robot_states", "actions", "waypoints_idx", "assigned_wps"
            )))
            if len(metadata["phase_schema"]) != len(numeric["waypoints_idx"]):
                raise ValueError("phase_schema must describe every recorded waypoint")
            count = len(numeric["actions"])
            target = out.create_group("trajectory")
            for name, value in trajectory.attrs.items():
                target.attrs[name] = value
            for name, value in src.attrs.items():
                if name != "metadata_json":
                    out.attrs[name] = value
            for name, value in numeric.items():
                target.create_dataset(name, data=value, compression="gzip", compression_opts=1)
            plans = {}
            for key in IMAGE_KEYS:
                dataset = trajectory[key]
                if (dataset.dtype != np.uint8 or dataset.ndim != 4 or dataset.shape[0] != count
                        or dataset.shape[-1] != 3):
                    raise ValueError(f"{key} must be uint8 RGB with shape (T, H, W, 3)")
                plan = image_preprocessing_plan(tuple(dataset.shape[1:3]), target_hw, mode=mode)
                _plan_calibration(plan, calibrations.get(key))
                mask_key = MASK_KEYS[key]
                if mask_key in trajectory:
                    original_mask = trajectory[mask_key][:]
                    mask_source = mask_sources.get(key, {}).get("source", "unspecified")
                else:
                    original_mask = np.ones(dataset.shape[1:3], dtype=np.bool_)
                    mask_source = "all_source_pixels_assumed_valid"
                transformed_mask = preprocess_valid_mask(original_mask, plan)
                target.create_dataset(mask_key, data=transformed_mask, compression="gzip", compression_opts=1)
                plan["valid_mask"] = {"dataset": mask_key, "source": mask_source,
                                      "transform": "conservative_bilinear_support_and_invalid_padding"}
                plans[key] = plan
                output = target.create_dataset(
                    key, shape=(count, *plan["target_hw"], 3), dtype=np.uint8,
                    compression="gzip", compression_opts=1, chunks=(1, *plan["target_hw"], 3),
                )
                for start in range(0, count, batch_size):
                    end = min(start + batch_size, count)
                    output[start:end] = preprocess_rgb_images(dataset[start:end], plan)
            native_sizes = {key: plans[key]["source_hw"] for key in IMAGE_KEYS}
            if "camera_native_hw" in metadata and metadata["camera_native_hw"] != native_sizes:
                raise ValueError("camera_native_hw does not match the raw trajectory")
            metadata["camera_native_hw"] = native_sizes
            metadata["image_preprocessing"] = {
                "format": "roboverse.icl.image_preprocessing.v1",
                "mode": mode,
                "target_hw": plans[IMAGE_KEYS[0]]["target_hw"],
                "quantization": "round_to_uint8_after_resize",
                "raw_source": {"relative_path": os.path.relpath(source, destination.parent), "sha256": initial_digest},
                "cameras": plans,
            }
            out.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True, allow_nan=False)
        if _sha256_file(source) != initial_digest:
            raise RuntimeError("Raw trajectory changed during preprocessing; output discarded")
        temporary.rename(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _temporary_file(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to replace {destination}")
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    return Path(name)


def write_trajectory(
    path: Path,
    *,
    metadata: Mapping[str, Any],
    images: Mapping[str, np.ndarray],
    robot_states,
    actions,
    waypoints_idx,
    assigned_wps,
    valid_masks: Mapping[str, np.ndarray] | None = None,
) -> Path:
    """Write a validated raw trajectory in the format expected by the training adapter."""
    path = Path(path)
    if path.suffix != ".hdf5":
        raise ValueError("Raw trajectory files must end in .hdf5")
    metadata = _metadata(metadata)
    arrays = _validated_arrays(images, robot_states, actions, waypoints_idx, assigned_wps)
    mask_sources = _mask_sources(metadata)
    if valid_masks is not None:
        if not isinstance(valid_masks, Mapping) or set(valid_masks) - set(IMAGE_KEYS):
            raise ValueError("valid_masks must be keyed by image dataset names")
        for key, mask in valid_masks.items():
            value = np.asarray(mask)
            if value.dtype != np.bool_ or value.shape != arrays[key].shape[1:3]:
                raise ValueError(f"{key} valid mask must be bool with its native (H, W) shape")
            arrays[MASK_KEYS[key]] = value
    if any(MASK_KEYS[key] not in arrays for key in mask_sources):
        raise ValueError("camera_valid_masks metadata references a missing mask")
    native_sizes = {key: list(arrays[key].shape[1:3]) for key in IMAGE_KEYS}
    if "camera_native_hw" in metadata and metadata["camera_native_hw"] != native_sizes:
        raise ValueError("camera_native_hw does not match the recorded image arrays")
    metadata["camera_native_hw"] = native_sizes
    if len(metadata["phase_schema"]) != len(arrays["waypoints_idx"]):
        raise ValueError("phase_schema must describe every recorded waypoint")
    temporary = _temporary_file(path)
    try:
        with h5py.File(temporary, "w") as h5:
            h5.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True, allow_nan=False)
            group = h5.create_group("trajectory")
            for key, value in arrays.items():
                group.create_dataset(key, data=value, compression="gzip", compression_opts=1)
        temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def read_metadata(path: Path) -> dict[str, Any]:
    """Read and validate private provenance without putting it into model inputs."""
    with h5py.File(path, "r") as h5:
        if "trajectory" not in h5:
            raise ValueError(f"Missing trajectory group: {path}")
        return _metadata(json.loads(h5.attrs["metadata_json"]))


def preprocessing_signature(metadata: Mapping[str, Any]) -> tuple | None:
    """Return the shared image policy, excluding camera-specific sizes and calibration."""
    preprocessing = metadata.get("image_preprocessing")
    if preprocessing is None:
        return None
    if not isinstance(preprocessing, Mapping) or preprocessing.get("format") != "roboverse.icl.image_preprocessing.v1":
        raise ValueError("Unsupported image preprocessing metadata")
    target = preprocessing.get("target_hw")
    mode = preprocessing.get("mode")
    image_preprocessing_plan((1, 1), target, mode=mode)
    if preprocessing.get("quantization") != "round_to_uint8_after_resize":
        raise ValueError("Unsupported preprocessed image quantization")
    cameras = preprocessing.get("cameras")
    if not isinstance(cameras, dict) or set(cameras) != set(IMAGE_KEYS):
        raise ValueError("Image preprocessing must describe all three cameras")
    native = metadata.get("camera_native_hw")
    if (not isinstance(native, dict) or any(not isinstance(cameras[key], dict)
            or native.get(key) != cameras[key].get("source_hw") for key in IMAGE_KEYS)):
        raise ValueError("camera_native_hw must agree with camera preprocessing plans")
    for key, plan in cameras.items():
        if not isinstance(plan, dict):
            raise ValueError(f"Malformed image preprocessing plan for {key}")
        expected = image_preprocessing_plan(plan.get("source_hw"), target, mode=mode)
        for field, value in expected.items():
            if field == "source_to_output_pixel_affine":
                actual = np.asarray(plan.get(field), dtype=np.float64)
                if actual.shape != (3, 3) or not np.allclose(actual, value, atol=1e-12, rtol=0):
                    raise ValueError(f"Invalid image preprocessing affine for {key}")
            elif plan.get(field) != value:
                raise ValueError(f"Invalid image preprocessing {field} for {key}")
        calibration = plan.get("calibration")
        if not isinstance(calibration, dict):
            raise ValueError(f"Missing calibration provenance for {key}")
        _plan_calibration(expected, calibration.get("input"))
        if calibration != expected["calibration"]:
            raise ValueError(f"Invalid transformed calibration for {key}")
        mask = plan.get("valid_mask")
        if (not isinstance(mask, dict) or mask.get("dataset") != MASK_KEYS[key]
                or not isinstance(mask.get("source"), str) or not mask["source"]
                or mask.get("transform") != "conservative_bilinear_support_and_invalid_padding"):
            raise ValueError(f"Invalid valid mask provenance for {key}")
    return mode, tuple(target), preprocessing["quantization"]


def resolve_physical_record(trajectory: Path, metadata: Mapping[str, Any], *, library: Path | None = None) -> Path:
    """Resolve the original physical record of a raw or preprocessed trajectory.

    A preprocessed copy references its raw source by relative path and SHA-256;
    the source must remain within ``library`` when that boundary is provided.
    """
    trajectory = Path(trajectory).resolve()
    source = trajectory
    if preprocessing_signature(metadata) is not None:
        reference = metadata["image_preprocessing"].get("raw_source", {})
        if not isinstance(reference, dict):
            raise ValueError("Preprocessed trajectory has no raw source provenance")
        relative, digest = reference.get("relative_path"), reference.get("sha256")
        if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
                or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError("Preprocessed trajectory has malformed raw source provenance")
        source = (trajectory.parent / relative).resolve()
        if source == trajectory or source.suffix != ".hdf5":
            raise ValueError("Preprocessed trajectory must reference a distinct raw .hdf5")
        if library is not None and not source.is_relative_to(Path(library).resolve()):
            raise ValueError("Raw source of preprocessed trajectory is outside the data library")
        if _sha256_file(source) != digest:
            raise ValueError(f"Raw source SHA-256 mismatch: {source}")
        raw = read_metadata(source)
        if preprocessing_signature(raw) is not None:
            raise ValueError("Preprocessed source must refer directly to an unprocessed raw trajectory")
        for key in ("origin_id", "task", "goal", "split", "object_id", "asset_sha256", "simulator",
                    "control_period", "camera_roles", "action_spec", "phase_schema", "physical_record"):
            if raw[key] != metadata[key]:
                raise ValueError(f"Raw source provenance disagrees on {key}")
    name = metadata["physical_record"]
    record = (source.parent / name).resolve()
    if record.parent != source.parent or record.suffix != ".npz":
        raise ValueError(f"{trajectory}: invalid physical record path")
    return record


def validate_splits(paths: list[Path]) -> None:
    """Reject origin or asset reuse across splits for the held-out-asset protocol."""
    origins: dict[str, str] = {}
    assets: dict[str, str] = {}
    for path in paths:
        meta = read_metadata(path)
        for value, seen, label in (
            (meta["origin_id"], origins, "origin"),
            (meta["asset_sha256"], assets, "asset"),
        ):
            if value in seen and seen[value] != meta["split"]:
                raise ValueError(f"{label} {value} occurs in both {seen[value]} and {meta['split']}")
            seen[value] = meta["split"]


def write_pair(path: Path, *, reference: Path, query: Path) -> Path:
    """Create a two-trajectory SynthICL task from independent, compatible recordings."""
    path, reference, query = Path(path), Path(reference).resolve(), Path(query).resolve()
    if path.suffix != ".h5":
        raise ValueError("SynthICL pair files must end in .h5")
    ref_meta, query_meta = read_metadata(reference), read_metadata(query)
    if reference == query or ref_meta["origin_id"] == query_meta["origin_id"]:
        raise ValueError("Reference and query must come from independent original episodes")
    for key in ("task", "goal", "split", "action_spec", "control_period", "camera_roles", "phase_schema"):
        if ref_meta[key] != query_meta[key]:
            raise ValueError(f"Reference and query disagree on {key}")
    if preprocessing_signature(ref_meta) != preprocessing_signature(query_meta):
        raise ValueError("Reference and query disagree on image preprocessing")
    temporary = _temporary_file(path)
    try:
        with h5py.File(temporary, "w") as h5:
            h5.attrs["format"] = "roboverse.icl.synthicl_pair.v1"
            for index, source in enumerate((reference, query)):
                h5[f"trajectory_{index}"] = h5py.ExternalLink(
                    os.path.relpath(source, path.parent.resolve()), "/trajectory"
                )
        temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
