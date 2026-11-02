"""Validate RoboVerse demonstrations and run the original SynthICL trainer."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import h5py

from metasim.utils.trajectory import read_header
from roboverse_learn.il.icl.data import (
    MASK_KEYS,
    preprocessing_signature,
    read_metadata,
    resolve_physical_record,
    validate_splits,
)


def validate_training_data(dataset: Path, library: Path, *, task_length: int | None = None) -> int:
    """Reject invalid provenance and reused origins before SynthICL's random split."""
    dataset, library = dataset.resolve(), library.resolve()
    raw_files = sorted(library.rglob("*.hdf5"))
    pairs = sorted(dataset.rglob("*.h5"))
    if not raw_files or len(pairs) < 2:
        raise ValueError("At least two independent pair files and a raw trajectory library are required")
    validate_splits(raw_files)
    used_origins: set[str] = set()
    image_policy_set = False
    image_policy = None
    for pair in pairs:
        sources = []
        with h5py.File(pair, "r") as h5:
            if set(h5.keys()) != {"trajectory_0", "trajectory_1"}:
                raise ValueError(f"Invalid paired trajectories: {pair}")
            for name in ("trajectory_0", "trajectory_1"):
                link = h5.get(name, getlink=True)
                if not isinstance(link, h5py.ExternalLink) or link.path != "/trajectory":
                    raise ValueError(f"{pair}: {name} must link to a raw trajectory")
                source = (pair.parent / link.filename).resolve()
                if not source.is_relative_to(library) or source.suffix != ".hdf5":
                    raise ValueError(f"{pair}: source must be inside the raw library")
                meta = read_metadata(source)
                if meta["split"] != "train":
                    raise ValueError(f"{source}: only training demonstrations may be used")
                if meta["origin_id"] in used_origins:
                    raise ValueError(f"Origin reused across pair files: {meta['origin_id']}")
                used_origins.add(meta["origin_id"])
                record = resolve_physical_record(source, meta, library=library)
                header = read_header(record)
                info, provenance = header["info"], header["provenance"]
                if (
                    info.get("success") is not True
                    or info.get("task") != meta["task"]
                    or info.get("goal") != meta["goal"]
                    or provenance.get("simulator") != meta["simulator"]
                    or abs(provenance["env_step_s"] - meta["control_period"]) > 1e-9
                ):
                    raise ValueError(f"{source}: physical record disagrees with demonstration metadata")
                if header["num_steps"] != h5[name]["actions"].shape[0] - 1:
                    raise ValueError(f"{source}: physical record and action count differ")
                if h5[name]["waypoints_idx"].shape != (6,):
                    raise ValueError(f"{source}: SynthICL requires six demonstration waypoints")
                signature = preprocessing_signature(meta)
                for key in ("images_front", "images_wrist", "images_side"):
                    image = h5[name][key]
                    if (image.ndim != 4 or image.dtype.name != "uint8" or image.shape[0] != h5[name]["actions"].shape[0]
                            or image.shape[-1] != 3 or min(image.shape[1:3]) < 1):
                        raise ValueError(f"{source}: invalid RGB frames for {key}")
                    if signature is not None and tuple(image.shape[1:3]) != signature[1]:
                        raise ValueError(f"{source}: {key} shape disagrees with image preprocessing metadata")
                    mask_key = MASK_KEYS[key]
                    if signature is not None or mask_key in h5[name]:
                        if mask_key not in h5[name]:
                            raise ValueError(f"{source}: missing valid mask for {key}")
                        mask = h5[name][mask_key]
                        if mask.dtype.name != "bool" or mask.shape != image.shape[1:3]:
                            raise ValueError(f"{source}: invalid valid mask for {key}")
                if task_length is not None and h5[name]["actions"].shape[0] > task_length:
                    raise ValueError(f"{source}: increase --task-length to avoid truncating the trajectory")
                sources.append(meta)
        for key in ("task", "goal", "split", "action_spec", "control_period", "camera_roles", "phase_schema"):
            if sources[0][key] != sources[1][key]:
                raise ValueError(f"{pair}: trajectories disagree on {key}")
        if preprocessing_signature(sources[0]) != preprocessing_signature(sources[1]):
            raise ValueError(f"{pair}: trajectories disagree on image preprocessing")
        signature = preprocessing_signature(sources[0])
        if not image_policy_set:
            image_policy, image_policy_set = signature, True
        elif signature != image_policy:
            raise ValueError("Training pairs use different image preprocessing policies")
        if signature is not None and signature[1] != (256, 256):
            raise ValueError("The current SynthICL training configuration requires preprocessed images at 256x256")
    return len(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthicl-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--dino-repo", type=Path, required=True)
    parser.add_argument("--dino-weights", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/icl"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--task-length", type=int, default=350)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not (args.synthicl_dir / "scripts/train.py").is_file():
        parser.error("--synthicl-dir must contain scripts/train.py")
    if not args.dino_repo.is_dir() or not args.dino_weights.is_file():
        parser.error("Provide the DINOv3 repository and the official ViT-S/16+ pretrained weights")
    if min(args.epochs, args.batch_size, args.samples_per_task, args.task_length) < 1 or args.num_workers < 0:
        parser.error("Invalid training sizes")
    if not 0 < args.train_split < 1:
        parser.error("--train-split must be between zero and one")
    count = validate_training_data(args.dataset_dir, args.library, task_length=args.task_length)
    if int(count * args.train_split) in (0, count):
        parser.error("The pair split would leave training or validation empty")
    sys.path.insert(0, str(args.synthicl_dir.resolve()))
    spec = importlib.util.spec_from_file_location("synthicl_original_train", args.synthicl_dir / "scripts/train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_configs = module.make_configs

    def make_configs(train_args):
        configs = original_configs(train_args)
        configs[2].task_length = args.task_length
        configs[2].n_samples_per_task = args.samples_per_task
        return configs

    module.make_configs = make_configs
    args.pretrained_path = None
    args.eval_frequency = 1000
    args.save_frequency = 5000
    args.disable_wandb = True
    args.wandb_name = None
    args.wandb_project = "synthicl"
    module.train(args)


if __name__ == "__main__":
    main()
