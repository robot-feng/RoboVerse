"""Evaluate one frozen SynthICL checkpoint in a physical RoboVerse episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from roboverse_learn.il.icl.data import (
    IMAGE_KEYS,
    image_preprocessing_plan,
    preprocess_rgb_images,
    preprocessing_signature,
    read_metadata,
    resolve_physical_record,
    validate_splits,
)
from roboverse_pack.tasks.icl.control import FrankaCartesianController
from roboverse_pack.tasks.icl.push_to_region import PickPlaceRegionTask, PushToRegionTask, build_scenario
from tools.icl.collect import CAMERAS
from tools.icl.train import validate_training_data


def image_tensor(value, size, device):
    """Match the original loader's uint8-to-RGB resize operation."""
    tensor = torch.as_tensor(np.asarray(value), device=device).permute(0, 3, 1, 2).float()
    if tensor.max() > 1.5:
        tensor = tensor / 255.0
    if tuple(tensor.shape[-2:]) != tuple(size):
        tensor = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False, antialias=True)
    return tensor.permute(0, 2, 3, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthicl-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dino-repo", type=Path, required=True)
    parser.add_argument("--dino-weights", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--sim", choices=("mujoco", "genesis", "newton"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--position-jitter", type=float, default=0.003)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seed < 0 or not 0 <= args.position_jitter <= 0.03:
        raise ValueError("Seed must be nonnegative and position jitter must be between 0 and 0.03 meters")
    if args.output.exists():
        raise FileExistsError(args.output)
    reference = args.reference.resolve()
    if not reference.is_relative_to(args.library.resolve()):
        raise ValueError("Reference must be inside the checked episode library")
    validate_splits(list(args.library.rglob("*.hdf5")))
    meta = read_metadata(reference)
    resolve_physical_record(reference, meta, library=args.library)
    if meta["split"] != "test" or "seed" not in meta or meta["seed"] == args.seed:
        raise ValueError("Use an independent held-out test demonstration with a different seed")
    config = json.loads((args.checkpoint.parent / "config.json").read_text())
    stats = json.loads((args.checkpoint.parent / "norm_stats.json").read_text())
    validate_training_data(Path(config["dataset"]["hdf5_path"]), args.library)
    if config["shared"]["obs_keys"] or config["shared"]["prompt_length"] != 6:
        raise ValueError("Expected the six-waypoint RGB-only SynthICL configuration")
    sys.path.insert(0, str(args.synthicl_dir.resolve()))
    from icil.config.configuration_act import ACTConfig
    from icil.config.policy_config import PolicyConfig
    from icil.config.shared_config import SharedConfig
    from icil.policy.fm_flowact_icil import FmFlowActICIL

    config["model"]["dino_repo"] = str(args.dino_repo.resolve())
    config["model"]["dino_weights"] = str(args.dino_weights.resolve())
    config["shared"]["device"] = args.device
    model = FmFlowActICIL(PolicyConfig(**config["model"]), SharedConfig(**config["shared"]), ACTConfig(**config["act"]))
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=True))
    model.to(args.device).eval()
    size = config["shared"]["image_size"]
    image_policy = preprocessing_signature(meta)
    if image_policy is not None and tuple(size) != image_policy[1]:
        raise ValueError("Reference preprocessing size differs from the checkpoint's image size")
    pair_path = next(iter(sorted(Path(config["dataset"]["hdf5_path"]).rglob("*.h5"))))
    with h5py.File(pair_path, "r") as pair:
        link = pair.get("trajectory_0", getlink=True)
        training_meta = read_metadata((pair_path.parent / link.filename).resolve())
    if preprocessing_signature(training_meta) != image_policy:
        raise ValueError("Reference and training data use different image preprocessing policies")
    with h5py.File(reference, "r") as h5:
        group = h5["trajectory"]
        indices = group["waypoints_idx"][:]
        if len(indices) != 6:
            raise ValueError("Reference must contain exactly six waypoints")
        prompt = {key: image_tensor(group[key][indices], size, args.device)[None] for key in IMAGE_KEYS}
    rng = np.random.default_rng(args.seed)
    xy = (
        0.5 + float(rng.uniform(-args.position_jitter, args.position_jitter)),
        float(rng.uniform(-args.position_jitter, args.position_jitter)),
    )
    scenario = build_scenario(simulator=args.sim, object_id=meta["object_id"], object_xy=xy)
    info = json.loads((Path(scenario.objects[0].urdf_path).parent / "provenance.json").read_text())
    fingerprint = hashlib.sha256(
        (info["source_files_sha256"]["textured.obj"] + info["source_files_sha256"]["collision.ply"]).encode()
    ).hexdigest()
    if fingerprint != meta["asset_sha256"]:
        raise ValueError("The query asset differs from the held-out demonstration asset")
    task_classes = {"icl.push_to_region": PushToRegionTask, "icl.pick_place_region": PickPlaceRegionTask}
    if meta["task"] not in task_classes:
        raise ValueError(f"Unsupported reference task: {meta['task']}")
    env = task_classes[meta["task"]](scenario, goal=meta["goal"])
    low = np.asarray(stats["action_min"], dtype=float)
    span = np.asarray(stats["action_max"], dtype=float) - low + 1e-6
    steps, clips, failure, initial, initial_joints, final = 0, 0, "", None, None, None
    try:
        obs, _ = env.reset()
        initial = obs.objects["object"].root_state[0].detach().cpu().tolist()
        initial_joints = obs.robots["franka"].joint_pos[0].detach().cpu().tolist()
        controller = FrankaCartesianController(scenario.robots[0], env.handler.get_joint_names("franka", sort=True))
        torch.manual_seed(args.seed)
        for _ in range(env.max_episode_steps):
            data = {f"prompt_{key}": prompt[key] for key in IMAGE_KEYS}
            data["prompt_action_seq"] = torch.zeros((1, 6, 7), device=args.device)
            data["action"] = torch.zeros((1, 1, 7), device=args.device)
            for key, camera in CAMERAS.items():
                frame = obs.cameras[camera].rgb[0].detach().cpu().numpy()[None]
                if image_policy is not None:
                    plan = image_preprocessing_plan(tuple(frame.shape[1:3]), tuple(size), mode=image_policy[0])
                    frame = preprocess_rgb_images(frame, plan)
                data[key] = image_tensor(frame, size, args.device)[None, None]
            with torch.inference_mode():
                prediction = model.inference(data)[0, 0].detach().cpu().numpy()
            action = prediction * span + low
            if not np.isfinite(action).all():
                raise ValueError("Policy produced a non-finite action")
            before = action.copy()
            for part, limit in ((slice(0, 3), 0.0209), (slice(3, 6), 0.209)):
                norm = np.linalg.norm(action[part])
                if norm > limit:
                    action[part] *= limit / norm
            action[6] = np.clip(action[6], 0, 1)
            clips += int(not np.allclose(before, action))
            q = obs.robots["franka"].joint_pos[0].detach().cpu().numpy()
            joints = controller.joint_targets(q, action)
            obs, _, _, timeout, _ = env.step([{"franka": {"dof_pos_target": joints}}])
            steps += 1
            if bool(env._terminated(obs)[0]) or bool(timeout[0]):
                break
        success = bool(env._terminated(obs)[0])
        final = obs.objects["object"].root_state[0].detach().cpu().tolist()
        if not success:
            failure = "Episode ended without satisfying the success checker"
    except Exception as exc:
        success = False
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()
    result = {
        "simulator": args.sim,
        "task": meta["task"],
        "goal": meta["goal"],
        "object_id": meta["object_id"],
        "asset_sha256": fingerprint,
        "query_seed": args.seed,
        "position_jitter": args.position_jitter,
        "reference_origin": meta["origin_id"],
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "initial_object_state": initial,
        "initial_robot_joints": initial_joints,
        "final_object_state": final,
        "success": success,
        "steps": steps,
        "clipped_actions": clips,
        "failure": failure,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
