# Copyright (c) 2023-2026 AgiBot Inc.
# SPDX-License-Identifier: MPL-2.0
#
# Adapted from Genie Sim benchmark (https://github.com/AgibotTech/genie_sim).
# Changes: import task data and statically extract the source reset configuration.
# Full license: roboverse_pack/tasks/icl/LICENSE.geniesim-benchmark
"""Import Genie Sim benchmark configurations and checker graphs as RoboVerse data."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

_INITIAL_POSE_SOURCE_PATHS = (
    "benchmark/config/robot_init_states.py",
    "utils/name_utils.py",
    "utils/infer_pre_process.py",
    "benchmark/envs/dummy_env.py",
    "benchmark/envs/vec_env.py",
)
_RESET_SEMANTICS_SHA256 = {
    "utils/name_utils.py": "e50f5462052a18f46602c4324f33cb5ebf8e9bb5c9dfbca8ef7fb5628f8d31e6",
    "utils/infer_pre_process.py": "e357cb62e750bc35fd633d1b79518bb6c4f3134b21573ab7abc4330224b6e3ff",
    "benchmark/envs/dummy_env.py": "13d9f201ef979eefb20b9a6694884133f601fedd3635d6eb5623f4759523d4c9",
    "benchmark/envs/vec_env.py": "df1a336e480e95fce7ff65267288bf49f38dde862165abf7620a0efb1599958b",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _revision(source_root: Path) -> str:
    """Return a verified clean Genie Sim checkout revision, or ``unknown``."""
    try:
        source_root = Path(source_root).resolve()

        def git(*args: str) -> str:
            return subprocess.run(
                ["git", "-C", str(source_root), *args], check=True, capture_output=True, text=True, encoding="utf-8"
            ).stdout.strip()

        checkout = Path(git("rev-parse", "--show-toplevel")).resolve()
        expected_root = checkout / "source" / "geniesim_benchmark" / "src" / "geniesim_benchmark"
        if expected_root.resolve() != source_root:
            return "unknown"
        origin = git("remote", "get-url", "origin").lower().rstrip("/").removesuffix(".git")
        origin = origin.replace("git@github.com:", "https://github.com/")
        if origin != "https://github.com/agibottech/genie_sim":
            return "unknown"
        if git("status", "--porcelain", "--untracked-files=all", "--", "."):
            return "unknown"
        revision = git("rev-parse", "HEAD")
        return revision if re.fullmatch(r"[0-9a-f]{40}", revision) else "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def source_file_hashes(pack: dict) -> dict[str, str]:
    """Collect the task files whose hashes identify an imported source snapshot."""
    files: dict[str, str] = {}

    def add(path: str, digest: str) -> None:
        if not isinstance(path, str) or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid benchmark source path or SHA-256")
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe benchmark source path: {path}")
        if path in files and files[path] != digest:
            raise ValueError(f"Conflicting benchmark source hashes: {path}")
        files[path] = digest

    for record in pack.get("runs", []):
        add(record["source_file"], record["source_sha256"])
    for category in ("evaluation_configs", "robot_configs"):
        for record in pack.get(category, {}).values():
            add(record["source_file"], record["source_sha256"])
    for instance in pack.get("instances", {}).values():
        for record in instance["source_files"].values():
            add(record["path"], record["sha256"])
    for record in pack.get("initial_pose_source", {}).get("source_files", {}).values():
        add(record["path"], record["sha256"])
    if not files:
        raise ValueError("Imported benchmark has no source file hashes")
    return files


def source_content_sha256(pack: dict) -> str:
    """Identify the imported source files independently of Git checkout metadata."""
    return _content_sha256(source_file_hashes(pack))


def imported_payload_sha256(pack: dict) -> str:
    """Hash derived benchmark data separately from mutable provenance metadata."""
    payload = {key: value for key, value in pack.items() if key != "source"}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _content_sha256(files: dict[str, str]) -> str:
    canonical = json.dumps(sorted(files.items()), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _static_assignments(path: Path):
    """Resolve simple source literals and named list concatenations without executing Python."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assignments = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    if target.id in assignments:
                        raise ValueError(f"Duplicate source assignment {target.id!r} in {path}")
                    assignments[target.id] = statement.value

    def resolve(node, active=frozenset()):
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool, type(None))):
            if isinstance(node.value, float) and not math.isfinite(node.value):
                raise ValueError(f"Non-finite source literal in {path}")
            return node.value
        if isinstance(node, (ast.List, ast.Tuple)):
            return [resolve(value, active) for value in node.elts]
        if isinstance(node, ast.Dict):
            result = {}
            for key, value in zip(node.keys, node.values, strict=True):
                resolved_key = resolve(key, active)
                if not isinstance(resolved_key, str) or resolved_key in result:
                    raise ValueError(f"Invalid or duplicate source dictionary key in {path}")
                result[resolved_key] = resolve(value, active)
            return result
        if isinstance(node, ast.Name):
            if node.id in active or node.id not in assignments:
                raise ValueError(f"Unknown or recursive source constant {node.id!r} in {path}")
            return resolve(assignments[node.id], active | {node.id})
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = resolve(node.operand, active)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Invalid source numeric literal in {path}")
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = resolve(node.left, active), resolve(node.right, active)
            if isinstance(left, list) and isinstance(right, list):
                return left + right
        raise ValueError(f"Unsupported source expression {type(node).__name__} in {path}")

    return assignments, resolve


def _source_initial_pose_snapshot(source_root: Path) -> dict:
    """Extract named G1/G2 reset poses using the upstream task and joint tables."""
    source_root = Path(source_root).resolve()
    files = {}
    for relative in _INITIAL_POSE_SOURCE_PATHS:
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing Genie Sim reset source: {path}")
        files[relative] = {"path": relative, "sha256": _sha256(path)}
    for relative, expected in _RESET_SEMANTICS_SHA256.items():
        if files[relative]["sha256"] != expected:
            raise ValueError(f"Unsupported Genie Sim reset semantics in {relative}; review the source before importing")

    task_nodes, task_resolve = _static_assignments(source_root / _INITIAL_POSE_SOURCE_PATHS[0])
    task_info = task_resolve(task_nodes["TASK_INFO_DICT"])
    if not isinstance(task_info, dict) or not all(isinstance(item, dict) for item in task_info.values()):
        raise ValueError("Genie Sim TASK_INFO_DICT must map task names to robot configurations")
    robot_nodes, robot_resolve = _static_assignments(source_root / _INITIAL_POSE_SOURCE_PATHS[1])
    configs_node = robot_nodes.get("ROBOT_CONFIGS")
    if not isinstance(configs_node, ast.Dict):
        raise ValueError("Genie Sim ROBOT_CONFIGS is not a literal dictionary")
    wanted_fields = {
        "arm_joints", "head_joints", "waist_joints", "gripper_joints", "init_gripper_open",
        "gripper_offset", "gripper_scale", "limit_val",
    }
    robot_configs = {}
    for key, value in zip(configs_node.keys, configs_node.values, strict=True):
        robot_name = robot_resolve(key)
        if robot_name not in {"G1_omnipicker", "G2_omnipicker"}:
            continue
        if not isinstance(value, ast.Dict):
            raise ValueError(f"Invalid source robot config: {robot_name}")
        config = {}
        for field, field_value in zip(value.keys, value.values, strict=True):
            field_name = robot_resolve(field)
            if field_name in wanted_fields:
                config[field_name] = robot_resolve(field_value)
        required = {"arm_joints", "head_joints", "waist_joints", "gripper_joints", "init_gripper_open"}
        if not required <= config.keys():
            raise ValueError(f"Incomplete source robot reset config: {robot_name}")
        robot_configs[robot_name] = config
    if set(robot_configs) != {"G1_omnipicker", "G2_omnipicker"}:
        raise ValueError("Source robot config must include G1 and G2 Omnipicker")

    joint_positions = {}
    for task_name, variants in task_info.items():
        for robot_name, config in robot_configs.items():
            if robot_name not in variants:
                continue
            state = variants[robot_name]
            body = state["body_state"]
            head = body[:2] if robot_name.startswith("G1_") else state["head_state"]
            waist = body[2:4] if robot_name.startswith("G1_") else body
            groups = (
                ("arm_joints", state["init_arm"]),
                ("head_joints", head),
                ("waist_joints", waist),
                ("gripper_joints", config["init_gripper_open"]),
            )
            named = {}
            for names_key, values in groups:
                names = config[names_key]
                if not isinstance(names, list) or not isinstance(values, list) or len(names) != len(values):
                    raise ValueError(f"Reset joint count mismatch for {task_name}/{robot_name}/{names_key}")
                for name, value in zip(names, values, strict=True):
                    if (
                        not isinstance(name, str) or name in named or isinstance(value, bool)
                        or not isinstance(value, (int, float)) or not math.isfinite(value)
                    ):
                        raise ValueError(f"Invalid reset joint value for {task_name}/{robot_name}: {name!r}")
                    named[name] = float(value)
            joint_positions.setdefault(task_name, {})[robot_name] = named
    return {
        "format": "roboverse.icl.geniesim_initial_poses.v1",
        "source_files": files,
        "task_info": task_info,
        "robot_configs": robot_configs,
        "joint_positions": joint_positions,
        "gripper_reset_source": "ROBOT_CONFIGS.init_gripper_open",
    }


def source_tree_content_sha256(source_root: Path) -> str:
    """Hash the same source files as an import without parsing USD scene payloads."""
    source_root = Path(source_root).resolve()
    configs = sorted((source_root / "config").glob("g[12]op_*.yaml"))
    if not configs:
        raise FileNotFoundError(f"No Genie Sim run configurations under {source_root}")
    paths = set(configs)
    paths.update(source_root / name for name in _INITIAL_POSE_SOURCE_PATHS)
    for config_path in configs:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        benchmark = config.get("benchmark", {}) if isinstance(config, dict) else {}
        family, subtask = benchmark.get("task_name"), benchmark.get("sub_task_name")
        if not all(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in (family, subtask)):
            raise ValueError(f"Missing scene family or subtask in {config_path}")
        eval_path = source_root / "benchmark" / "config" / "eval_tasks" / f"{family}.json"
        paths.add(eval_path)
        evaluation = _load_json(eval_path)
        robot_name = evaluation.get("robot", {}).get("robot_cfg")
        if robot_name:
            if not isinstance(robot_name, str) or Path(robot_name).name != robot_name:
                raise ValueError(f"Unsafe Genie Sim robot config reference in {eval_path}: {robot_name!r}")
            paths.add(source_root / "app" / "robot_cfg" / robot_name)
        instance_root = source_root / "benchmark" / "config" / "llm_task" / subtask
        if not instance_root.is_dir():
            raise FileNotFoundError(f"Missing Genie Sim task instances: {instance_root}")
        instances = [path for path in instance_root.iterdir() if path.is_dir()]
        if not instances:
            raise ValueError(f"No scene instances found for {subtask}")
        for instance in instances:
            paths.update(instance / name for name in ("scene_info.json", "problems.json", "instructions.json", "scene.usda"))
    files = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing Genie Sim source file: {path}")
        files[path.relative_to(source_root).as_posix()] = _sha256(path)
    return _content_sha256(files)


def _resolve_source_revision(source_root: Path, override: str | None = None) -> str:
    """Use an audited explicit upstream pin or the source checkout's own revision."""
    if override is None:
        return _revision(source_root)
    if not re.fullmatch(r"[0-9a-f]{40}", override):
        raise ValueError("source_revision must be a full lowercase 40-character Git commit SHA")
    return override


def correct_source_provenance(
    *,
    source_root: Path,
    output: Path,
    source_revision: str | None = None,
) -> dict:
    """Correct an existing import only after checking each original source file."""
    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    path = output / "benchmark.json"
    pack = json.loads(path.read_text(encoding="utf-8"))
    if pack.get("format") != "roboverse.icl.geniesim_benchmark_specs.v1":
        raise ValueError(f"Unexpected imported benchmark format: {path}")
    files = source_file_hashes(pack)
    for relative, expected in files.items():
        candidate = (source_root / relative).resolve()
        if not candidate.is_relative_to(source_root) or not candidate.is_file() or _sha256(candidate) != expected:
            raise ValueError(f"Benchmark source file checksum mismatch: {relative}")
    fingerprint = source_content_sha256(pack)
    if fingerprint != source_tree_content_sha256(source_root):
        raise ValueError("Imported benchmark source-file set does not match the source tree")
    fresh = import_specs(source_root)
    if {key: value for key, value in pack.items() if key != "source"} != {
        key: value for key, value in fresh.items() if key != "source"
    }:
        raise ValueError("Imported benchmark payload does not match the verified source files")
    source = pack["source"]
    if source.get("repository") != "https://github.com/AgibotTech/genie_sim":
        raise ValueError("Imported benchmark declares an unexpected source repository")
    old_revision = source.get("revision", "unknown")
    new_revision = _resolve_source_revision(source_root, source_revision)
    if old_revision != new_revision:
        source["revision_correction"] = {
            "previous_revision": old_revision,
            "reason": (
                "Explicit upstream revision was verified with the GenieSim inventory audit."
                if source_revision is not None
                else "Legacy Git lookup could resolve an unrelated parent workspace."
            ),
        }
    source["revision"] = new_revision
    source["content_sha256"] = fingerprint
    source["imported_payload_sha256"] = imported_payload_sha256(pack)
    encoded = json.dumps(pack, ensure_ascii=False, indent=2) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=".benchmark-", suffix=".json", dir=output)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return pack


def _load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _collect_assets(scene_info: dict) -> list[str]:
    if not scene_info:
        return []
    layout = scene_info.get("layout")
    if not isinstance(layout, dict):
        raise ValueError("Genie Sim scene_info.json has no layout object")
    return sorted({str(item["usd"]) for item in layout.values() if isinstance(item, dict) and item.get("usd")})


def _usd_override_value(value):
    """Convert authored USD scalar, vector, matrix or quaternion values into JSON."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Scene override contains a non-finite scalar")
        return value
    if hasattr(value, "GetReal") and hasattr(value, "GetImaginary"):
        return [_usd_override_value(float(value.GetReal())), *[_usd_override_value(float(x)) for x in value.GetImaginary()]]
    if hasattr(value, "path") and isinstance(value.path, str):
        return value.path
    try:
        return [_usd_override_value(item) for item in value]
    except TypeError as exc:
        raise ValueError(f"Unsupported scene override value {type(value).__name__}") from exc


def _scene_authored_overrides(payload_prim) -> dict:
    """Capture scene-layer physics and child-frame opinions while payloads stay unloaded."""
    root_path = payload_prim.GetPath()
    stack = [payload_prim]
    attributes, relationships = [], []
    while stack:
        prim = stack.pop()
        stack.extend(reversed(prim.GetAllChildren()))
        relative_path = str(prim.GetPath().MakeRelativePath(root_path))
        for attr in prim.GetAttributes():
            name = attr.GetName()
            relevant = name.startswith(("physics:", "physx")) or (
                prim != payload_prim and (name.startswith("xformOp:") or name == "xformOpOrder")
            )
            if not relevant or not attr.HasAuthoredValueOpinion():
                continue
            times = attr.GetTimeSamples()
            attributes.append({
                "prim": relative_path,
                "attribute": name,
                "type": str(attr.GetTypeName()),
                "value": _usd_override_value(attr.Get()),
                "time_samples": [{"time": time, "value": _usd_override_value(attr.Get(time))} for time in times],
            })
        for rel in prim.GetRelationships():
            if rel.GetName().startswith("material:binding:physics") and rel.HasAuthoredTargets():
                relationships.append({
                    "prim": relative_path,
                    "relationship": rel.GetName(),
                    "targets": [str(target) for target in rel.GetTargets()],
                })
    return {"format": "roboverse.icl.geniesim_scene_authored_overrides.v1",
            "attributes": attributes, "relationships": relationships}


def _scene_snapshot_from_usd(scene_path: Path) -> dict | None:
    """Capture authored payloads, visibility and transforms without loading asset payloads."""
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom
    except ImportError as exc:
        raise RuntimeError("USD bindings (pxr) are required to import Genie Sim source scenes") from exc

    stage = Usd.Stage.Open(str(scene_path), load=Usd.Stage.LoadNone)
    if stage is None:
        raise ValueError(f"Could not parse Genie Sim scene USD: {scene_path}")
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not math.isfinite(meters_per_unit) or meters_per_unit <= 0:
        raise ValueError(f"Invalid metersPerUnit in Genie Sim scene: {scene_path}")

    if not stage.GetPrimAtPath("/World/Objects"):
        return None

    objects = {}
    for prim in stage.TraverseAll():
        if not prim.HasPayload() or not str(prim.GetPath()).startswith("/World/Objects/"):
            continue
        payloads = prim.GetMetadata("payload")
        if not isinstance(payloads, Sdf.PayloadListOp):
            continue
        references = payloads.GetAppliedItems()
        if len(references) != 1:
            raise ValueError(f"Expected one effective payload in {scene_path} at {prim.GetPath()}")
        asset_path = references[0].assetPath.replace("\\", "/")
        asset_id = Path(asset_path).parent.name
        if not asset_id or not all(character.isalnum() or character in "_-" for character in asset_id):
            raise ValueError(f"Unsupported payload asset path in {scene_path}: {asset_path!r}")
        if not asset_path.startswith("/geniesim_assets/objects/") or Path(asset_path).name not in {
            "Aligned.usda", "Aligned.usd"
        }:
            raise ValueError(f"Unsupported Genie Sim object payload in {scene_path}: {asset_path!r}")
        if prim.GetName() in objects:
            raise ValueError(f"Duplicate payload instance name in {scene_path}: {prim.GetName()!r}")

        # XformCache returns identity for unloaded payload prims in some USD
        # versions. Compose authored local matrices directly, child to parent.
        ancestors = []
        current = prim
        while current and current.IsValid():
            ancestors.append(current)
            current = current.GetParent()
        world = Gf.Matrix4d(1.0)
        for ancestor in reversed(ancestors):
            # Some Genie Sim payload prims author xformOp attributes without
            # the Xform schema type, so test the API result rather than bool().
            local = UsdGeom.Xformable(ancestor).GetLocalTransformation()
            if isinstance(local, tuple):
                local = local[0]
            world = local * world
        transform = Gf.Transform(world)
        translation = transform.GetTranslation()
        quaternion = transform.GetRotation().GetQuat()
        scale = transform.GetScale()
        xyz = [float(value) * meters_per_unit for value in translation]
        xyzw = [float(value) for value in quaternion.GetImaginary()] + [float(quaternion.GetReal())]
        scale = [float(value) for value in scale]
        if not all(math.isfinite(value) for value in (*xyz, *xyzw, *scale)) or any(value <= 0 for value in scale):
            raise ValueError(f"Non-finite transform on Genie Sim payload: {prim.GetPath()}")
        norm = math.sqrt(sum(value * value for value in xyzw))
        if norm < 1e-8:
            raise ValueError(f"Invalid orientation on Genie Sim payload: {prim.GetPath()}")
        objects[prim.GetName()] = {
            "asset_id": asset_id,
            "asset_path": asset_path,
            "prim_path": str(prim.GetPath()),
            "active": bool(prim.IsActive()),
            "xyz": xyz,
            "xyzw": [value / norm for value in xyzw],
            "scale": scale,
            "scene_authored_overrides": _scene_authored_overrides(prim),
        }
    return {"format": "roboverse.icl.geniesim_source_scene.v1", "sha256": _sha256(scene_path), "objects": objects}


def _scene_layout_from_usd(scene_path: Path) -> dict:
    """Recover active rigid-object poses from payload references in a source scene USDA."""
    snapshot = _scene_snapshot_from_usd(scene_path)
    if snapshot is None:
        return {}
    return {
        name: {
            "usd": item["asset_id"],
            "xyz": item["xyz"],
            "xyzw": item["xyzw"],
            "scale": item["scale"],
        }
        for name, item in snapshot["objects"].items()
        if item["active"]
    }


def _scene_reconciliation(scene_info: dict, snapshot: dict) -> dict:
    """Record differences between source metadata and the authored scene payloads."""
    metadata = scene_info.get("layout", {})
    if not isinstance(metadata, dict):
        raise ValueError("Genie Sim scene_info.json has no layout object")
    objects = snapshot["objects"]
    active_names = {name for name, item in objects.items() if item["active"]}
    return {
        "metadata_only": sorted(set(metadata) - set(objects)),
        "source_only": sorted(active_names - set(metadata)),
        "inactive": sorted(name for name, item in objects.items() if not item["active"]),
        "asset_id_mismatches": {
            name: {"metadata": entry["usd"], "source_payload": objects[name]["asset_id"]}
            for name, entry in sorted(metadata.items())
            if name in active_names and isinstance(entry, dict) and entry.get("usd")
            and entry["usd"] != objects[name]["asset_id"]
        },
    }


def import_specs(source_root: Path, *, source_revision: str | None = None) -> dict:
    """Preserve every run, scene instance, instruction and source checker graph."""
    source_root = source_root.resolve()
    resolved_source_revision = _resolve_source_revision(source_root, source_revision)
    config_root = source_root / "config"
    benchmark_root = source_root / "benchmark" / "config"
    task_root = benchmark_root / "llm_task"
    eval_root = benchmark_root / "eval_tasks"
    configs = sorted(config_root.glob("g[12]op_*.yaml"))
    if not configs or not task_root.is_dir() or not eval_root.is_dir():
        raise FileNotFoundError(
            f"Expected Genie Sim config/, benchmark/config/llm_task and eval_tasks under {source_root}"
        )

    instances: dict[str, dict] = {}
    evaluation_configs: dict[str, dict] = {}
    robot_configs: dict[str, dict] = {}
    semantic_tasks: dict[str, dict] = {}
    runs = []
    empty_scene_info_count = 0
    for config_path in configs:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        benchmark = config.get("benchmark", {}) if isinstance(config, dict) else {}
        robot = benchmark.get("platform")
        scene_family = benchmark.get("task_name")
        subtask = benchmark.get("sub_task_name")
        if not all(
            isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+", value)
            for value in (robot, scene_family, subtask)
        ):
            raise ValueError(f"Missing platform, task_name or sub_task_name in {config_path}")

        eval_path = eval_root / f"{scene_family}.json"
        if not eval_path.is_file():
            raise FileNotFoundError(f"Missing Genie Sim evaluation configuration: {eval_path}")
        eval_config = _load_json(eval_path)
        evaluation_configs[scene_family] = {
            "source_file": eval_path.relative_to(source_root).as_posix(),
            "source_sha256": _sha256(eval_path),
            "config": eval_config,
        }
        robot_config_name = eval_config.get("robot", {}).get("robot_cfg")
        if robot_config_name:
            if not isinstance(robot_config_name, str) or Path(robot_config_name).name != robot_config_name:
                raise ValueError(f"Unsafe Genie Sim robot config reference in {eval_path}: {robot_config_name!r}")
            robot_config_path = source_root / "app" / "robot_cfg" / robot_config_name
            robot_config = _load_json(robot_config_path)
            robot_configs[robot_config_name] = {
                "source_file": robot_config_path.relative_to(source_root).as_posix(),
                "source_sha256": _sha256(robot_config_path),
                "config": robot_config,
            }

        task_instances = task_root / subtask
        if not task_instances.is_dir():
            raise FileNotFoundError(f"Missing Genie Sim task instances: {task_instances}")
        instance_ids = sorted((path.name for path in task_instances.iterdir() if path.is_dir()), key=int)
        if not instance_ids:
            raise ValueError(f"No scene instances found for {subtask}")
        for instance_id in instance_ids:
            key = f"{subtask}/{instance_id}"
            if key in instances:
                continue
            instance_root = task_instances / instance_id
            required = {
                name: instance_root / name
                for name in ("scene_info.json", "problems.json", "instructions.json", "scene.usda")
            }
            missing = [path for path in required.values() if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Incomplete source scene instance {key}: {missing}")
            scene_info = _load_json(required["scene_info.json"])
            scene_info_was_empty = not scene_info
            scene_layout_source = "scene_info.json"
            source_scene = _scene_snapshot_from_usd(required["scene.usda"])
            if scene_info_was_empty:
                empty_scene_info_count += 1
                recovered_layout = {
                    name: {"usd": item["asset_id"], "xyz": item["xyz"], "xyzw": item["xyzw"], "scale": item["scale"]}
                    for name, item in (source_scene or {}).get("objects", {}).items()
                    if item["active"]
                }
                if recovered_layout:
                    scene_info = {"layout": recovered_layout, "layout_source": "scene.usda_payloads"}
                    scene_layout_source = "scene.usda_payloads"
            if source_scene is not None:
                source_scene["reconciliation"] = _scene_reconciliation(scene_info, source_scene)
            problems = _load_json(required["problems.json"])
            instructions = _load_json(required["instructions.json"])
            instances[key] = {
                "task": subtask,
                "instance_id": instance_id,
                "source_files": {
                    name: {
                        "path": path.relative_to(source_root).as_posix(),
                        "sha256": _sha256(path),
                    }
                    for name, path in required.items()
                },
                "scene_id": scene_info.get("scene_id"),
                "seed": scene_info.get("seed"),
                "scene_info": scene_info,
                "scene_info_was_empty": scene_info_was_empty,
                "scene_layout_source": scene_layout_source,
                "source_scene": source_scene,
                "checker_graphs": problems,
                "instructions": instructions,
                "asset_ids": sorted({
                    item["asset_id"] for item in source_scene["objects"].values() if item["active"]
                }) if source_scene is not None else _collect_assets(scene_info),
            }

        semantic_key = f"{robot}/{subtask}"
        task = semantic_tasks.setdefault(
            semantic_key,
            {
                "id": semantic_key,
                "robot": robot,
                "subtask": subtask,
                "scene_families": [],
                "instances": [f"{subtask}/{instance_id}" for instance_id in instance_ids],
                "run_ids": [],
            },
        )
        if scene_family not in task["scene_families"]:
            task["scene_families"].append(scene_family)

        runs.append({
            "id": config_path.stem,
            "source_file": config_path.relative_to(source_root).as_posix(),
            "source_sha256": _sha256(config_path),
            "robot": robot,
            "scene_family": scene_family,
            "subtask": subtask,
            "semantic_task": semantic_key,
            "run_config": config,
            "evaluation_config": scene_family,
            "instances": [f"{subtask}/{instance_id}" for instance_id in instance_ids],
        })
        task["run_ids"].append(config_path.stem)

    initial_pose_source = _source_initial_pose_snapshot(source_root)
    for run in runs:
        evaluation = evaluation_configs[run["evaluation_config"]]["config"]
        source_robot_config = evaluation.get("robot", {}).get("robot_cfg", "")
        if not isinstance(source_robot_config, str):
            raise ValueError(f"Invalid source robot config for {run['id']}: {source_robot_config!r}")
        matches = [name for name in initial_pose_source["robot_configs"] if name in source_robot_config]
        if len(matches) != 1 or matches[0] not in initial_pose_source["joint_positions"].get(run["subtask"], {}):
            raise ValueError(f"Missing unambiguous source reset pose for {run['id']}: {source_robot_config!r}")
        run["initial_pose_robot"] = matches[0]

    pack = {
        "format": "roboverse.icl.geniesim_benchmark_specs.v1",
        "source": {
            "name": "Genie Sim Benchmark",
            "repository": "https://github.com/AgibotTech/genie_sim",
            "revision": resolved_source_revision,
            "license": "MPL-2.0",
            "task_instance_count": len(instances),
            "run_configuration_count": len(runs),
            "semantic_task_count": len(semantic_tasks),
            "scene_assignment_count": sum(len(task["instances"]) for task in semantic_tasks.values()),
            "empty_scene_info_count": empty_scene_info_count,
            "recovered_scene_layout_count": sum(
                item["scene_layout_source"] == "scene.usda_payloads" for item in instances.values()
            ),
            "source_scene_snapshot_count": sum(item["source_scene"] is not None for item in instances.values()),
            "source_scene_scaled_instance_count": sum(
                any(
                    obj["active"] and any(abs(value - 1.0) > 1e-5 for value in obj["scale"])
                    for obj in item["source_scene"]["objects"].values()
                )
                for item in instances.values() if item["source_scene"] is not None
            ),
        },
        "evaluation_configs": evaluation_configs,
        "robot_configs": robot_configs,
        "initial_pose_source": initial_pose_source,
        "semantic_tasks": [semantic_tasks[key] for key in sorted(semantic_tasks)],
        "runs": runs,
        "instances": instances,
        "asset_ids": sorted({asset_id for item in instances.values() for asset_id in item["asset_ids"]}),
    }
    pack["source"]["content_sha256"] = source_content_sha256(pack)
    pack["source"]["imported_payload_sha256"] = imported_payload_sha256(pack)
    return pack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Path to genie_sim-main/source/geniesim_benchmark/src/geniesim_benchmark",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("roboverse_data/icl/geniesim_benchmark"),
        help="New ignored data directory on the RoboVerse drive",
    )
    parser.add_argument(
        "--correct-provenance",
        action="store_true",
        help="Verify source-file hashes and correct the revision in an existing import",
    )
    parser.add_argument(
        "--source-revision",
        help="Explicit upstream commit SHA, first verified with audit_geniesim_inventory.py against the source tree",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if args.correct_provenance:
        pack = correct_source_provenance(
            source_root=args.source_root,
            output=output,
            source_revision=args.source_revision,
        )
        readme_path = output / "README.txt"
        if readme_path.is_file():
            lines = readme_path.read_text(encoding="utf-8").splitlines()
            lines = [line for line in lines if not line.startswith("Source content SHA-256:")]
            lines = [
                f"Source revision: {pack['source']['revision']}" if line.startswith("Source revision:") else line
                for line in lines
            ]
            lines.append(f"Source content SHA-256: {pack['source']['content_sha256']}")
            readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Verified benchmark provenance: {output}")
        return
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing imported benchmark data: {output}")
    pack = import_specs(args.source_root, source_revision=args.source_revision)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="geniesim-specs-", dir=output.parent) as temporary:
        staging = Path(temporary) / "benchmark"
        staging.mkdir()
        path = staging / "benchmark.json"
        path.write_text(json.dumps(pack, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (staging / "README.txt").write_text(
            "Imported Genie Sim task specifications, scene metadata, checker graphs and instructions.\n"
            "No demonstration episodes or videos are included. Source assets are referenced by asset ID.\n"
            f"Source revision: {pack['source']['revision']}\n"
            f"Source content SHA-256: {pack['source']['content_sha256']}\n"
            "Source license: MPL-2.0; see RoboVerse third-party notices.\n",
            encoding="utf-8",
        )
        shutil.move(str(staging), output)
    print(
        f"Imported {pack['source']['run_configuration_count']} run configs, "
        f"{pack['source']['task_instance_count']} scene instances, "
        f"{len(pack['asset_ids'])} unique asset IDs -> {output}"
    )


if __name__ == "__main__":
    main()
