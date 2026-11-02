"""Index source benchmark configurations for the RoboVerse task migration."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.icl.import_geniesim_specs import (
    _revision,
    imported_payload_sha256,
    source_content_sha256,
    source_tree_content_sha256,
)

_GENIESIM_VALIDATED_SOURCE_CONTENT_SHA256 = "95ffb5b92eb03b38824581cf64b83d83d983296110a0fd2cab90aca2aeafda2c"
_GENIESIM_VALIDATED_IMPORT_PAYLOAD_SHA256 = "8ee1bead07b566db5f7367cded56c3442fd918b6c28cf3e5434b06f037d70bba"
_G1_SMOKE_TESTED_TASKS = {
    "bimanual_chip_handover",
    "pick_specific_object",
    "place_object_into_box_color",
    "pack_in_supermarket",
    "place_block_into_drawer",
    "pick_block_color",
    "pick_block_size",
    "sort_fruit",
}
_G2_ASSET_ALIAS_RUNS = {
    "g2op_if_pick_object_type",
    "g2op_robust_backgen_pick_object_type",
    "g2op_robust_camposgen_pick_object_type",
    "g2op_robust_camqualgen_pick_object_type",
    "g2op_robust_instructgen_pick_object_type",
    "g2op_robust_posegen_pick_object_type",
}
_BACKENDS = ("mujoco", "genesis", "newton")


def _run_migration_status(
    robot: str, subtask: str, run_id: str, content_sha256: str, payload_sha256: str | None
) -> str:
    if (
        content_sha256 != _GENIESIM_VALIDATED_SOURCE_CONTENT_SHA256
        or payload_sha256 != _GENIESIM_VALIDATED_IMPORT_PAYLOAD_SHA256
    ):
        return "implemented_unverified" if robot in {"g1_op", "g2_op"} else "not_started"
    if robot == "g1_op" and subtask in _G1_SMOKE_TESTED_TASKS:
        return "smoke_tested"
    if robot != "g2_op":
        return "not_started"
    if run_id in _G2_ASSET_ALIAS_RUNS:
        return "smoke_tested_with_asset_alias"
    return "smoke_tested"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _variant(stem: str) -> tuple[str, str | None]:
    match = re.match(r"^g[12]op_(s2r|if|manip|spatial|robust(?:_[a-z]+)?)_", stem)
    if match is None:
        raise ValueError(f"Unrecognized benchmark configuration family: {stem}")
    family = match.group(1)
    robustness = family.removeprefix("robust_") if family.startswith("robust_") else None
    return family, robustness


def _checker_vocabulary(task_root: Path) -> list[str]:
    """Find leaf action names used by the source checker graphs."""
    names: set[str] = set()

    def visit(node) -> None:
        if isinstance(node, dict):
            for name, value in node.items():
                if name == "Acts" or re.fullmatch(r"problem\d+", name):
                    visit(value)
                elif name in {"ActionList", "ActionSetWaitAny", "ActionSetWaitAll"} or name.startswith(
                    "ActionSetWaitSome"
                ):
                    visit(value)
                elif name == "ActionWaitForTime":
                    names.add(name)
                elif name not in {"Problem", "Init", "Objects"}:
                    names.add(name)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    for path in task_root.rglob("problems.json"):
        problem = json.loads(path.read_text(encoding="utf-8"))
        visit(problem)
    return sorted(names)


def _resolve_source_revision(source_root: Path, override: str | None = None) -> str:
    """Use a verified explicit upstream pin or the source checkout's own revision."""
    if override is None:
        return _revision(source_root)
    if not re.fullmatch(r"[0-9a-f]{40}", override):
        raise ValueError("source_revision must be a full lowercase 40-character Git commit SHA")
    return override


def build_catalog(
    source_root: Path,
    *,
    imported_spec: Path | None = None,
    source_revision: str | None = None,
) -> dict:
    """Create a provenance-linked inventory without copying upstream task code or assets."""
    source_root = source_root.resolve()
    config_root = source_root / "config"
    task_root = source_root / "benchmark" / "config" / "llm_task"
    eval_root = source_root / "benchmark" / "config" / "eval_tasks"
    if not config_root.is_dir() or not task_root.is_dir() or not eval_root.is_dir():
        raise FileNotFoundError(f"Expected config/, benchmark/config/llm_task/ and eval_tasks/ under {source_root}")

    runs = []
    semantic_tasks: dict[tuple[str, str], dict] = {}
    source_revision = _resolve_source_revision(source_root, source_revision)
    tree_sha256 = source_tree_content_sha256(source_root)
    active_payload_sha256 = None
    if imported_spec is not None:
        imported = json.loads(Path(imported_spec).read_text(encoding="utf-8"))
        metadata = imported.get("source", {})
        active_payload_sha256 = imported_payload_sha256(imported)
        if (
            metadata.get("content_sha256") != source_content_sha256(imported)
            or metadata.get("content_sha256") != tree_sha256
            or metadata.get("imported_payload_sha256") != active_payload_sha256
        ):
            raise ValueError("Imported benchmark does not match the supplied source tree and provenance hashes")
        if source_revision is not None and metadata.get("revision") != source_revision:
            raise ValueError("Imported benchmark source revision does not match the explicit catalog source revision")
    family_counts: Counter[str] = Counter()
    source_scene_instances: set[tuple[str, str]] = set()
    for config_path in sorted(config_root.glob("g[12]op_*.yaml")):
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        benchmark = data.get("benchmark", {}) if isinstance(data, dict) else {}
        robot, scene, subtask = (benchmark.get(key) for key in ("platform", "task_name", "sub_task_name"))
        if not all(isinstance(value, str) and value for value in (robot, scene, subtask)):
            raise ValueError(f"Missing platform, task_name or sub_task_name in {config_path.name}")
        family, robustness = _variant(config_path.stem)
        instance_root = task_root / subtask
        instances = (
            sorted(
                path.name for path in instance_root.iterdir() if path.is_dir() and (path / "scene_info.json").is_file()
            )
            if instance_root.is_dir()
            else []
        )
        source_scene_instances.update((subtask, instance) for instance in instances)
        eval_config = eval_root / f"{scene}.json"
        task_key = (robot, subtask)
        roboverse_task = "icl.geniesim_benchmark" if robot in {"g1_op", "g2_op"} else None
        run_id = config_path.stem
        migration_status = _run_migration_status(robot, subtask, run_id, tree_sha256, active_payload_sha256)
        backend_status = (
            "smoke_tested"
            if migration_status == "smoke_tested"
            else (
                migration_status
                if migration_status == "smoke_tested_with_asset_alias"
                else ("source_asset_missing" if migration_status == "source_asset_missing" else "not_run")
            )
        )
        task = semantic_tasks.setdefault(
            task_key,
            {
                "robot": robot,
                "subtask": subtask,
                "scene_families": [],
                "source_instances": instances,
                "source_definition": f"benchmark/config/llm_task/{subtask}/<instance>/",
                "run_ids": [],
                "roboverse_task": roboverse_task,
                "migration_status": migration_status,
                "backend_checks": {backend: backend_status for backend in _BACKENDS},
                "behavioral_parity": "not_verified",
            },
        )
        task["run_ids"].append(run_id)
        if scene not in task["scene_families"]:
            task["scene_families"].append(scene)
        family_counts[family] += 1
        runs.append({
            "id": run_id,
            "source_config": f"config/{config_path.name}",
            "source_config_sha256": _sha256(config_path),
            "family": family,
            "robustness_variant": robustness,
            "robot": robot,
            "scene_family": scene,
            "subtask": subtask,
            "evaluation_config": f"benchmark/config/eval_tasks/{eval_config.name}" if eval_config.is_file() else None,
            "roboverse_task": roboverse_task,
            "migration_status": migration_status,
            "backend_checks": {backend: backend_status for backend in _BACKENDS},
            "behavioral_parity": "not_verified",
        })

    if not runs:
        raise ValueError(f"No g1op/g2op benchmark YAML files found in {config_root}")
    for task in semantic_tasks.values():
        task_runs = [run for run in runs if run["id"] in task["run_ids"]]
        statuses = {run["migration_status"] for run in task_runs}
        if statuses == {"smoke_tested"}:
            task["migration_status"] = "smoke_tested"
        elif statuses == {"smoke_tested_with_asset_alias"}:
            task["migration_status"] = "smoke_tested_with_asset_alias"
        elif statuses == {"source_asset_missing"}:
            task["migration_status"] = "source_asset_missing"
        elif statuses == {"implemented_unverified"}:
            task["migration_status"] = "implemented_unverified"
        elif "smoke_tested" in statuses:
            task["migration_status"] = "partially_smoke_tested"
        else:
            task["migration_status"] = "not_started"
        task["backend_checks"] = {
            backend: (
                "smoke_tested"
                if all(run["backend_checks"][backend] == "smoke_tested" for run in task_runs)
                else "smoke_tested_with_asset_alias"
                if all(run["backend_checks"][backend] == "smoke_tested_with_asset_alias" for run in task_runs)
                else "source_asset_missing"
                if all(run["backend_checks"][backend] == "source_asset_missing" for run in task_runs)
                else "partially_smoke_tested"
                if any(run["backend_checks"][backend] == "smoke_tested" for run in task_runs)
                else "not_run"
            )
            for backend in _BACKENDS
        }

    unique_families = sorted({entry["family"] for entry in runs})
    smoke_tested_count = sum(run["migration_status"] == "smoke_tested" for run in runs)
    asset_blocked_count = sum(run["migration_status"] == "source_asset_missing" for run in runs)
    asset_alias_count = sum(run["migration_status"] == "smoke_tested_with_asset_alias" for run in runs)
    return {
        "format": "roboverse.icl.benchmark_catalog.v1",
        "source": {
            "name": "Genie Sim Benchmark",
            "repository": "https://github.com/AgibotTech/genie_sim",
            "revision": source_revision,
            "revision_scope": (
                "Task configs, evaluation configs, scene inputs, and benchmark runtime; see the GenieSim inventory "
                "audit for exact upstream tree comparison and local package differences."
            ),
            "content_sha256": tree_sha256,
            "license": "MPL-2.0",
            "config_root": "config/",
        },
        "scope": {
            "configuration_count": len(runs),
            "semantic_task_count": len(semantic_tasks),
            "configuration_family_counts": dict(sorted(family_counts.items())),
            "robot_configuration_counts": dict(sorted(Counter(run["robot"] for run in runs).items())),
            "scene_instance_count": len(source_scene_instances),
            "scene_instance_task_assignment_count": sum(
                len(task["source_instances"]) for task in semantic_tasks.values()
            ),
            "behavioral_parity_claimed": False,
            "shared_task_runtime": {
                "roboverse_task": "icl.geniesim_benchmark",
                "source_robots": ["g1_op", "g2_op"],
                "joint_position_action_dimensions": {"g1_op": 20, "g2_op": 24},
                "synthicl_action_dimensions": 7,
                "synthicl_action_spec": "tcp_world_delta_xyz_rotvec_gripper_open_v1",
                "bimanual_tcp_action_dimensions": 14,
                "action_modes": ["joint_position", "tcp_delta", "bimanual_tcp_delta"],
                "cartesian_controller_arm": "source_or_explicit_default_right",
                "configured_control_rate_hz": 30,
                "control_and_timing_parity": "not_verified",
                "supported_simulators": ["mujoco", "genesis", "newton"],
                "default_cameras": ["head_rgb", "head_fisheye_left", "head_fisheye_right"],
                "camera_projection_note": "OpenCV fisheye; G1 temporarily borrows G2 optical calibration",
                "supported_run_count": sum(run["roboverse_task"] == "icl.geniesim_benchmark" for run in runs),
                "smoke_tested_run_count": smoke_tested_count,
                "source_asset_blocked_run_count": asset_blocked_count,
                "smoke_tested_with_asset_alias_run_count": asset_alias_count,
                "validation_source_revision": "unknown",
                "validation_source_content_sha256": _GENIESIM_VALIDATED_SOURCE_CONTENT_SHA256,
                "validation_import_payload_sha256": _GENIESIM_VALIDATED_IMPORT_PAYLOAD_SHA256,
                "active_import_payload_sha256": active_payload_sha256,
                "validation_scope": "Historical reset/render/one-step checks used the legacy import without USD scene snapshots or task-specific reset poses.",
                "historical_smoke_tested_run_count": 80,
                "historical_smoke_with_asset_alias_run_count": 6,
                "status": (
                    "smoke_tested"
                    if smoke_tested_count == len(runs)
                    else "partially_smoke_tested"
                    if smoke_tested_count or asset_alias_count
                    else "implemented_unverified"
                ),
                "behavioral_parity": "not_verified",
                "limitations": [
                    "Eight pick_object_type scenes name waic_beverage_bottle_009 in metadata but their USD payloads explicitly reference genie_beverage_bottle_009; the legacy metadata-only import used an unverified compatibility alias",
                    "articulated source assets use converted URDF/MJCF approximations",
                    "source controller and demonstration parity",
                    "cross-backend PBR and room-rendering parity",
                    "physical G1 fisheye calibration and cross-backend visual parity",
                    "VLM scoring",
                ],
            },
        },
        "source_checker_vocabulary": _checker_vocabulary(task_root),
        "checker_status": {
            "prototype_predicates": ["Inside"],
            "implemented_checkers": [
                "Cover",
                "Follow",
                "InBBox",
                "Inside",
                "LiftUp",
                "MixedRules",
                "Onfloor",
                "Ontop",
                "PickUpOnGripper",
                "PushPull",
                "RelativePosition",
                "StableGrasp",
                "Stack",
                "StepOut",
                "Upright",
            ],
            "external_scoring_checkers": ["VLM"],
            "graph_operators": ["ActionList", "ActionSetWaitAny", "ActionSetWaitAll", "ActionSetWaitSome"],
            "parity_verified_predicates": [],
        },
        "semantic_tasks": [semantic_tasks[key] for key in sorted(semantic_tasks)],
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        required=True,
        type=Path,
        help="Path to the upstream geniesim_benchmark/src/geniesim_benchmark directory",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--imported-spec", type=Path, help="Optional imported benchmark.json to bind validation status")
    parser.add_argument(
        "--source-revision",
        help="Explicit upstream commit SHA verified with audit_geniesim_inventory.py for a source archive without .git",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; choose a new path")
    catalog = build_catalog(
        args.source_root,
        imported_spec=args.imported_spec,
        source_revision=args.source_revision,
    )
    output.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(catalog["scope"], ensure_ascii=False))


if __name__ == "__main__":
    main()
