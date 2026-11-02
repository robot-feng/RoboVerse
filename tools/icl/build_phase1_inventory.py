"""Build the source-to-RoboVerse Phase 1 ICL environment migration snapshot.

The matrix is an inventory of source environments and runtime status, not a
pass report. Phase 1 verifies assets, create/reset/step, finite state,
task-predicate positive/negative cases, camera rendering, and interactive
inspectability. It does not require a successful manipulation rollout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BACKENDS = ("mujoco", "genesis", "newton")
STATUSES = {"DISCOVERED", "MIGRATING", "READY_FOR_TEST", "VERIFIED", "FAILED", "BLOCKED", "UNSUPPORTED"}
MATRIX_FIELDS = (
    "source",
    "upstream_revision",
    "source_task_id",
    "source_config_id",
    "source_config",
    "task_variant",
    "source_robot_task_alias",
    "scene_id",
    "source_robot",
    "target_robot",
    "migrated_task_id",
    "simulator",
    "status",
    "asset_status",
    "runtime_status",
    "contract_status",
    "viewer_status",
    "blocker",
    "evidence",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _revision_for_env_alias(alias: str) -> str:
    if alias.startswith("L3_"):
        return alias[3:].replace("-v1", "L3-v1")
    if re.match(r"L[0-2]_", alias):
        return alias[3:]
    return alias


def _registered_ig_environments(source_root: Path) -> dict[str, str]:
    task_root = source_root / "mani_skill" / "envs" / "tasks" / "tabletop"
    environments: dict[str, str] = {}
    decorator = re.compile(r"@register_env\([\"']([^\"']+)")
    for subdirectory in ("dual_tasks", "dual_tasks_l3"):
        directory = task_root / subdirectory
        for path in sorted(directory.glob("*.py")):
            for match in decorator.finditer(path.read_text(encoding="utf-8", errors="replace")):
                environments[match.group(1)] = path.relative_to(source_root).as_posix()
    return environments


def _ig_row_status(human_id: str, level: str, data_root: Path) -> tuple[str, str, str, str]:
    """Return status, asset status, target task, and current blocker for IG."""
    asset_root = data_root / "icl" / "objects" / "imitator_game"
    if human_id == "human_H7":
        if level == "L0":
            return (
                "BLOCKED",
                "apple wrapper mass mismatch; exact basket wrapper missing",
                "imitator_game.h7.pick_apple_basket.l0",
                "Prepare the exact pinned RoboTwin breadbasket and correct H7's apple mass before backend validation.",
            )
        if level in {"L1", "L2"}:
            asset_detail = (
                "L1 source offset is not implemented"
                if level == "L1"
                else "exact pear wrapper and L2 adapter are missing"
            )
            return (
                "MIGRATING",
                asset_detail,
                "",
                "Port this level's source scene/object setup and reset behavior; add positive/negative predicate checks before verification.",
            )
        return (
            "BLOCKED",
            "exact plastic-box model 2 wrapper missing",
            "imitator_game.h7.pick_apple_basket.l3",
            "Acquire and prepare the exact source plastic-box model 2, then check its source geometry and mass.",
        )
    if human_id == "human_H19":
        if level == "L3":
            expected = asset_root / "h19" / "robotwin_prepared" / "023_tissue-box" / "model_3" / "object.urdf"
            detail = (
                "exact prepared tissue-box wrapper missing" if not expected.is_file() else "prepared wrapper present"
            )
            return (
                "BLOCKED" if not expected.is_file() else "MIGRATING",
                detail,
                "imitator_game.h19.clean_desk.l3",
                "Convert the exact model-3 collision decomposition without substituting a hull; the current converter rejects the source non-convex components.",
            )
        return (
            "MIGRATING",
            "task-specific YCB wrappers are present; simulator loading is unverified",
            f"imitator_game.h19.clean_desk.{level.lower()}",
            "Record runtime and viewer checks on each backend, and add positive/negative task-predicate cases before verification.",
        )
    return (
        "DISCOVERED",
        "required asset set not migrated or validated",
        "",
        "No RoboVerse scene adapter for this source family; migrate its exact assets and scene setup before runtime checks.",
    )


def _ig_rows(source_root: Path, mapping_path: Path, data_root: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    mapping = _read_json(mapping_path)["task_mappings"]
    registered = _registered_ig_environments(source_root)
    rows: list[dict[str, str]] = []
    mapped_envs: set[str] = set()
    human_without_scene = []
    mapped_alias_count = 0
    mapped_robot_alias_count = 0
    for item in mapping:
        human_id = item["human_task_id"]
        aliases = item.get("sim_task_id", [])
        robot_aliases = item.get("robot_task_id", [])
        robot_by_level: dict[str, list[str]] = {}
        for robot_alias in robot_aliases:
            match = re.search(r"_(L[0-3])(?:_(\d+))?$", robot_alias)
            if match:
                suffix = f"_{match.group(2)}" if match.group(2) else ""
                robot_by_level.setdefault(match.group(1), []).append(f"{match.group(1)}{suffix}")
            else:
                robot_by_level.setdefault("unknown", []).append("unknown")
        sim_by_level: dict[str, list[str]] = {}
        for alias in aliases:
            match = re.match(r"^(L[0-3])_", alias)
            level = match.group(1) if match else "unknown"
            sim_by_level.setdefault(level, []).append(alias)

        if robot_aliases and not aliases:
            human_without_scene.append({"human_task_id": human_id, "robot_task_ids": robot_aliases})
        for level in sorted(set(robot_by_level) | set(sim_by_level)):
            level_robot_aliases = robot_by_level.get(level, [])
            level_sim_aliases = sim_by_level.get(level, [])
            if not level_sim_aliases:
                for robot_alias in level_robot_aliases:
                    for backend in BACKENDS:
                        rows.append(
                            _row(
                                source="The-Imitator-Game",
                                revision="3c3404af98727f6d4783537732d0db8e973dfba5",
                                task_id=human_id,
                                config_id="",
                                source_config="",
                                variant=robot_alias,
                                source_robot_task_alias=robot_alias,
                                scene_id="",
                                source_robot="panda_wristcam_pair",
                                target_robot="",
                                target_task="",
                                simulator=backend,
                                status="BLOCKED",
                                asset_status="not_applicable; source simulator task ID absent",
                                runtime_status="NOT_APPLICABLE_SOURCE_SCENE_MISSING",
                                contract_status="NOT_TESTED",
                                viewer_status="NOT_RUN",
                                blocker="The source task mapping has a robot-task alias but no simulator task ID or executable scene for this level.",
                                evidence="../icl/The-Imitator-Game/examples/baselines/lerobot_dataset/task_mapping.json",
                            )
                        )
                continue

            for alias in level_sim_aliases:
                env_id = _revision_for_env_alias(alias)
                mapped_envs.add(env_id)
                mapped_alias_count += 1
                if env_id not in registered:
                    raise ValueError(f"The-Imitator-Game mapping points to unregistered environment {env_id!r}")
                matched_robot_aliases = level_robot_aliases or [""]
                for robot_alias in matched_robot_aliases:
                    if robot_alias:
                        mapped_robot_alias_count += 1
                    status, asset_status, target_task, blocker = _ig_row_status(human_id, level, data_root)
                    if not robot_alias:
                        status = "BLOCKED"
                        asset_status = "source simulator environment exists; source robot-task alias missing"
                        target_task = ""
                        blocker = "The simulator mapping lists this level but the human-task mapping has no robot-task alias for it."
                    target_robot = "roboverse_franka_pair" if human_id in {"human_H7", "human_H19"} else ""
                    for backend in BACKENDS:
                        rows.append(
                            _row(
                                source="The-Imitator-Game",
                                revision="3c3404af98727f6d4783537732d0db8e973dfba5",
                                task_id=human_id,
                                config_id=alias,
                                source_config=registered[env_id],
                                variant=robot_alias or level,
                                source_robot_task_alias=robot_alias,
                                scene_id=env_id,
                                source_robot="panda_wristcam_pair",
                                target_robot=target_robot,
                                target_task=target_task,
                                simulator=backend,
                                status=status,
                                asset_status=asset_status,
                                runtime_status="NOT_RUN",
                                contract_status="NOT_TESTED",
                                viewer_status="NOT_RUN",
                                blocker=blocker,
                                evidence=(
                                    "roboverse_pack/tasks/imitator_game/README.md; "
                                    "tests/test_imitator_game.py (configuration/asset contract tests only)"
                                    if human_id in {"human_H7", "human_H19"}
                                    else "../icl/The-Imitator-Game/examples/baselines/lerobot_dataset/task_mapping.json"
                                ),
                            )
                        )

    unmapped_environment_ids = sorted(set(registered) - mapped_envs)
    for env_id in unmapped_environment_ids:
        for backend in BACKENDS:
            rows.append(
                _row(
                    source="The-Imitator-Game",
                    revision="3c3404af98727f6d4783537732d0db8e973dfba5",
                    task_id="",
                    config_id=env_id,
                    source_config=registered[env_id],
                    variant="UNMAPPED",
                    scene_id=env_id,
                    source_robot="panda_wristcam_pair",
                    target_robot="",
                    target_task="",
                    simulator=backend,
                    status="BLOCKED",
                    asset_status="not inventoried for RoboVerse",
                    runtime_status="NOT_RUN",
                    contract_status="NOT_TESTED",
                    viewer_status="NOT_RUN",
                    blocker="Registered source simulator environment has no human_task_id mapping in the pinned source task catalog.",
                    evidence="../icl/The-Imitator-Game/mani_skill/envs/tasks/tabletop/dual_tasks",
                )
            )

    counts = {
        "human_task_mapping_count": len(mapping),
        "human_mappings_without_simulator_environment": len(human_without_scene),
        "human_mappings_without_simulator_environment_ids": [entry["human_task_id"] for entry in human_without_scene],
        "mapped_simulator_level_alias_count": mapped_alias_count,
        "source_robot_task_aliases_with_simulator_mapping": mapped_robot_alias_count,
        "source_robot_task_alias_count": sum(len(item.get("robot_task_id", [])) for item in mapping),
        "source_code_comparison_evidence": "examples/output/qa_runs/icl_phase1_inventory_20261102/upstream_source_comparison.json",
        "simulator_robot_alias_mismatches": [
            {"human_task_id": human_id, "level": "L3", "issue": "simulator alias exists without robot-task alias"}
            for human_id in ("human_H29", "human_H33", "human_H36", "human_H39", "human_H43")
        ],
        "simulator_aliases_without_robot_task_mapping": 5,
        "robot_aliases_without_simulator_mapping": ["human_H25", "human_H37", "human_H38"],
        "robot_task_aliases_without_simulator_mapping": 11,
        "registered_base_environment_count": sum(
            path.startswith("mani_skill/envs/tasks/tabletop/dual_tasks/") for path in registered.values()
        ),
        "registered_l3_environment_count": sum(
            path.startswith("mani_skill/envs/tasks/tabletop/dual_tasks_l3/") for path in registered.values()
        ),
        "registered_source_environment_count": len(registered),
        "registered_environments_without_human_mapping": unmapped_environment_ids,
        "source_robot": "panda_wristcam_pair",
        "current_target_robot_realization": "roboverse_franka_pair (not source wrist-camera embodiment)",
    }
    return rows, counts


def _synth_rows(source_root: Path, data_root: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    config_path = source_root / "configs" / "all_objects_skills.yaml"
    lines = config_path.read_text(encoding="utf-8").splitlines()
    task_types: list[str] = []
    in_task = False
    in_types = False
    for line in lines:
        if line == "task:":
            in_task, in_types = True, False
            continue
        if in_task and line.startswith("environment:"):
            break
        if in_task and line.strip() == "types:":
            in_types = True
            continue
        if in_task and in_types:
            match = re.match(r"^\s{4}([a-z0-9_]+):\s*[0-9.]+\s*$", line)
            if match:
                task_types.append(match.group(1))
            elif line and not line.startswith("#") and not line.startswith("    "):
                in_types = False
    expected = ("pick_and_place_gn", "push_gn", "pour_gn", "random_gn")
    if tuple(task_types) != expected:
        raise ValueError(f"Unexpected SynthICL skill list in {config_path}: {task_types!r}")

    profiles = ("G1", "G2")
    asset_ids = (
        "synthicl_bowl_base1",
        "synthicl_cup_base1",
        "synthicl_tea_box_base2",
        "synthicl_woodenblock_small_base3",
    )
    asset_root = data_root / "icl" / "objects"
    missing = [asset for asset in asset_ids if not (asset_root / asset / "object.urdf").is_file()]
    rows = []
    for task_id in task_types:
        for robot in profiles:
            for backend in BACKENDS:
                rows.append(
                    _row(
                        source="SynthICL pseudo-demo",
                        revision="131d22f9a4a8f066aa2fd7bee8d88118dcc0766a",
                        task_id=task_id,
                        config_id="configs/all_objects_skills.yaml",
                        source_config="configs/all_objects_skills.yaml",
                        variant="default",
                        scene_id=task_id,
                        source_robot="franka",
                        target_robot=robot,
                        target_task="",
                        simulator=backend,
                        status="MIGRATING",
                        asset_status=(
                            "four task objects have local prepared wrappers"
                            if not missing
                            else f"missing wrappers: {', '.join(missing)}"
                        ),
                        runtime_status="NOT_RUN",
                        contract_status="NOT_IMPLEMENTED",
                        viewer_status="NOT_RUN",
                        blocker="RoboVerse currently has static scene profiles only; the source task's scene-specific objects/layout and per-backend environment checks still need completion.",
                        evidence="roboverse_pack/tasks/icl/icl_static_scenes.py; tests/test_icl_static_scenes.py (static configuration tests only)",
                    )
                )
    return rows, {
        "source_revision": "131d22f9a4a8f066aa2fd7bee8d88118dcc0766a",
        "source_code_comparison_evidence": "examples/output/qa_runs/icl_phase1_inventory_20261102/upstream_source_comparison.json",
        "skills": list(task_types),
        "static_scene_profiles": list(task_types),
        "target_robots": list(profiles),
        "local_task_object_ids": list(asset_ids),
        "missing_task_object_wrappers": missing,
        "task_execution_and_predicates_implemented": False,
        "dtd_textures_available": (data_root / "icl" / "textures" / "dtd" / "images").is_dir(),
    }


def _genie_rows(catalog_path: Path, audit_path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    catalog = _read_json(catalog_path)
    audit = _read_json(audit_path)
    run_records = {run["id"]: run for run in catalog["runs"]}
    semantic_tasks = {(task["robot"], task["subtask"]): task for task in catalog["semantic_tasks"]}
    rows = []
    pair_assignments = 0
    for run_id, run in run_records.items():
        pair = semantic_tasks[(run["robot"], run["subtask"])]
        pair_assignments += len(pair["source_instances"])
        target_robot = "geniesim_g1_omnipicker_closed_v11" if run["robot"] == "g1_op" else "geniesim_g2_omnipicker_v11"
        for scene_instance in pair["source_instances"]:
            scene_id = f"{run['subtask']}/{scene_instance}"
            for backend in BACKENDS:
                rows.append(
                    _row(
                        source="GenieSim benchmark",
                        revision=catalog["source"]["revision"],
                        task_id=f"{run['robot']}:{run['subtask']}",
                        config_id=run_id,
                        source_config=run["source_config"],
                        variant=run["family"]
                        + (f":{run['robustness_variant']}" if run.get("robustness_variant") else ""),
                        scene_id=scene_id,
                        source_robot=run["robot"],
                        target_robot=target_robot,
                        target_task="icl.geniesim_benchmark",
                        simulator=backend,
                        status="MIGRATING",
                        asset_status="source wrapper manifest present; runtime mesh/collision checks not established",
                        runtime_status="NOT_RUN",
                        contract_status="NOT_TESTED",
                        viewer_status="NOT_RUN",
                        blocker="No reproducible asset-resolution, create/reset/step, finite-state, rendered-preview, and viewer evidence is bound to this run x scene x robot x backend combination.",
                        evidence="roboverse_pack/tasks/icl/benchmark_catalog.json; examples/output/qa_runs/icl_phase1_inventory_20261102/geniesim_inventory_upstream_comparison.json",
                    )
                )

    config_coverage = audit["source_config_coverage"]
    scene_coverage = audit["scene_instance_coverage"]
    asset_coverage = audit["asset_coverage"]
    run_count = len(run_records)
    if run_count != config_coverage["source_count"] or run_count != 86:
        raise ValueError("GenieSim source config inventory and catalog disagree")
    return rows, {
        "source_revision": catalog["source"]["revision"],
        "run_config_count": run_count,
        "config_hash_matches": config_coverage["config_hash_status_counts"].get("match", 0),
        "semantic_task_pair_count": len(catalog["semantic_tasks"]),
        "unique_scene_count": scene_coverage["source_unique_scene_count"],
        "task_scene_assignments": scene_coverage["source_task_assignment_count"],
        "run_config_scene_assignments": pair_assignments,
        "scene_source_files_present": scene_coverage["required_source_file_presence"],
        "source_payload_asset_count": asset_coverage["source_payload_asset_count"],
        "asset_wrapper_resolution_counts": asset_coverage["resolution_status_counts"],
        "scene_assignments_with_unresolved_asset_wrappers": len(
            asset_coverage["scenes_with_unresolved_asset_wrappers"]
        ),
        "imported_spec_bundle_records_individually_verified": all(
            item["individual_records_verified"] for item in audit["imported_spec_bundle_files"]
        ),
        "backend_run_status_counts": audit["semantic_task_coverage"]["backend_status_declarations"],
        "upstream_runtime_comparison": audit["source"]["upstream_package_comparison"]["categories"],
    }


def _row(**values: str) -> dict[str, str]:
    row = {field: "" for field in MATRIX_FIELDS}
    aliases = {
        "revision": "upstream_revision",
        "task_id": "source_task_id",
        "config_id": "source_config_id",
        "variant": "task_variant",
        "target_task": "migrated_task_id",
    }
    for key, value in values.items():
        normalized_key = aliases.get(key, key)
        if normalized_key not in row:
            raise ValueError(f"Unexpected migration-matrix field {key!r}")
        row[normalized_key] = value
    if row["status"] not in STATUSES:
        raise ValueError(f"Invalid migration status {row['status']!r}")
    if row["simulator"] not in BACKENDS:
        raise ValueError(f"Invalid simulator {row['simulator']!r}")
    return row


def _git_value(arguments: tuple[str, ...], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(["git", *arguments], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _package_versions() -> dict[str, str | None]:
    names = ("roboverse-metasim", "roboverse-py", "mujoco", "genesis-world", "newton", "mujoco-warp", "torch")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _comparison_summary(report_path: Path, source_name: str) -> dict[str, Any]:
    report = _read_json(report_path)
    try:
        item = next(record for record in report["results"] if record["source"] == source_name)
    except (KeyError, StopIteration) as exc:
        raise ValueError(f"No upstream comparison for {source_name!r} in {report_path}") from exc
    return {
        "pinned_revision": item["pinned_revision"],
        "upstream_file_count": item["upstream_file_count"],
        "matched_file_count": item["matched_file_count"],
        "missing_local_files": item["missing_local_files"],
        "content_mismatches": item["content_mismatches"],
        "local_only_files": item["local_only_files"],
        "scopes": item["scopes"],
    }


def build(
    output_dir: Path,
    ig_root: Path,
    synth_root: Path,
    genie_catalog_path: Path,
    genie_audit_path: Path,
    source_comparison_path: Path,
) -> dict[str, Any]:
    data_root = ROOT / "roboverse_data"
    ig_mapping_path = ig_root / "examples" / "baselines" / "lerobot_dataset" / "task_mapping.json"
    ig_rows, ig_summary = _ig_rows(ig_root, ig_mapping_path, data_root)
    synth_rows, synth_summary = _synth_rows(synth_root, data_root)
    genie_rows, genie_summary = _genie_rows(genie_catalog_path, genie_audit_path)
    ig_summary["source_file_comparison"] = _comparison_summary(source_comparison_path, "The-Imitator-Game")
    synth_summary["source_file_comparison"] = _comparison_summary(source_comparison_path, "SynthICL pseudo-demo")

    custom_rows = [
        _row(
            source="User-supplied GenieSim G1 apple scene",
            revision="user-provided g1_apple_scene_delivery_20261031_v4",
            task_id="g1_apple_scene_delivery_20261031_v4",
            config_id="roboverse_data/icl/scenes/g1_apple_scene_delivery_20261031_v4",
            source_config="roboverse_data/icl/scenes/g1_apple_scene_delivery_20261031_v4",
            variant="default",
            scene_id="g1_apple_scene_delivery_20261031_v4",
            source_robot="G1",
            target_robot="geniesim_g1_omnipicker_closed_v11",
            target_task="icl.geniesim_g1_apple_delivery",
            simulator=backend,
            status="MIGRATING",
            asset_status="canonical local scene/robot assets present; current preview evidence is not hash-bound",
            runtime_status="PARTIAL_UNBOUND",
            contract_status="NOT_TESTED",
            viewer_status="NOT_RUN",
            blocker="Existing short scene previews do not bind the exact code tree and asset hashes or record the Phase 1 viewer-inspectability check.",
            evidence="examples/output/ICL_Benchmark/environments/g1_apple_scene/; examples/output/qa_runs/icl_unified_migration_20261102/ground_alignment_v11/",
        )
        for backend in BACKENDS
    ]
    rows = ig_rows + synth_rows + genie_rows + custom_rows
    keys = [
        tuple(
            row[field]
            for field in (
                "source",
                "source_task_id",
                "source_config_id",
                "task_variant",
                "scene_id",
                "source_robot",
                "target_robot",
                "simulator",
            )
        )
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Migration matrix contains duplicate source/task/config/scene/robot/backend rows")

    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "migration_matrix.csv"
    with matrix_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MATRIX_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)

    statuses = Counter(row["status"] for row in rows)
    inventory = {
        "format": "roboverse.icl.phase1_inventory.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": "Phase 1 environment migration: source inventory, assets, create/reset/step, finite state, task-predicate positive/negative cases, camera rendering, backend, and viewer inspectability. No manipulation rollout, replay, data collection, or training.",
        "worktree": {
            "repository_root": ".",
            "git_commit": _git_value(("rev-parse", "HEAD"), ROOT),
            "git_branch": _git_value(("branch", "--show-current"), ROOT),
            "note": "The worktree contains unrelated pre-existing staged and unstaged changes; the commit hash identifies the base revision only.",
        },
        "runtime_environment": {
            "python": sys.version,
            "executable": str(Path(sys.executable).resolve()),
            "package_versions": _package_versions(),
        },
        "sources": {
            "the_imitator_game": ig_summary,
            "synthicl_pseudo_demo": synth_summary,
            "geniesim_benchmark": genie_summary,
            "user_supplied_g1_apple_scene": {
                "environment_id": "g1_apple_scene_delivery_20261031_v4",
                "current_backend_previews": list(BACKENDS),
                "phase1_verification_status": "MIGRATING",
                "runtime_evidence_binding": "incomplete",
                "task_contract_phase": "predicate semantics and positive/negative cases are required; no manipulation planning is required",
            },
        },
        "upstream_source_comparison": {
            "path": str(source_comparison_path.relative_to(ROOT))
            if source_comparison_path.is_relative_to(ROOT)
            else str(source_comparison_path),
            "sha256": _sha256(source_comparison_path),
            "method_scope_note": _read_json(source_comparison_path).get("method", ""),
        },
        "matrix": {
            "path": "migration_matrix.csv",
            "sha256": _sha256(matrix_path),
            "row_count": len(rows),
            "rows_by_status": dict(sorted(statuses.items())),
            "rows_by_source": dict(sorted(Counter(row["source"] for row in rows).items())),
            "verified_row_count": statuses.get("VERIFIED", 0),
            "runtime_passed_row_count": 0,
            "contract_passed_row_count": sum(row["contract_status"] == "PASS" for row in rows),
            "note": "No row is VERIFIED by this static inventory. `NOT_RUN` and `PARTIAL_UNBOUND` are not passes.",
        },
        "output_policy": {
            "gallery": "examples/output/ICL_Benchmark/environments/",
            "gallery_contents": "rendered PNG/GIF/MP4 previews and a short README only; GIFs are derived from overview MP4s; no copied source assets/configs/logs/reports",
            "qa_evidence": "examples/output/qa_runs/",
            "publish_rule": "publish an environment only after every declared Phase 1 backend has passing commit/asset-bound asset-resolution, create/reset/step, finite-state, task-predicate positive/negative, viewer-interaction, and rendered-preview evidence; manipulation rollouts are not required",
        },
    }
    inventory_path = output_dir / "inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ig-source", type=Path, default=ROOT.parent / "icl" / "The-Imitator-Game")
    parser.add_argument("--synth-source", type=Path, default=ROOT.parent / "icl" / "synthicl-pseudo-demo")
    parser.add_argument(
        "--geniesim-catalog",
        type=Path,
        default=ROOT / "roboverse_pack" / "tasks" / "icl" / "benchmark_catalog.json",
    )
    parser.add_argument(
        "--geniesim-audit",
        type=Path,
        default=ROOT
        / "examples"
        / "output"
        / "qa_runs"
        / "icl_phase1_inventory_20261102"
        / "geniesim_inventory_upstream_comparison.json",
    )
    parser.add_argument(
        "--source-comparison-report",
        type=Path,
        default=ROOT
        / "examples"
        / "output"
        / "qa_runs"
        / "icl_phase1_inventory_20261102"
        / "upstream_source_comparison.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "examples" / "output" / "qa_runs" / "icl_phase1_inventory_20261102",
    )
    args = parser.parse_args()
    inventory = build(
        output_dir=args.output_dir.resolve(),
        ig_root=args.ig_source.resolve(),
        synth_root=args.synth_source.resolve(),
        genie_catalog_path=args.geniesim_catalog.resolve(),
        genie_audit_path=args.geniesim_audit.resolve(),
        source_comparison_path=args.source_comparison_report.resolve(),
    )
    print(json.dumps(inventory["matrix"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
