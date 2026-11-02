"""Audit GenieSim source coverage without loading a simulator or large spec bundle.

This is a file-level inventory only. It verifies source configuration hashes,
scene input presence, prepared asset manifests, and catalog declarations. It
does not import the benchmark JSON into memory, build scenes, launch a backend,
or establish task success/behavioral parity.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

BACKENDS = ("mujoco", "genesis", "newton")
SCENE_FILES = ("scene_info.json", "problems.json", "instructions.json", "scene.usda")
PAYLOAD_RE = re.compile(r"(?:prepend|append)\s+payload\s*=\s*@([^@]+)@")
UPSTREAM_PACKAGE_PREFIX = "source/geniesim_benchmark"
LOCAL_PACKAGE_PREFIX = "src/geniesim_benchmark"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_git_revision(source_root: Path) -> tuple[str | None, bool]:
    """Return a revision only if the source checkout has its own Git metadata."""
    try:
        root = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if Path(root).resolve() != source_root.resolve():
            return None, False
        revision = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return revision, True
    except (OSError, subprocess.CalledProcessError):
        return None, False


def _git_blob_sha1(path: Path) -> str:
    """Compute the object ID Git assigns to one regular file or symlink."""
    if path.is_symlink():
        content = os.fsencode(os.readlink(path))
        size = len(content)
        digest = hashlib.sha1(b"blob " + str(size).encode("ascii") + b"\0" + content)
        return digest.hexdigest()

    size = path.stat().st_size
    digest = hashlib.sha1()
    digest.update(b"blob " + str(size).encode("ascii") + b"\0")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _upstream_tree(repo: Path, revision: str) -> tuple[str, dict[str, str]]:
    """Resolve one exact commit and read the benchmark package's tracked blob IDs."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("upstream_revision must be a full lowercase 40-character Git commit SHA")
    repo = Path(repo).resolve()
    try:
        resolved = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--verify", f"{revision}^{{commit}}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if resolved != revision:
            raise ValueError(f"Upstream revision resolved to {resolved}, not the requested {revision}")
        output = subprocess.run(
            ["git", "-C", str(repo), "ls-tree", "-r", "-z", resolved, "--", UPSTREAM_PACKAGE_PREFIX],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"Cannot read pinned GenieSim tree {revision} from Git repository {repo}: {exc}") from exc

    prefix = UPSTREAM_PACKAGE_PREFIX + "/"
    tree: dict[str, str] = {}
    for record in output.split(b"\0"):
        if not record:
            continue
        metadata, path_bytes = record.split(b"\t", 1)
        mode, kind, object_id = metadata.decode("ascii").split()
        if kind != "blob":
            continue
        path = os.fsdecode(path_bytes)
        if path.startswith(prefix):
            tree[path[len(prefix) :]] = object_id
    if not tree:
        raise ValueError(f"Pinned commit {revision} has no {UPSTREAM_PACKAGE_PREFIX}/ tree")
    return resolved, tree


def _compare_source_package(source_root: Path, upstream_tree: dict[str, str]) -> dict[str, Any]:
    """Compare the local package snapshot with upstream and classify task-critical files."""
    package_root = Path(source_root).resolve().parents[1]
    local_tree: dict[str, str] = {}
    for path in package_root.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        local_tree[path.relative_to(package_root).as_posix()] = _git_blob_sha1(path)

    def compare_paths(paths: set[str]) -> dict[str, Any]:
        local_paths = paths & set(local_tree)
        upstream_paths = paths & set(upstream_tree)
        shared = local_paths & upstream_paths
        mismatches = sorted(path for path in shared if local_tree[path] != upstream_tree[path])
        local_only = sorted(local_paths - upstream_paths)
        upstream_only = sorted(upstream_paths - local_paths)
        matched = len(shared) - len(mismatches)
        return {
            "local_file_count": len(local_paths),
            "upstream_file_count": len(upstream_paths),
            "content_match_count": matched,
            "content_mismatch_count": len(mismatches),
            "local_only_count": len(local_only),
            "upstream_only_count": len(upstream_only),
            "exact_match": not (mismatches or local_only or upstream_only),
            "content_mismatch_files": mismatches,
            "local_only_files": local_only,
            "upstream_only_files": upstream_only,
        }

    all_paths = set(local_tree) | set(upstream_tree)
    categories = {
        "run_configs": {
            path
            for path in all_paths
            if path.startswith(f"{LOCAL_PACKAGE_PREFIX}/config/")
            and Path(path).name.startswith(("g1op_", "g2op_"))
            and path.endswith(".yaml")
        },
        "evaluation_configs": {
            path
            for path in all_paths
            if path.startswith(f"{LOCAL_PACKAGE_PREFIX}/benchmark/config/eval_tasks/") and path.endswith(".json")
        },
        "scene_inputs": {
            path for path in all_paths if path.startswith(f"{LOCAL_PACKAGE_PREFIX}/benchmark/config/llm_task/")
        },
        "benchmark_runtime_python": {
            path for path in all_paths if path.startswith(f"{LOCAL_PACKAGE_PREFIX}/benchmark/") and path.endswith(".py")
        },
        "benchmark_skills": {path for path in all_paths if path.startswith("skills/")},
        "whole_package": all_paths,
    }
    return {
        "local_package_root_relative_to_source_root": "../../",
        "upstream_package_prefix": UPSTREAM_PACKAGE_PREFIX,
        "categories": {name: compare_paths(paths) for name, paths in categories.items()},
    }


def _scene_adapter_aliases(adapter_path: Path | None) -> dict[str, dict[str, Any]]:
    """Read only the literal compatibility-alias mapping; never import the adapter."""
    if adapter_path is None or not adapter_path.is_file():
        return {}
    tree = ast.parse(adapter_path.read_text(encoding="utf-8"), filename=str(adapter_path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_COMPATIBILITY_ASSET_ALIASES" for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            return value if isinstance(value, dict) else {}
    return {}


def _source_scene_assets(scene_path: Path) -> list[str]:
    """Read asset IDs from authored USD payload paths, without opening USD."""
    if not scene_path.is_file():
        return []
    content = scene_path.read_text(encoding="utf-8", errors="replace")
    asset_ids = set()
    for raw_path in PAYLOAD_RE.findall(content):
        normalized = raw_path.replace("\\", "/").rstrip("/")
        asset_id = PurePosixPath(normalized).parent.name
        if asset_id and asset_id not in {"objects", "geniesim_assets"}:
            asset_ids.add(asset_id)
    return sorted(asset_ids)


def _asset_manifest(asset_root: Path, asset_id: str) -> dict[str, Any]:
    manifest_path = asset_root / asset_id / "provenance.json"
    if not manifest_path.is_file():
        return {"available": False, "manifest_valid": False}
    try:
        manifest = _json(manifest_path)
    except (OSError, json.JSONDecodeError):
        return {"available": True, "manifest_valid": False}
    valid = (
        isinstance(manifest, dict)
        and manifest.get("format") == "roboverse.icl.geniesim_rigid_object.v3"
        and manifest.get("asset_id") == asset_id
        and isinstance(manifest.get("source_revision"), str)
    )
    return {
        "available": True,
        "manifest_valid": valid,
        "manifest_sha256": _sha256(manifest_path),
        "source_revision": manifest.get("source_revision") if isinstance(manifest, dict) else None,
        "source_asset_directory": manifest.get("source_asset_directory") if isinstance(manifest, dict) else None,
    }


def _catalog_task_map(catalog: dict) -> dict[tuple[str, str], dict]:
    return {
        (str(item.get("robot", "")), str(item.get("subtask", ""))): item
        for item in catalog.get("semantic_tasks", [])
        if isinstance(item, dict)
    }


def build_audit(
    *,
    source_root: Path,
    catalog_path: Path,
    extensions_path: Path | None = None,
    assets_root: Path | None = None,
    scene_adapter_path: Path | None = None,
    spec_bundles: list[Path] | None = None,
    source_artifact_root: Path | None = None,
    upstream_repo: Path | None = None,
    upstream_revision: str | None = None,
) -> dict[str, Any]:
    """Compare a pinned source checkout with RoboVerse inventory declarations."""
    source_root = source_root.resolve()
    catalog_path = catalog_path.resolve()
    catalog = _json(catalog_path)
    if catalog.get("format") != "roboverse.icl.benchmark_catalog.v1":
        raise ValueError(f"Unsupported benchmark catalog format in {catalog_path}")

    config_root = source_root / "config"
    task_root = source_root / "benchmark" / "config" / "llm_task"
    eval_root = source_root / "benchmark" / "config" / "eval_tasks"
    if not config_root.is_dir() or not task_root.is_dir() or not eval_root.is_dir():
        raise FileNotFoundError("Expected config/, benchmark/config/llm_task/ and eval_tasks/ under source root")

    source_configs = sorted(config_root.glob("g[12]op_*.yaml"))
    if (upstream_repo is None) != (upstream_revision is None):
        raise ValueError("upstream_repo and upstream_revision must be supplied together")
    source_reference_comparison = None
    resolved_upstream_revision = None
    if upstream_repo is not None and upstream_revision is not None:
        resolved_upstream_revision, upstream_tree = _upstream_tree(upstream_repo, upstream_revision)
        source_reference_comparison = _compare_source_package(source_root, upstream_tree)
    catalog_runs = catalog.get("runs", [])
    runs_by_id = {str(run.get("id")): run for run in catalog_runs if isinstance(run, dict)}
    source_run_ids: set[str] = set()
    source_semantic_pairs: set[tuple[str, str]] = set()
    source_subtasks: set[str] = set()
    eval_refs: set[str] = set()
    source_run_records: list[dict[str, Any]] = []

    for path in source_configs:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        benchmark = data.get("benchmark", {}) if isinstance(data, dict) else {}
        robot = benchmark.get("platform")
        scene_family = benchmark.get("task_name")
        subtask = benchmark.get("sub_task_name")
        if not all(isinstance(value, str) and value for value in (robot, scene_family, subtask)):
            raise ValueError(f"Missing benchmark platform/task_name/sub_task_name in {path.name}")
        run_id = path.stem
        source_run_ids.add(run_id)
        source_semantic_pairs.add((robot, subtask))
        source_subtasks.add(subtask)
        eval_ref = f"benchmark/config/eval_tasks/{scene_family}.json"
        eval_refs.add(eval_ref)
        catalog_run = runs_by_id.get(run_id)
        source_hash = _sha256(path)
        if isinstance(catalog_run, dict):
            hash_value = catalog_run.get("source_config_sha256")
            hash_status = "match" if hash_value == source_hash else "missing" if not hash_value else "mismatch"
        else:
            hash_status = "not_cataloged"
        source_run_records.append({
            "id": run_id,
            "robot": robot,
            "scene_family": scene_family,
            "subtask": subtask,
            "source_config_sha256": source_hash,
            "catalog_hash_status": hash_status,
            "catalog_migration_status": catalog_run.get("migration_status") if catalog_run else None,
            "catalog_backend_checks": catalog_run.get("backend_checks", {}) if catalog_run else {},
        })

    catalog_semantics = _catalog_task_map(catalog)
    source_instances: dict[str, list[str]] = {}
    missing_scene_files: list[dict[str, Any]] = []
    source_asset_ids: set[str] = set()
    scene_asset_map: dict[str, list[str]] = {}
    empty_scene_info: list[str] = []
    file_presence = Counter()

    for subtask in sorted(source_subtasks):
        subtask_root = task_root / subtask
        instance_paths = (
            sorted(
                (path for path in subtask_root.iterdir() if path.is_dir()),
                key=lambda item: (not item.name.isdecimal(), int(item.name) if item.name.isdecimal() else item.name),
            )
            if subtask_root.is_dir()
            else []
        )
        ids = [path.name for path in instance_paths]
        source_instances[subtask] = ids
        for instance_path in instance_paths:
            scene_key = f"{subtask}/{instance_path.name}"
            missing = [name for name in SCENE_FILES if not (instance_path / name).is_file()]
            for name in SCENE_FILES:
                file_presence[name] += int((instance_path / name).is_file())
            if missing:
                missing_scene_files.append({"scene": scene_key, "missing_files": missing})
            info_path = instance_path / "scene_info.json"
            if info_path.is_file():
                try:
                    info = _json(info_path)
                except (OSError, json.JSONDecodeError):
                    info = None
                if not info:
                    empty_scene_info.append(scene_key)
            asset_ids = _source_scene_assets(instance_path / "scene.usda")
            scene_asset_map[scene_key] = asset_ids
            source_asset_ids.update(asset_ids)

    catalog_scene_keys: set[str] = set()
    catalog_assignments = 0
    for task in catalog.get("semantic_tasks", []):
        if not isinstance(task, dict):
            continue
        instances = task.get("source_instances", [])
        if isinstance(instances, list):
            catalog_assignments += len(instances)
            subtask = str(task.get("subtask", ""))
            catalog_scene_keys.update(f"{subtask}/{instance}" for instance in instances)
    source_scene_keys = {f"{task}/{instance}" for task, ids in source_instances.items() for instance in ids}

    adapter_aliases = _scene_adapter_aliases(scene_adapter_path)
    asset_status: dict[str, dict[str, Any]] = {}
    for asset_id in sorted(source_asset_ids):
        exact = _asset_manifest(assets_root, asset_id) if assets_root else {"available": False, "manifest_valid": False}
        if exact["manifest_valid"]:
            asset_status[asset_id] = {"resolution": "exact_prepared", "manifest": exact}
            continue
        alias = adapter_aliases.get(asset_id)
        alias_id = alias.get("asset_id") if isinstance(alias, dict) else None
        alias_manifest = (
            _asset_manifest(assets_root, str(alias_id))
            if assets_root is not None and isinstance(alias_id, str)
            else {"available": False, "manifest_valid": False}
        )
        if alias_manifest["manifest_valid"]:
            asset_status[asset_id] = {
                "resolution": "compatibility_alias_prepared_identity_unverified",
                "resolved_asset_id": alias_id,
                "alias_reason": alias.get("reason"),
                "manifest": alias_manifest,
            }
        else:
            asset_status[asset_id] = {
                "resolution": "missing_prepared_wrapper",
                "alias_candidate": alias_id,
                "manifest": exact,
            }

    scene_asset_coverage = Counter()
    scene_asset_gaps = []
    for scene_key, asset_ids in sorted(scene_asset_map.items()):
        unresolved = [
            asset_id
            for asset_id in asset_ids
            if asset_status.get(asset_id, {}).get("resolution") == "missing_prepared_wrapper"
        ]
        if unresolved:
            scene_asset_coverage["with_unresolved_assets"] += 1
            scene_asset_gaps.append({"scene": scene_key, "unresolved_asset_ids": unresolved})
        else:
            scene_asset_coverage["all_source_payloads_have_prepared_wrapper"] += 1

    status_counts = Counter(
        str(run.get("migration_status", "missing")) for run in catalog_runs if isinstance(run, dict)
    )
    backend_status_counts = {
        backend: dict(
            Counter(
                str(run.get("backend_checks", {}).get(backend, "missing"))
                for run in catalog_runs
                if isinstance(run, dict)
            )
        )
        for backend in BACKENDS
    }
    referenced_eval_missing = sorted(ref for ref in eval_refs if not (source_root / ref).is_file())
    catalog_semantic_keys = set(catalog_semantics)
    source_git_revision, source_git_verified = _source_git_revision(source_root)

    # A candidate means only that source files and current prepared asset wrappers
    # appear available for its first scene; it is not a scene-build or smoke result.
    candidate_runs = []
    for run in source_run_records:
        if run["robot"] != "g1_op":
            continue
        scene_key = f"{run['subtask']}/0"
        if scene_key not in scene_asset_map:
            continue
        gaps = [
            asset
            for asset in scene_asset_map[scene_key]
            if asset_status.get(asset, {}).get("resolution") == "missing_prepared_wrapper"
        ]
        candidate_runs.append({
            "run_id": run["id"],
            "scene": scene_key,
            "roboverse_task": runs_by_id.get(run["id"], {}).get("roboverse_task"),
            "source_assets_unresolved": gaps,
            "static_asset_inventory_candidate": not gaps
            and not any(item["scene"] == scene_key for item in missing_scene_files),
            "meaning": "source inputs and asset wrapper files only; scene build, backend smoke, and task success not checked",
        })
    candidate_runs.sort(key=lambda item: (not item["static_asset_inventory_candidate"], item["run_id"]))

    spec_bundle_records = []
    for path in sorted((Path(p).resolve() for p in (spec_bundles or [])), key=lambda item: item.name):
        spec_bundle_records.append({
            "name": f"{path.parent.name}/{path.name}",
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": _sha256(path) if path.is_file() else None,
            "individual_records_verified": False,
        })

    extension_records = []
    if extensions_path is not None and extensions_path.is_file():
        extensions = _json(extensions_path)
        if extensions.get("format") != "roboverse.icl.benchmark_extensions.v1":
            raise ValueError(f"Unsupported benchmark extensions format in {extensions_path}")
        repo_root = catalog_path.parents[3]  # .../roboverse_pack/tasks/icl -> repository root
        artifact_root = source_artifact_root.resolve() if source_artifact_root else repo_root.parent
        for extension in extensions.get("extensions", []):
            if not isinstance(extension, dict):
                continue
            prepared_path = repo_root / str(extension.get("prepared_scene_path", ""))
            source = extension.get("source", {})
            archive_name = source.get("archive") if isinstance(source, dict) else None
            archive_path = artifact_root / archive_name if isinstance(archive_name, str) else None
            expected_hash = source.get("archive_sha256") if isinstance(source, dict) else None
            actual_hash = _sha256(archive_path) if archive_path is not None and archive_path.is_file() else None
            extension_records.append({
                "id": extension.get("id"),
                "roboverse_task": extension.get("roboverse_task"),
                "declared_migration_status": extension.get("migration_status"),
                "declared_backend_checks": extension.get("backend_checks", {}),
                "prepared_scene_directory_exists": prepared_path.is_dir(),
                "source_archive_name": archive_name,
                "source_archive_exists": bool(archive_path and archive_path.is_file()),
                "source_archive_hash_status": (
                    "match"
                    if actual_hash and expected_hash == actual_hash
                    else "mismatch"
                    if actual_hash and expected_hash
                    else "not_available"
                ),
                "video_validation": extension.get("video_validation", {}),
                "behavioral_parity": extension.get("behavioral_parity", "not_declared"),
            })

    return {
        "format": "roboverse.icl.geniesim_inventory_audit.v1",
        "audit_scope": {
            "source_config_import": "catalog membership and source-config SHA-256 only",
            "scene_construction": "not tested; source scene input presence and asset wrapper inventory only",
            "backend_smoke": "not run by this audit; reported catalog labels are declarations, not new evidence",
            "semantic_success": "not tested",
            "behavioral_parity": "not tested",
        },
        "source": {
            "catalog_asserted_revision": catalog.get("source", {}).get("revision"),
            "observed_git_revision": source_git_revision,
            "observed_git_revision_verified_for_source_root": source_git_verified,
            "pinned_upstream_revision": resolved_upstream_revision,
            "upstream_package_comparison": source_reference_comparison,
            "note": (
                "A parent Git repository revision is not accepted as the upstream source revision. "
                "When supplied, pinned_upstream_revision is verified by Git and local package differences are listed."
            ),
        },
        "roboverse_inventory_inputs": {
            "catalog_sha256": _sha256(catalog_path),
            "extensions_sha256": (
                _sha256(extensions_path) if extensions_path is not None and extensions_path.is_file() else None
            ),
        },
        "source_config_coverage": {
            "source_count": len(source_configs),
            "catalog_run_count": len(catalog_runs),
            "missing_from_catalog": sorted(source_run_ids - set(runs_by_id)),
            "catalog_only_run_ids": sorted(set(runs_by_id) - source_run_ids),
            "config_hash_status_counts": dict(Counter(run["catalog_hash_status"] for run in source_run_records)),
            "runs": sorted(source_run_records, key=lambda item: item["id"]),
        },
        "evaluation_config_coverage": {
            "referenced_unique_count": len(eval_refs),
            "missing_references": referenced_eval_missing,
            "catalog_reference_mismatches": sorted(
                run["id"]
                for run in source_run_records
                if run["id"] in runs_by_id and runs_by_id[run["id"]].get("evaluation_config") not in eval_refs
            ),
        },
        "semantic_task_coverage": {
            "source_pair_count": len(source_semantic_pairs),
            "catalog_pair_count": len(catalog_semantic_keys),
            "source_pairs_missing_from_catalog": [
                list(item) for item in sorted(source_semantic_pairs - catalog_semantic_keys)
            ],
            "catalog_pairs_not_in_source": [
                list(item) for item in sorted(catalog_semantic_keys - source_semantic_pairs)
            ],
            "migration_status_declarations": dict(status_counts),
            "backend_status_declarations": backend_status_counts,
        },
        "scene_instance_coverage": {
            "source_unique_scene_count": len(source_scene_keys),
            "catalog_unique_scene_count": len(catalog_scene_keys),
            "source_task_assignment_count": sum(
                len(source_instances.get(subtask, [])) for _, subtask in source_semantic_pairs
            ),
            "catalog_task_assignment_count": catalog_assignments,
            "source_instances_missing_from_catalog": sorted(source_scene_keys - catalog_scene_keys),
            "catalog_instances_missing_from_source": sorted(catalog_scene_keys - source_scene_keys),
            "required_source_file_presence": dict(file_presence),
            "required_source_file_names": list(SCENE_FILES),
            "incomplete_source_instances": missing_scene_files,
            "empty_scene_info_instances_using_usd_payload_inventory": sorted(empty_scene_info),
        },
        "asset_coverage": {
            "source_payload_asset_count": len(source_asset_ids),
            "source_payload_asset_ids": sorted(source_asset_ids),
            "local_asset_root_available": bool(assets_root and assets_root.is_dir()),
            "resolution_status_counts": dict(Counter(item["resolution"] for item in asset_status.values())),
            "assets": asset_status,
            "scene_asset_coverage_counts": dict(scene_asset_coverage),
            "scenes_with_unresolved_asset_wrappers": scene_asset_gaps,
            "scope_note": "A valid local provenance manifest establishes wrapper-file presence only; it does not establish render, collision, physics, or task correctness.",
        },
        "cataloged_runs": {
            "records": len(catalog_runs),
            "catalog_semantic_records": len(catalog.get("semantic_tasks", [])),
            "source_run_records": len(source_run_records),
            "config_records_are_individually_imported": False,
            "note": "Catalog/config hash agreement is not the same as validating each record inside an imported benchmark JSON bundle.",
        },
        "imported_spec_bundle_files": spec_bundle_records,
        "benchmark_extensions": extension_records,
        "next_static_candidates": candidate_runs[:12],
        "recommended_next_run_ids": [
            "g1op_s2r_grasp_targets",
            "g1op_s2r_sort_fruit",
            "g1op_s2r_select_color",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="GenieSim benchmark package source root")
    parser.add_argument("--catalog", type=Path, default=Path("roboverse_pack/tasks/icl/benchmark_catalog.json"))
    parser.add_argument("--extensions", type=Path, default=Path("roboverse_pack/tasks/icl/benchmark_extensions.json"))
    parser.add_argument("--assets-root", type=Path, default=Path("roboverse_data/icl/objects/geniesim"))
    parser.add_argument("--scene-adapter", type=Path, default=Path("roboverse_pack/tasks/icl/geniesim_scene.py"))
    parser.add_argument("--source-artifact-root", type=Path, default=None)
    parser.add_argument("--spec-bundle", type=Path, action="append", default=[])
    parser.add_argument(
        "--upstream-repo", type=Path, help="Local Git metadata clone containing the pinned source commit"
    )
    parser.add_argument("--upstream-revision", help="Exact 40-character upstream Git commit SHA to compare against")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.upstream_repo is None) != (args.upstream_revision is None):
        parser.error("--upstream-repo and --upstream-revision must be supplied together")

    report = build_audit(
        source_root=args.source_root,
        catalog_path=args.catalog,
        extensions_path=args.extensions if args.extensions.is_file() else None,
        assets_root=args.assets_root,
        scene_adapter_path=args.scene_adapter,
        spec_bundles=args.spec_bundle,
        source_artifact_root=args.source_artifact_root,
        upstream_repo=args.upstream_repo,
        upstream_revision=args.upstream_revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        "GenieSim inventory audit: "
        f"{report['source_config_coverage']['source_count']} configs, "
        f"{report['scene_instance_coverage']['source_unique_scene_count']} scenes, "
        f"{report['asset_coverage']['source_payload_asset_count']} payload asset IDs -> {args.output}"
    )


if __name__ == "__main__":
    main()
