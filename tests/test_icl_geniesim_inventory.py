"""File-level GenieSim inventory audit tests; no simulator is imported."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from tools.icl.audit_geniesim_inventory import _compare_source_package, build_audit
from tools.icl.build_task_catalog import _resolve_source_revision
from tools.icl.import_geniesim_specs import _resolve_source_revision as _resolve_import_source_revision


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _fixture(tmp_path: Path, *, missing_asset: bool = False) -> tuple[Path, Path, Path]:
    source_root = tmp_path / "source"
    config_path = _write(
        source_root / "config" / "g1op_s2r_pick.yaml",
        yaml.safe_dump(
            {"benchmark": {"platform": "g1_op", "task_name": "table_task_g1_op", "sub_task_name": "pick"}},
            sort_keys=False,
        ),
    )
    scene_root = source_root / "benchmark" / "config" / "llm_task" / "pick" / "0"
    _write(scene_root / "scene_info.json", "{}\n")
    _write(scene_root / "problems.json", "{}\n")
    _write(scene_root / "instructions.json", "{}\n")
    _write(
        scene_root / "scene.usda",
        'def Xform "apple" (prepend payload = @/geniesim_assets/objects/benchmark/apple/fruit_01/Aligned.usda@) {}\n',
    )
    _write(source_root / "benchmark" / "config" / "eval_tasks" / "table_task_g1_op.json", "{}\n")

    assets_root = tmp_path / "assets"
    if not missing_asset:
        _write(
            assets_root / "fruit_01" / "provenance.json",
            json.dumps({
                "format": "roboverse.icl.geniesim_rigid_object.v3",
                "asset_id": "fruit_01",
                "source_revision": "a" * 40,
                "source_asset_directory": "objects/benchmark/apple/fruit_01",
            }),
        )

    catalog_path = tmp_path / "benchmark_catalog.json"
    catalog = {
        "format": "roboverse.icl.benchmark_catalog.v1",
        "source": {"revision": "unknown"},
        "runs": [
            {
                "id": "g1op_s2r_pick",
                "source_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "migration_status": "implemented_unverified",
                "evaluation_config": "benchmark/config/eval_tasks/table_task_g1_op.json",
                "backend_checks": {"mujoco": "not_run", "genesis": "not_run", "newton": "not_run"},
                "roboverse_task": "icl.geniesim_benchmark",
            }
        ],
        "semantic_tasks": [
            {
                "robot": "g1_op",
                "subtask": "pick",
                "source_instances": ["0"],
                "migration_status": "implemented_unverified",
                "backend_checks": {"mujoco": "not_run", "genesis": "not_run", "newton": "not_run"},
            }
        ],
    }
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    return source_root, catalog_path, assets_root


def test_inventory_separates_catalog_coverage_from_runtime_validation(tmp_path: Path):
    source_root, catalog_path, assets_root = _fixture(tmp_path)

    report = build_audit(
        source_root=source_root,
        catalog_path=catalog_path,
        assets_root=assets_root,
    )

    assert report["source_config_coverage"]["config_hash_status_counts"] == {"match": 1}
    assert report["semantic_task_coverage"]["source_pair_count"] == 1
    assert report["scene_instance_coverage"]["source_unique_scene_count"] == 1
    assert report["scene_instance_coverage"]["incomplete_source_instances"] == []
    assert report["asset_coverage"]["resolution_status_counts"] == {"exact_prepared": 1}
    assert report["audit_scope"]["scene_construction"].startswith("not tested")
    assert report["audit_scope"]["semantic_success"] == "not tested"
    assert report["semantic_task_coverage"]["backend_status_declarations"] == {
        "mujoco": {"not_run": 1},
        "genesis": {"not_run": 1},
        "newton": {"not_run": 1},
    }


def test_inventory_marks_missing_asset_wrapper_without_claiming_scene_success(tmp_path: Path):
    source_root, catalog_path, assets_root = _fixture(tmp_path, missing_asset=True)

    report = build_audit(
        source_root=source_root,
        catalog_path=catalog_path,
        assets_root=assets_root,
    )

    assert report["asset_coverage"]["resolution_status_counts"] == {"missing_prepared_wrapper": 1}
    assert report["asset_coverage"]["scene_asset_coverage_counts"] == {"with_unresolved_assets": 1}
    assert report["asset_coverage"]["scenes_with_unresolved_asset_wrappers"] == [
        {"scene": "pick/0", "unresolved_asset_ids": ["fruit_01"]}
    ]
    assert report["cataloged_runs"]["config_records_are_individually_imported"] is False


def test_catalog_can_pin_an_explicit_audited_upstream_revision(tmp_path: Path):
    revision = "cd46c52d89e2ad40a62e1af7aa05b7ab1c67f7b7"
    assert _resolve_source_revision(tmp_path / "source", revision) == revision
    assert _resolve_import_source_revision(tmp_path / "source", revision) == revision
    with pytest.raises(ValueError, match="full lowercase 40-character Git commit SHA"):
        _resolve_source_revision(tmp_path / "source", "main")
    with pytest.raises(ValueError, match="full lowercase 40-character Git commit SHA"):
        _resolve_import_source_revision(tmp_path / "source", "main")


def test_upstream_tree_comparison_reports_task_scope_and_package_deltas(tmp_path: Path):
    package_root = tmp_path / "local_package"
    source_root = package_root / "src" / "geniesim_benchmark"
    files = {
        "src/geniesim_benchmark/config/g1op_s2r_pick.yaml": "run: pick\n",
        "src/geniesim_benchmark/benchmark/config/eval_tasks/table.json": "{}\n",
        "src/geniesim_benchmark/benchmark/config/llm_task/pick/0/scene.usda": "#usda 1.0\n",
        "src/geniesim_benchmark/benchmark/task_runtime.py": "local runtime\n",
        "skills/run-benchmark/SKILL.md": "run task\n",
        "pyproject.toml": "local-only dependency\n",
    }
    upstream_contents = {
        **{relative: content for relative, content in files.items() if relative != "pyproject.toml"},
        "src/geniesim_benchmark/benchmark/task_runtime.py": "upstream runtime\n",
        "src/geniesim_benchmark/benchmark/config/eval_tasks/missing.json": "{}\n",
    }
    for relative, content in files.items():
        path = package_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")

    def git_blob_id(content: str) -> str:
        payload = content.encode("utf-8")
        return hashlib.sha1(b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload).hexdigest()

    comparison = _compare_source_package(
        source_root,
        {relative: git_blob_id(content) for relative, content in upstream_contents.items()},
    )
    categories = comparison["categories"]
    assert categories["run_configs"]["exact_match"] is True, categories["run_configs"]
    assert categories["scene_inputs"]["exact_match"] is True
    assert categories["benchmark_skills"]["exact_match"] is True
    assert categories["evaluation_configs"]["upstream_only_files"] == [
        "src/geniesim_benchmark/benchmark/config/eval_tasks/missing.json"
    ]
    assert categories["benchmark_runtime_python"]["content_mismatch_files"] == [
        "src/geniesim_benchmark/benchmark/task_runtime.py"
    ]
    assert categories["whole_package"]["local_only_files"] == ["pyproject.toml"]
