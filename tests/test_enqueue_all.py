"""Integration tests for scripts/enqueue_all.py: run it as a real subprocess
(the way the user actually invokes it) against a temp DRIVE_ROOT, rather than
importing internals -- this is a top-level script, not a package module.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "enqueue_all.py"

EXPECTED_UNIQUE_CONFIGS = 67  # 70 table rows - 3 dedup'd (baseline/glgf/hero @ kvasir seed42)


def _run(args, drive_root):
    env = dict(os.environ)
    env["DRIVE_ROOT"] = str(drive_root)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_dry_run_writes_nothing(tmp_path):
    drive_root = tmp_path / "drive_root"
    result = _run(["--dry-run"], drive_root)

    assert result.returncode == 0, result.stderr
    assert f"TOTAL unique config_ids: {EXPECTED_UNIQUE_CONFIGS}" in result.stdout
    assert not (drive_root / "results" / "manifest.json").exists()
    assert not (drive_root / "configs").exists() or not any((drive_root / "configs").iterdir())


def test_populates_manifest_with_all_pending_jobs_no_duplicates(tmp_path):
    drive_root = tmp_path / "drive_root"
    result = _run([], drive_root)
    assert result.returncode == 0, result.stderr

    manifest_path = drive_root / "results" / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(manifest["jobs"]) == EXPECTED_UNIQUE_CONFIGS
    assert all(job["status"] == "PENDING" for job in manifest["jobs"].values())

    configs_dir = drive_root / "configs"
    yaml_files = list(configs_dir.glob("*.yaml"))
    assert len(yaml_files) == EXPECTED_UNIQUE_CONFIGS

    sidecar_path = drive_root / "results" / "experiment_matrix.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert len(sidecar) == EXPECTED_UNIQUE_CONFIGS
    assert all("roles" in entry and entry["roles"] for entry in sidecar.values())

    for config_id, job in manifest["jobs"].items():
        assert Path(job["config_yaml_path"]).name == f"{config_id}.yaml"


def test_rerun_is_idempotent_no_duplicate_jobs(tmp_path):
    drive_root = tmp_path / "drive_root"
    _run([], drive_root)
    result2 = _run([], drive_root)
    assert result2.returncode == 0, result2.stderr

    manifest = json.loads((drive_root / "results" / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["jobs"]) == EXPECTED_UNIQUE_CONFIGS


def test_gpu_hour_table_uses_only_measured_values(tmp_path):
    drive_root = tmp_path / "drive_root"
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps({"unet": 4.0, "malunet": 3.0}), encoding="utf-8")

    result = _run(["--dry-run", "--sec-per-epoch-json", str(probe)], drive_root)
    assert result.returncode == 0, result.stderr
    assert "GPU-hours" in result.stdout
    assert "UNMEASURED model_name(s)" in result.stdout  # egeunet/ultralight_baseline/glgf/lc_ss2d not in probe


def test_no_probe_prints_no_fabricated_gpu_hour_number(tmp_path):
    drive_root = tmp_path / "drive_root"
    result = _run(["--dry-run"], drive_root)
    assert result.returncode == 0, result.stderr
    assert "Not computed: no --sec-per-epoch-json given." in result.stdout


def test_hero_descriptor_default_matches_prior_behaviour(tmp_path):
    drive_root = tmp_path / "drive_root"
    result = _run(["--dry-run"], drive_root)
    assert result.returncode == 0, result.stderr
    assert "hero_descriptor (Phase-2 only) = 'contrast'" in result.stdout
    assert f"TOTAL unique config_ids: {EXPECTED_UNIQUE_CONFIGS}" in result.stdout


def test_hero_descriptor_plain_changes_only_phase2_hero_config_ids(tmp_path):
    drive_root_contrast = tmp_path / "drive_root_contrast"
    drive_root_plain = tmp_path / "drive_root_plain"

    _run([], drive_root_contrast)
    result_plain = _run(["--hero-descriptor", "plain"], drive_root_plain)
    assert result_plain.returncode == 0, result_plain.stderr
    assert "hero_descriptor (Phase-2 only) = 'plain'" in result_plain.stdout

    manifest_contrast = json.loads((drive_root_contrast / "results" / "manifest.json").read_text(encoding="utf-8"))
    manifest_plain = json.loads((drive_root_plain / "results" / "manifest.json").read_text(encoding="utf-8"))

    # same total job count either way (Phase-1 fixed row count + Phase-2 fixed row count, still deduped the same way)
    assert len(manifest_contrast["jobs"]) == len(manifest_plain["jobs"]) == EXPECTED_UNIQUE_CONFIGS

    sidecar_contrast = json.loads((drive_root_contrast / "results" / "experiment_matrix.json").read_text(encoding="utf-8"))
    sidecar_plain = json.loads((drive_root_plain / "results" / "experiment_matrix.json").read_text(encoding="utf-8"))

    phase1_ids_contrast = {cid for cid, e in sidecar_contrast.items() if any(r.startswith("phase1") for r in e["roles"])}
    phase1_ids_plain = {cid for cid, e in sidecar_plain.items() if any(r.startswith("phase1") for r in e["roles"])}
    assert phase1_ids_contrast == phase1_ids_plain  # Phase-1's ablation matrix never changes

    assert set(sidecar_contrast) != set(sidecar_plain)  # but Phase-2 hero rows DO get new config_ids
