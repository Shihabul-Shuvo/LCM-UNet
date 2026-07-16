import json
import time
from pathlib import Path

import pytest

from lcmunet import config as config_module
from lcmunet import run_manifest as rm


def test_enqueue_next_pending_mark_done(tmp_path):
    rm.enqueue(tmp_path, "cfg_a", "configs/cfg_a.yaml")
    rm.enqueue(tmp_path, "cfg_b", "configs/cfg_b.yaml")

    job = rm.next_pending(tmp_path)
    assert job["config_id"] == "cfg_a"
    assert job["status"] == rm.STATUS_RUNNING
    assert job["heartbeat"] is not None

    # cfg_a is now RUNNING, so next_pending should return cfg_b
    job2 = rm.next_pending(tmp_path)
    assert job2["config_id"] == "cfg_b"

    # nothing left pending
    assert rm.next_pending(tmp_path) is None

    rm.mark_done(tmp_path, "cfg_a")
    manifest = rm._read(tmp_path)
    assert manifest["jobs"]["cfg_a"]["status"] == rm.STATUS_DONE


def test_enqueue_is_idempotent(tmp_path):
    rm.enqueue(tmp_path, "cfg_a", "configs/cfg_a.yaml")
    rm.enqueue(tmp_path, "cfg_a", "configs/cfg_a_v2.yaml")  # no force -> ignored

    manifest = rm._read(tmp_path)
    assert len(manifest["jobs"]) == 1
    assert manifest["jobs"]["cfg_a"]["config_yaml_path"] == "configs/cfg_a.yaml"


def test_mark_failed_records_error(tmp_path):
    rm.enqueue(tmp_path, "cfg_a", "configs/cfg_a.yaml")
    rm.next_pending(tmp_path)
    rm.mark_failed(tmp_path, "cfg_a", error="CUDA OOM")

    manifest = rm._read(tmp_path)
    assert manifest["jobs"]["cfg_a"]["status"] == rm.STATUS_FAILED
    assert manifest["jobs"]["cfg_a"]["error"] == "CUDA OOM"


def test_reset_stale_reclaims_killed_running_job(tmp_path):
    rm.enqueue(tmp_path, "cfg_a", "configs/cfg_a.yaml")
    rm.next_pending(tmp_path)  # -> RUNNING

    # simulate a session that died long ago: backdate the heartbeat directly
    manifest = rm._read(tmp_path)
    manifest["jobs"]["cfg_a"]["heartbeat"] = time.time() - 3600  # 60 min ago
    rm._write(tmp_path, manifest)

    reclaimed = rm.reset_stale(tmp_path, timeout_min=30)
    assert reclaimed == ["cfg_a"]

    manifest = rm._read(tmp_path)
    assert manifest["jobs"]["cfg_a"]["status"] == rm.STATUS_PENDING

    # a fresh session can now pick it back up
    job = rm.next_pending(tmp_path)
    assert job["config_id"] == "cfg_a"


def test_reset_stale_leaves_fresh_running_job_alone(tmp_path):
    rm.enqueue(tmp_path, "cfg_a", "configs/cfg_a.yaml")
    rm.next_pending(tmp_path)  # fresh heartbeat

    reclaimed = rm.reset_stale(tmp_path, timeout_min=30)
    assert reclaimed == []


def test_run_queue_processes_all_pending_and_resumes_after_stale(tmp_path):
    rm.enqueue(tmp_path, "cfg_a", "configs/cfg_a.yaml")
    rm.enqueue(tmp_path, "cfg_b", "configs/cfg_b.yaml")

    seen = []

    def runner_fn(job):
        seen.append(job["config_id"])

    rm.run_queue(tmp_path, runner_fn, max_minutes=1)

    assert seen == ["cfg_a", "cfg_b"]
    manifest = rm._read(tmp_path)
    assert manifest["jobs"]["cfg_a"]["status"] == rm.STATUS_DONE
    assert manifest["jobs"]["cfg_b"]["status"] == rm.STATUS_DONE


def test_run_queue_marks_failed_on_exception_and_continues(tmp_path):
    rm.enqueue(tmp_path, "cfg_a", "configs/cfg_a.yaml")
    rm.enqueue(tmp_path, "cfg_b", "configs/cfg_b.yaml")

    def runner_fn(job):
        if job["config_id"] == "cfg_a":
            raise RuntimeError("boom")

    rm.run_queue(tmp_path, runner_fn, max_minutes=1)

    manifest = rm._read(tmp_path)
    assert manifest["jobs"]["cfg_a"]["status"] == rm.STATUS_FAILED
    assert "boom" in manifest["jobs"]["cfg_a"]["error"]
    assert manifest["jobs"]["cfg_b"]["status"] == rm.STATUS_DONE


def test_next_pending_respects_job_filter(tmp_path):
    rm.enqueue(tmp_path, "cfg_a", "configs/cfg_a.yaml")
    rm.enqueue(tmp_path, "cfg_b", "configs/cfg_b.yaml")

    only_b = lambda job: job["config_id"] == "cfg_b"
    job = rm.next_pending(tmp_path, job_filter=only_b)
    assert job["config_id"] == "cfg_b"

    manifest = rm._read(tmp_path)
    assert manifest["jobs"]["cfg_a"]["status"] == rm.STATUS_PENDING  # left untouched, not claimed
    assert manifest["jobs"]["cfg_b"]["status"] == rm.STATUS_RUNNING


def test_next_pending_job_filter_returns_none_when_nothing_matches(tmp_path):
    rm.enqueue(tmp_path, "cfg_a", "configs/cfg_a.yaml")
    assert rm.next_pending(tmp_path, job_filter=lambda job: False) is None

    manifest = rm._read(tmp_path)
    assert manifest["jobs"]["cfg_a"]["status"] == rm.STATUS_PENDING


def test_run_queue_with_job_filter_only_processes_matching_jobs(tmp_path):
    rm.enqueue(tmp_path, "cfg_a", "configs/cfg_a.yaml")
    rm.enqueue(tmp_path, "cfg_b", "configs/cfg_b.yaml")
    rm.enqueue(tmp_path, "cfg_c", "configs/cfg_c.yaml")

    seen = []
    phase_ids = {"cfg_a", "cfg_c"}
    rm.run_queue(tmp_path, lambda job: seen.append(job["config_id"]), max_minutes=1, job_filter=lambda job: job["config_id"] in phase_ids)

    assert sorted(seen) == ["cfg_a", "cfg_c"]
    manifest = rm._read(tmp_path)
    assert manifest["jobs"]["cfg_a"]["status"] == rm.STATUS_DONE
    assert manifest["jobs"]["cfg_c"]["status"] == rm.STATUS_DONE
    assert manifest["jobs"]["cfg_b"]["status"] == rm.STATUS_PENDING  # never touched by this filtered run


def test_run_queue_resumes_stale_job_from_fresh_session(tmp_path):
    """Simulates: session 1 claims a job and dies; session 2 calls run_queue and finishes it."""
    rm.enqueue(tmp_path, "cfg_a", "configs/cfg_a.yaml")
    rm.next_pending(tmp_path)  # session 1 claims it, then "dies"

    manifest = rm._read(tmp_path)
    manifest["jobs"]["cfg_a"]["heartbeat"] = time.time() - 3600
    rm._write(tmp_path, manifest)

    seen = []
    rm.run_queue(tmp_path, lambda job: seen.append(job["config_id"]), max_minutes=1)

    assert seen == ["cfg_a"]
    manifest = rm._read(tmp_path)
    assert manifest["jobs"]["cfg_a"]["status"] == rm.STATUS_DONE


# ---- sync_manifest_with_active_datasets (dataset-scope toggle) --------------
# The full multi-dataset scenario (item 6 of the prompt: switching scope,
# idempotency across repeated calls) lives in
# tests/test_dataset_scope_integration.py, alongside prepare_all_datasets.
# These are the narrower, single-concern tests specific to run_manifest.py.


def test_sync_manifest_dry_run_writes_nothing(paths, monkeypatch):
    monkeypatch.setattr(config_module, "ACTIVE_DATASETS", ["kvasir_seg", "cvc_clinicdb"])

    result = rm.sync_manifest_with_active_datasets(paths, dry_run=True)

    assert result["dry_run"] is True
    assert len(result["in_scope"]) > 0
    assert result["newly_enqueued"] == []
    assert not (paths.results / "manifest.json").exists()
    assert not any(Path(paths.configs).glob("*.yaml"))


def test_sync_manifest_only_datasets_in_scope_get_config_yaml_files(paths, monkeypatch):
    monkeypatch.setattr(config_module, "ACTIVE_DATASETS", ["kvasir_seg", "cvc_clinicdb"])

    rm.sync_manifest_with_active_datasets(paths)

    from lcmunet.config import RunConfig

    yaml_files = list(Path(paths.configs).glob("*.yaml"))
    assert len(yaml_files) > 0
    for path in yaml_files:
        loaded = RunConfig.load_yaml(path)
        assert loaded.dataset in ("kvasir_seg", "cvc_clinicdb")


def test_manifest_status_counts_by_dataset_breaks_down_correctly(paths, monkeypatch):
    monkeypatch.setattr(config_module, "ACTIVE_DATASETS", ["kvasir_seg", "cvc_clinicdb"])
    rm.sync_manifest_with_active_datasets(paths)

    job = rm.next_pending(paths.results)
    rm.mark_done(paths.results, job["config_id"])

    counts = rm.manifest_status_counts_by_dataset(paths)

    total_done = sum(bucket.get("DONE", 0) for bucket in counts.values())
    assert total_done == 1
    assert set(counts) <= {"kvasir_seg", "cvc_clinicdb"}  # nothing enqueued references isic2017/isic2018

    total_pending = sum(bucket.get("PENDING", 0) for bucket in counts.values())
    assert total_pending == sum(1 for j in rm._read(paths.results)["jobs"].values() if j["status"] == rm.STATUS_PENDING)
