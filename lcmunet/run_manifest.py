"""JSON job queue (results/manifest.json) that survives a killed Colab session.

A fresh Colab session can call run_queue() and it will pick up exactly where
a previous, disconnected session left off: reset_stale() reclaims jobs that
were RUNNING when the old session died (no heartbeat update in too long),
then PENDING jobs are processed one at a time.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

STATUS_PENDING = "PENDING"
STATUS_RUNNING = "RUNNING"
STATUS_DONE = "DONE"
STATUS_FAILED = "FAILED"

_STATUSES = (STATUS_PENDING, STATUS_RUNNING, STATUS_DONE, STATUS_FAILED)


def _manifest_path(results_dir: str | Path) -> Path:
    return Path(results_dir) / "manifest.json"


def _read(results_dir: str | Path) -> Dict[str, Any]:
    path = _manifest_path(results_dir)
    if not path.exists():
        return {"jobs": {}}
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest.setdefault("jobs", {})
    return manifest


def _write(results_dir: str | Path, manifest: Dict[str, Any]) -> None:
    path = _manifest_path(results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    tmp_path.replace(path)


def enqueue(
    results_dir: str | Path,
    config_id: str,
    config_yaml_path: str | Path,
    force: bool = False,
) -> Dict[str, Any]:
    """Add a job as PENDING. No-op if config_id is already queued, unless force=True."""
    manifest = _read(results_dir)
    jobs = manifest["jobs"]
    if config_id in jobs and not force:
        return jobs[config_id]

    now = time.time()
    job = {
        "config_id": config_id,
        "config_yaml_path": str(config_yaml_path),
        "status": STATUS_PENDING,
        "heartbeat": None,
        "created_at": now,
        "updated_at": now,
        "error": None,
    }
    jobs[config_id] = job
    _write(results_dir, manifest)
    return job


def next_pending(
    results_dir: str | Path, job_filter: Optional[Callable[[Dict[str, Any]], bool]] = None
) -> Optional[Dict[str, Any]]:
    """Atomically claim the oldest PENDING job: marks it RUNNING with a fresh heartbeat.

    job_filter, if given, restricts candidates to jobs where job_filter(job)
    is True (e.g. "only Phase-1 config_ids") -- PENDING jobs that don't match
    are left untouched, not claimed, not skipped-and-lost.
    """
    manifest = _read(results_dir)
    jobs = manifest["jobs"]

    candidates = [j for j in jobs.values() if j["status"] == STATUS_PENDING]
    if job_filter is not None:
        candidates = [j for j in candidates if job_filter(j)]
    candidates.sort(key=lambda j: j["created_at"])
    if not candidates:
        return None

    job = candidates[0]
    now = time.time()
    job["status"] = STATUS_RUNNING
    job["heartbeat"] = now
    job["updated_at"] = now
    _write(results_dir, manifest)
    return job


def heartbeat(results_dir: str | Path, config_id: str) -> None:
    manifest = _read(results_dir)
    job = manifest["jobs"].get(config_id)
    if job is None:
        raise KeyError(f"unknown job: {config_id}")
    now = time.time()
    job["heartbeat"] = now
    job["updated_at"] = now
    _write(results_dir, manifest)


def mark_done(results_dir: str | Path, config_id: str) -> None:
    manifest = _read(results_dir)
    job = manifest["jobs"].get(config_id)
    if job is None:
        raise KeyError(f"unknown job: {config_id}")
    job["status"] = STATUS_DONE
    job["updated_at"] = time.time()
    job["error"] = None
    _write(results_dir, manifest)


def mark_failed(results_dir: str | Path, config_id: str, error: str = "") -> None:
    manifest = _read(results_dir)
    job = manifest["jobs"].get(config_id)
    if job is None:
        raise KeyError(f"unknown job: {config_id}")
    job["status"] = STATUS_FAILED
    job["updated_at"] = time.time()
    job["error"] = error
    _write(results_dir, manifest)


def reset_stale(results_dir: str | Path, timeout_min: float = 30.0) -> List[str]:
    """Reclaim RUNNING jobs whose heartbeat is older than timeout_min (a session that died)."""
    manifest = _read(results_dir)
    jobs = manifest["jobs"]
    now = time.time()
    reclaimed: List[str] = []

    for config_id, job in jobs.items():
        if job["status"] != STATUS_RUNNING:
            continue
        last_beat = job.get("heartbeat") or job.get("updated_at") or 0
        if now - last_beat > timeout_min * 60.0:
            job["status"] = STATUS_PENDING
            job["heartbeat"] = None
            job["updated_at"] = now
            reclaimed.append(config_id)

    if reclaimed:
        _write(results_dir, manifest)
    return reclaimed


def run_queue(
    results_dir: str | Path,
    runner_fn: Callable[[Dict[str, Any]], None],
    max_minutes: float = 60.0,
    stale_timeout_min: float = 30.0,
    job_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> None:
    """Drive the job queue until it is empty or the time budget runs out.

    Call this at the top of a fresh Colab session. It first reclaims jobs
    left RUNNING by a killed session, then repeatedly claims the next PENDING
    job and calls runner_fn(job). runner_fn is responsible for raising on any
    real failure; run_queue records that as FAILED and moves on to the next
    job rather than crashing the whole queue (orchestration resilience, not a
    correctness catch-and-continue).

    job_filter, if given, restricts which PENDING jobs this call will claim
    (e.g. "only Phase-1 config_ids", see lcmunet/gate2_report.py) -- jobs
    that don't match are left PENDING for a later, differently-filtered (or
    unfiltered) call, never touched by this one.
    """
    start = time.time()
    reset_stale(results_dir, timeout_min=stale_timeout_min)

    while (time.time() - start) / 60.0 < max_minutes:
        job = next_pending(results_dir, job_filter=job_filter)
        if job is None:
            break
        config_id = job["config_id"]
        try:
            runner_fn(job)
        except Exception as exc:  # noqa: BLE001 - orchestration boundary, see docstring
            mark_failed(results_dir, config_id, error=repr(exc))
        else:
            mark_done(results_dir, config_id)
