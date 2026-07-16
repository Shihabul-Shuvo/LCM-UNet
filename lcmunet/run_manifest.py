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


def _resolve_hero_descriptor_type(paths) -> str:
    """Opportunistic: uses Gate-2's real desc_winner if Gate-2 has already
    PROCEED'd (methodology section 10.1: "winner used in all later
    results"), else falls back to 'contrast' (the a-priori default,
    lcmunet.config.DEFAULT_MODEL_CFG). Never hard-requires Gate-2 -- this is
    called unconditionally every colab_runner.ipynb session, often long
    before Phase-1 has even finished, and enqueuing Phase-2 jobs (PENDING,
    not run) is harmless; the real gate is enforced separately, at
    run_queue time, via job_filter / lcmunet.gate2_report.require_gate2_proceed.
    """
    try:
        from lcmunet.gate2_report import require_gate2_proceed

        gate2_rules = require_gate2_proceed(paths)
        return gate2_rules["desc_winner"]
    except Exception:  # noqa: BLE001 -- Gate-2 not run/not PROCEED yet is the common, expected case here
        return "contrast"


def sync_manifest_with_active_datasets(
    paths=None,
    hero_descriptor_type: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Idempotently enqueues every RunConfig in the full experiment matrix
    (lcmunet.experiment_matrix.build_all: Phase-1 Kvasir ablations, Phase-2
    headline/ISIC/comparator rows) whose `.dataset` is currently in
    lcmunet.config.ACTIVE_DATASETS. Configs for a dataset NOT in scope are
    never written or enqueued at all -- not as PENDING, not as a
    placeholder; they simply don't exist in the manifest until that dataset
    is added to ACTIVE_DATASETS and this is called again.

    Cross-dataset generalisation (Kvasir<->CVC) needs no separate handling
    here: it is an EVALUATION step over the already-trained Kvasir/CVC
    Phase-2 headline hero+baseline checkpoints (lcmunet/cross_dataset_eval.py),
    not a distinct RunConfig/manifest job -- enqueuing those headline rows
    (already covered by build_all() above) is exactly what it needs.

    IDEMPOTENT: lcmunet.run_manifest.enqueue() is already a no-op for a
    config_id already in the manifest (any status: PENDING/RUNNING/DONE/
    FAILED) -- calling this function 50 times in a row, or once every
    colab_runner.ipynb session, produces byte-identical results except for
    genuinely NEW jobs the first time a dataset newly enters ACTIVE_DATASETS.
    Existing jobs' status/heartbeat/error are never touched here.

    dry_run=True builds and classifies the matrix but writes nothing (no
    config .yaml files, manifest.json untouched) -- for reporting/preview.
    """
    if paths is None:
        from lcmunet.paths import get_paths

        paths = get_paths()

    from lcmunet import experiment_matrix as em
    from lcmunet.config import ACTIVE_DATASETS

    scan_impl, scan_impl_source = em.resolve_scan_impl(paths)
    if hero_descriptor_type is None:
        hero_descriptor_type = _resolve_hero_descriptor_type(paths)

    merged = em.build_all(scan_impl, hero_descriptor_type=hero_descriptor_type)
    before_ids = set(_read(paths.results)["jobs"].keys())

    in_scope: Dict[str, Any] = {}
    out_of_scope_datasets = set()
    for config_id, (config, roles) in merged.items():
        if config.dataset not in ACTIVE_DATASETS:
            out_of_scope_datasets.add(config.dataset)
            continue
        in_scope[config_id] = (config, roles)
        if not dry_run:
            yaml_path = Path(paths.configs) / f"{config_id}.yaml"
            config.save_yaml(yaml_path)
            enqueue(paths.results, config_id, str(yaml_path))

    newly_enqueued = sorted(set(in_scope) - before_ids) if not dry_run else []

    return {
        "scan_impl": scan_impl,
        "scan_impl_source": scan_impl_source,
        "hero_descriptor_type": hero_descriptor_type,
        "active_datasets": list(ACTIVE_DATASETS),
        "in_scope": in_scope,  # {config_id: (RunConfig, [roles, ...])}
        "out_of_scope_datasets": sorted(out_of_scope_datasets),
        "n_out_of_scope_configs": len(merged) - len(in_scope),
        "newly_enqueued": newly_enqueued,
        "dry_run": dry_run,
    }


def manifest_status_counts_by_dataset(paths) -> Dict[str, Dict[str, int]]:
    """{dataset: {STATUS: count}} over every job currently in the manifest,
    keyed by looking up each job's config_id against the full experiment
    matrix (Phase-1+Phase-2, all 4 datasets, so this reflects jobs enqueued
    under any past ACTIVE_DATASETS setting, not just the current one) --
    for a session-summary print, not used by run_queue itself.
    """
    from lcmunet import experiment_matrix as em

    scan_impl, _source = em.resolve_scan_impl(paths)
    merged = em.build_all(scan_impl)
    dataset_of_config_id = {cid: config.dataset for cid, (config, _roles) in merged.items()}

    manifest = _read(paths.results)
    counts: Dict[str, Dict[str, int]] = {}
    for config_id, job in manifest["jobs"].items():
        dataset = dataset_of_config_id.get(config_id, "unknown")
        bucket = counts.setdefault(dataset, {s: 0 for s in _STATUSES})
        bucket[job["status"]] = bucket.get(job["status"], 0) + 1
    return counts


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
