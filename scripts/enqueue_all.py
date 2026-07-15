"""Populate results/manifest.json with the full experiment matrix (methodology
sections 9, 10, 13, 14): Phase-1 Kvasir-SEG ablations (cheap go/no-go) and
Phase-2 headline/ISIC/comparator runs, all enqueued PENDING.

No new modeling -- this only assembles RunConfig objects that already exist
(lcmunet/experiment_matrix.py) and hands them to lcmunet.run_manifest.enqueue.

Usage:
    python scripts/enqueue_all.py                       # populate + print report
    python scripts/enqueue_all.py --dry-run              # print only, write nothing
    python scripts/enqueue_all.py --sec-per-epoch-json path/to/probe.json
        # also prints a GPU-hour table computed from REAL measured sec/epoch
        # per model_name (see the file's own docstring below for the format).
        # Without this flag, no GPU-hour number is printed -- GLOBAL RULES:
        # efficiency is measured, never quoted/estimated, and no such
        # measurement exists yet on this local CPU dev machine.
    python scripts/enqueue_all.py --hero-descriptor plain
        # Sets Phase-2's hero rows' descriptor_type (methodology section
        # 10.1: "winner used in all later results"). Defaults to 'contrast'
        # (the a-priori default) -- after Gate-2 runs, pass whatever
        # lcmunet.gate2_report's desc_winner turned out to be (see
        # notebooks/07_phase2.ipynb, which does this automatically). Only
        # affects Phase-2 hero config_ids; Phase-1's ablation matrix (which
        # includes the FIXED 'contrast' Desc-ablation reference row) never
        # changes.

IMPORTANT (surfaced again at the bottom of the printed report): on Colab
free tier, Phase-2 is many multi-hour runs. Run Phase-1 first; only start
Phase-2 jobs after Gate-2 (methodology section 13 Phase-1 decision rules)
passes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `python scripts/enqueue_all.py` works from any cwd

from lcmunet.config import RunConfig
from lcmunet.experiment_matrix import build_all, resolve_scan_impl
from lcmunet.paths import get_paths
from lcmunet.run_manifest import enqueue

# Ordered, longest/most-specific-first so a role matches exactly one bucket.
TABLE_PREFIXES: List[str] = [
    "phase1_ablation_A",
    "phase1_ablation_C",
    "phase1_ablation_D",
    "phase1_desc",
    "phase1_placement",
    "phase2_headline_kvasir_seg",
    "phase2_headline_cvc_clinicdb",
    "phase2_isic2017",
    "phase2_isic2018",
    "phase2_comparators_optional_if_time_cvc_clinicdb",
    "phase2_comparators_kvasir_seg",
]


def _bucket(role: str) -> str:
    for prefix in TABLE_PREFIXES:
        if role.startswith(prefix):
            return prefix
    return "unclassified"  # should never happen; fail visibly in the report rather than silently miscount


def _write_configs_and_enqueue(
    merged: Dict[str, Tuple[RunConfig, List[str]]], paths, dry_run: bool
) -> None:
    for config_id, (config, roles) in merged.items():
        yaml_path = Path(paths.configs) / f"{config_id}.yaml"
        if not dry_run:
            config.save_yaml(yaml_path)
            enqueue(paths.results, config_id, str(yaml_path))


def _write_matrix_sidecar(merged: Dict[str, Tuple[RunConfig, List[str]]], paths, dry_run: bool) -> Path:
    """results/experiment_matrix.json: config_id -> roles/tables it belongs
    to. NOT consumed by run_manifest (job identity is the RunConfig hash
    alone) -- pure documentation so a future analysis script can look up
    "which config_id is Ablation C's row" without re-deriving the matrix.
    """
    payload = {
        config_id: {
            "run_name": config.run_name,
            "model_name": config.model_name,
            "dataset": config.dataset,
            "seed": config.seed,
            "model_cfg": config.model_cfg,
            "roles": roles,
        }
        for config_id, (config, roles) in merged.items()
    }
    path = Path(paths.results) / "experiment_matrix.json"
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    return path


def _print_run_count_report(merged: Dict[str, Tuple[RunConfig, List[str]]]) -> None:
    bucket_counts: Dict[str, int] = {}
    for _config_id, (_config, roles) in merged.items():
        for role in roles:
            bucket_counts[_bucket(role)] = bucket_counts.get(_bucket(role), 0) + 1

    phase1_configs = {cid for cid, (_c, roles) in merged.items() if any(r.startswith("phase1") for r in roles)}
    phase2_configs = {cid for cid, (_c, roles) in merged.items() if any(r.startswith("phase2") for r in roles)}
    overlap = phase1_configs & phase2_configs

    print("=== Experiment matrix: table row counts (methodology section 10.1/10.2) ===")
    for prefix in TABLE_PREFIXES:
        print(f"  {prefix:55s} {bucket_counts.get(prefix, 0):3d} row(s)")
    if bucket_counts.get("unclassified"):
        print(f"  UNCLASSIFIED (bug -- fix TABLE_PREFIXES)        {bucket_counts['unclassified']:3d} row(s)")

    print()
    print("=== Unique physical training jobs (what actually gets enqueued) ===")
    print(f"  Phase-1 unique configs:  {len(phase1_configs)}")
    print(f"  Phase-2 unique configs:  {len(phase2_configs)}")
    print(f"  Shared (dedup'd) between Phase-1 and Phase-2: {len(overlap)}")
    print(f"  TOTAL unique config_ids: {len(merged)}")
    print("  (no duplicate config_ids by construction -- merged dict is keyed by config_id)")


def _print_gpu_hour_report(merged: Dict[str, Tuple[RunConfig, List[str]]], sec_per_epoch_json: str | None) -> None:
    print()
    print("=== GPU-hour estimate (epochs x measured sec/epoch) ===")
    if not sec_per_epoch_json:
        print("  Not computed: no --sec-per-epoch-json given.")
        print("  This is intentional, not an oversight -- GLOBAL RULES: efficiency is")
        print("  measured, never quoted/estimated, and no real Colab measurement exists")
        print("  yet on this machine. To get this table: in Colab, time one real epoch")
        print("  per model_name (ultralight_baseline/glgf/lc_ss2d/unet/malunet/egeunet)")
        print("  on the target GPU, save {\"model_name\": seconds, ...} to a JSON file,")
        print("  and re-run: python scripts/enqueue_all.py --sec-per-epoch-json <file>")
        return

    with open(sec_per_epoch_json, "r", encoding="utf-8") as f:
        sec_per_epoch = json.load(f)

    totals: Dict[str, float] = {}
    missing_models = set()
    for _config_id, (config, _roles) in merged.items():
        if config.model_name not in sec_per_epoch:
            missing_models.add(config.model_name)
            continue
        hours = config.epochs * sec_per_epoch[config.model_name] / 3600.0
        totals[config.model_name] = totals.get(config.model_name, 0.0) + hours

    grand_total = 0.0
    for model_name, hours in sorted(totals.items()):
        print(f"  {model_name:20s} {hours:8.1f} GPU-hours ({sec_per_epoch[model_name]:.1f} sec/epoch measured)")
        grand_total += hours
    print(f"  {'TOTAL':20s} {grand_total:8.1f} GPU-hours (CRUDE: excludes eval time, resume overhead, data loading warm-up)")
    if missing_models:
        print(f"  UNMEASURED model_name(s), excluded from the total above: {sorted(missing_models)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the report; do not write config files or touch manifest.json")
    parser.add_argument("--sec-per-epoch-json", default=None, help="path to a JSON {model_name: measured_seconds_per_epoch} file from a real Colab timing probe")
    parser.add_argument(
        "--hero-descriptor", default="contrast", choices=["contrast", "plain"],
        help="Phase-2 hero rows' descriptor_type -- the Desc ablation winner (default 'contrast', the a-priori default; pass Gate-2's actual desc_winner once known)",
    )
    args = parser.parse_args()

    paths = get_paths()
    scan_impl, scan_impl_source = resolve_scan_impl(paths)
    merged = build_all(scan_impl, hero_descriptor_type=args.hero_descriptor)

    print(f"scan_impl = {scan_impl!r} (source: {scan_impl_source})")
    print(f"hero_descriptor (Phase-2 only) = {args.hero_descriptor!r}")
    print()

    _write_configs_and_enqueue(merged, paths, dry_run=args.dry_run)
    sidecar_path = _write_matrix_sidecar(merged, paths, dry_run=args.dry_run)
    if args.dry_run:
        print("--dry-run: no files written, manifest.json untouched.\n")
    else:
        print(f"Configs written to {paths.configs}")
        print(f"Manifest updated at {Path(paths.results) / 'manifest.json'}")
        print(f"Role/table lookup written to {sidecar_path}\n")

    _print_run_count_report(merged)
    _print_gpu_hour_report(merged, args.sec_per_epoch_json)

    print()
    print("=== Before running any of this in Colab ===")
    print("  1. Run Phase-1 first (10 configs, Kvasir-SEG, 1 seed each) and evaluate")
    print("     against methodology section 13's Phase-1 decision rules (Gate 2) --")
    print("     see notebooks/06_phase1_gate2.ipynb. Only enqueue-and-run Phase-2 jobs")
    print("     after Gate 2 PROCEEDs (lcmunet.gate2_report.require_gate2_proceed is a")
    print("     hard runtime gate, not just a human checkbox) -- Phase-2 is many")
    print("     multi-hour runs and will exhaust Colab free-tier GPU-hours fast.")
    print("     Re-run this script with --hero-descriptor <Gate-2's desc_winner> once")
    print("     known (see notebooks/07_phase2.ipynb, which does this automatically).")
    print("  2. 'glgf' is implemented (lcmunet/glgf.py, GLGFUNet) -- its rows are real,")
    print("     trainable jobs, same as ultralight_baseline/lc_ss2d/comparators.")
    print("  3. Cross-dataset generalisation (train Kvasir/test CVC and reverse) is")
    print("     NOT enqueued here -- it is an evaluation step over the Phase-2 hero/")
    print("     baseline checkpoints, not a new training config. See the module")
    print("     docstring in lcmunet/experiment_matrix.py for why.")


if __name__ == "__main__":
    main()
