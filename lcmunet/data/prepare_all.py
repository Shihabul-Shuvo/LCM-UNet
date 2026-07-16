"""Single entry point for the entire prepare pipeline (methodology section
7): extract -> preprocess -> split, for all four datasets, called from ONE
cell in notebooks/colab_runner.ipynb. Nothing is downloaded here -- the
user places each dataset's .zip under DRIVE_ROOT/data_raw/<Name>/ by hand
(see lcmunet/data/download.py's module docstring for exactly where to get
each one and what filename convention to expect).

Idempotent end to end, with two layers of caching:
  1. DRIVE_ROOT/data/<name>/prepared.json -- a marker written once a
     dataset's raw pairs + split are both confirmed ready. If present, the
     dataset is skipped INSTANTLY (no filesystem scan at all) on every
     later session -- this is the fast path re-running colab_runner.ipynb
     every session relies on. Delete this file to force a full re-check.
  2. Even without the marker, lcmunet.data.download's ensure_* functions
     and lcmunet.data.splits' build_*_split functions are each
     independently idempotent (they check the actual extracted/split state
     before doing any work), so a missing-marker-but-actually-ready dataset
     still resolves fast, just not "instant".

Per-dataset failures (no .zip placed yet, a changed archive layout, a CVC
sequence-leakage assertion) are caught individually here -- one bad/missing
dataset never blocks the other three.

DATASET SCOPE (lcmunet.config.ACTIVE_DATASETS): only datasets in that list
are attempted at all. A dataset NOT in scope is reported SKIPPED, not
FAILED -- its data_raw/ folder is never required to exist and no error is
raised for it. Edit ACTIVE_DATASETS in lcmunet/config.py to bring a
dataset back into scope; the next call here will prepare it automatically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from lcmunet import config as config_module
from lcmunet.data import raw_layout as rl
from lcmunet.data import splits as splits_module
from lcmunet.data.download import ensure_dataset_ready

_SPLIT_BUILDERS = {
    "kvasir_seg": lambda paths: splits_module.build_kvasir_split(paths),
    "cvc_clinicdb": lambda paths: splits_module.build_cvc_split(paths),
    "isic2017": lambda paths: splits_module.build_isic_split(paths, "isic2017"),
    "isic2018": lambda paths: splits_module.build_isic_split(paths, "isic2018"),
}


def _marker_path(paths, name: str) -> Path:
    return Path(paths.data) / name / "prepared.json"


def _load_marker(paths, name: str) -> Optional[Dict[str, Any]]:
    path = _marker_path(paths, name)
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None  # corrupt/partial marker -- treat as "not prepared", re-check for real


def _write_marker(paths, name: str, n_pairs: int, counts: Dict[str, int]) -> Path:
    path = _marker_path(paths, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"n_pairs": n_pairs, "counts": counts}, f, indent=2, sort_keys=True)
    return path


def _prepare_one_dataset(name: str, paths) -> Dict[str, Any]:
    cached = _load_marker(paths, name)
    if cached is not None:
        print(f"{name}: already prepared (cached at {_marker_path(paths, name)}). Skipping instantly.")
        return {"status": "PASS", "n_pairs": cached["n_pairs"], "counts": cached["counts"], "cached": True}

    n_pairs = ensure_dataset_ready(name, paths.data_raw)
    split_payload = _SPLIT_BUILDERS[name](paths)
    _write_marker(paths, name, n_pairs, split_payload["counts"])
    return {"status": "PASS", "n_pairs": n_pairs, "counts": split_payload["counts"], "cached": False}


def prepare_all_datasets(paths=None) -> Dict[str, Dict[str, Any]]:
    """Extracts (from a manually-placed .zip) and splits all four datasets.
    Never raises for a single dataset's failure -- returns a per-dataset
    report dict ({"status": "PASS"/"FAIL", ...}) and prints a PASS/FAIL
    summary table. Safe to call every Colab session: a dataset already
    marked prepared in Drive is skipped instantly.
    """
    if paths is None:
        from lcmunet.paths import get_paths

        paths = get_paths()

    active_datasets = config_module.ACTIVE_DATASETS

    report: Dict[str, Dict[str, Any]] = {}
    for name in rl.DATASET_NAMES:
        if name not in active_datasets:
            print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
            print(f"{name}: SKIPPED (not in ACTIVE_DATASETS={list(active_datasets)!r}, see lcmunet/config.py). No data_raw/ folder required while inactive.")
            report[name] = {"status": "SKIPPED", "error": "not in ACTIVE_DATASETS"}
            continue

        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
        try:
            report[name] = _prepare_one_dataset(name, paths)
        except rl.RawDataMissingError as exc:
            print(exc.instructions)
            print(f"[FAILED] {name}: raw data not available.")
            report[name] = {"status": "FAIL", "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 -- one bad dataset must never block the others
            print(f"[FAILED] {name}: {exc!r}")
            report[name] = {"status": "FAIL", "error": repr(exc)}

    _print_summary_table(report)
    return report


def _print_summary_table(report: Dict[str, Dict[str, Any]]) -> None:
    print(f"\n{'=' * 78}\nDATA PREPARATION SUMMARY\n{'=' * 78}")
    header = f"{'dataset':<14} {'status':<10} {'n_pairs':>8} {'train':>7} {'val':>6} {'test':>6}"
    print(header)
    print("-" * len(header))
    for name in rl.DATASET_NAMES:
        row = report[name]
        if row["status"] == "PASS":
            counts = row["counts"]
            cached_note = " (cached)" if row.get("cached") else ""
            print(f"{name:<14} {'PASS':<10} {row['n_pairs']:>8} {counts['train']:>7} {counts['val']:>6} {counts['test']:>6}{cached_note}")
        elif row["status"] == "SKIPPED":
            print(f"{name:<14} {'SKIPPED':<10} {'-':>8} {'-':>7} {'-':>6} {'-':>6}   (not in ACTIVE_DATASETS)")
        else:
            error_preview = str(row.get("error", ""))[:80]
            print(f"{name:<14} {'FAIL':<10} {'-':>8} {'-':>7} {'-':>6} {'-':>6}   ({error_preview})")
    n_pass = sum(1 for r in report.values() if r["status"] == "PASS")
    n_skipped = sum(1 for r in report.values() if r["status"] == "SKIPPED")
    n_attempted = len(report) - n_skipped
    print("-" * len(header))
    print(f"{n_pass}/{n_attempted} in-scope datasets ready" + (f"; {n_skipped} SKIPPED (not in ACTIVE_DATASETS)." if n_skipped else "."))
    if n_pass < n_attempted:
        print("See per-dataset errors above for exactly what to fix (place the missing .zip), then re-run this cell.")


if __name__ == "__main__":
    prepare_all_datasets()
