"""Single entry point for the entire download-and-prepare pipeline
(methodology section 7): download -> extract -> preprocess -> split, for
all four datasets, called from ONE cell in notebooks/colab_runner.ipynb.

Idempotent end to end: a dataset already fully present under data_raw/ (raw
pairs + split file) is a fast, no-network no-op on re-run -- nothing is
re-downloaded, re-extracted, or re-split unnecessarily (lcmunet.data.
download's ensure_* functions and lcmunet.data.splits' build_*_split
functions are each independently idempotent; this module just chains them).

Per-dataset failures (missing Kaggle auth, a dead link, a changed archive
layout, a CVC sequence-leakage assertion) are caught individually here --
one bad dataset never blocks the other three. Preprocessing itself
(resize/normalise/binarise, lcmunet/data/preprocess.py) is applied lazily,
per-sample, by the DataLoader -- there is no separate batch-preprocessing
step to run here.
"""

from __future__ import annotations

from typing import Any, Dict

from lcmunet.data import raw_layout as rl
from lcmunet.data import splits as splits_module
from lcmunet.data.download import ensure_dataset_ready
from lcmunet.data.kaggle_auth import KaggleAuthMissingError

_SPLIT_BUILDERS = {
    "kvasir_seg": lambda paths: splits_module.build_kvasir_split(paths),
    "cvc_clinicdb": lambda paths: splits_module.build_cvc_split(paths),
    "isic2017": lambda paths: splits_module.build_isic_split(paths, "isic2017"),
    "isic2018": lambda paths: splits_module.build_isic_split(paths, "isic2018"),
}


def _prepare_one_dataset(name: str, paths) -> Dict[str, Any]:
    n_pairs = ensure_dataset_ready(name, paths)
    split_payload = _SPLIT_BUILDERS[name](paths)
    return {"status": "PASS", "n_pairs": n_pairs, "counts": split_payload["counts"]}


def prepare_all_datasets(paths=None) -> Dict[str, Dict[str, Any]]:
    """Downloads (if missing), extracts, and splits all four datasets. Never
    raises for a single dataset's failure -- returns a per-dataset report
    dict ({"status": "PASS"/"FAIL", ...}) and prints a PASS/FAIL summary
    table. Safe to call every Colab session: a dataset that's already fully
    ready in Drive is a fast, no-network no-op.
    """
    if paths is None:
        from lcmunet.paths import get_paths

        paths = get_paths()

    report: Dict[str, Dict[str, Any]] = {}
    for name in rl.DATASET_NAMES:
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
        try:
            report[name] = _prepare_one_dataset(name, paths)
        except KaggleAuthMissingError as exc:
            print(f"[FAILED] {name}: {exc}")
            report[name] = {"status": "FAIL", "error": str(exc)}
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
    header = f"{'dataset':<14} {'status':<6} {'n_pairs':>8} {'train':>7} {'val':>6} {'test':>6}"
    print(header)
    print("-" * len(header))
    for name in rl.DATASET_NAMES:
        row = report[name]
        if row["status"] == "PASS":
            counts = row["counts"]
            print(f"{name:<14} {'PASS':<6} {row['n_pairs']:>8} {counts['train']:>7} {counts['val']:>6} {counts['test']:>6}")
        else:
            error_preview = str(row.get("error", ""))[:80]
            print(f"{name:<14} {'FAIL':<6} {'-':>8} {'-':>7} {'-':>6} {'-':>6}   ({error_preview})")
    n_pass = sum(1 for r in report.values() if r["status"] == "PASS")
    print("-" * len(header))
    print(f"{n_pass}/{len(rl.DATASET_NAMES)} datasets ready.")
    if n_pass < len(rl.DATASET_NAMES):
        print("See per-dataset errors above for exactly what to fix, then re-run this cell (already-ready datasets are skipped instantly).")


if __name__ == "__main__":
    prepare_all_datasets()
