"""Final pre-submission checklist (methodology section 16).

Verifies programmatically what it CAN and prints PASS/UNKNOWN/FAIL for each
item, plus collects every "[... verify ...]" marker in the methodology
document into one TODO list.

Status convention:
  PASS/FAIL  -- fully determinable from code alone, right now, on any
                machine (Step-0 audit, config defaults, descriptor/W_delta
                structure). A FAIL here means something is actually wrong.
  UNKNOWN    -- depends on an artifact that only exists after a real Colab
                run with real data (CVC sequence map, efficiency.csv,
                Phase-1/Phase-2 results, mechanism figures, stats report).
                Missing artifact -> UNKNOWN ("not run yet"), not FAIL
                ("run and found broken") -- once the artifact exists, the
                check resolves to a real PASS or FAIL.

Usage:
    python scripts/final_checklist.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `python scripts/final_checklist.py` works from any cwd

METHODOLOGY_PATH = Path(__file__).resolve().parent.parent / "docs" / "LCM-UNet_FINAL_methodology_v4.1.md"

VERIFY_MARKER_RE = re.compile(r"\[[^\[\]]*verify[^\[\]]*\]", re.IGNORECASE)


def _result(name: str, status: str, detail: str) -> Dict[str, Any]:
    return {"name": name, "status": status, "detail": detail}


# ---- code-level checks (always PASS/FAIL, no Colab needed) -----------------


def check_step0_audit() -> Dict[str, Any]:
    from lcmunet.audit import run_step0_audit

    try:
        report = run_step0_audit(verbose=False)
    except Exception as exc:  # noqa: BLE001 -- turn any audit failure into a checklist FAIL, not a crash
        return _result("step0_audit", "FAIL", f"Step-0 audit raised: {exc!r}")
    if report.get("all_6_items_passed"):
        return _result("step0_audit", "PASS", "All 6 Step-0 audit items confirmed (lcmunet.audit.run_step0_audit).")
    return _result("step0_audit", "FAIL", f"Step-0 audit did not report all_6_items_passed: {report}")


def check_alpha_and_wdelta_defaults() -> Dict[str, Any]:
    from lcmunet.config import DEFAULT_MODEL_CFG

    alpha_ok = DEFAULT_MODEL_CFG.get("alpha_init") == 0.01
    wdelta_ok = abs(DEFAULT_MODEL_CFG.get("wdelta_std", -1.0) - 1e-3) < 1e-12
    if alpha_ok and wdelta_ok:
        return _result("alpha_and_wdelta_defaults", "PASS", "DEFAULT_MODEL_CFG: alpha_init=0.01, wdelta_std=1e-3.")
    return _result(
        "alpha_and_wdelta_defaults", "FAIL",
        f"DEFAULT_MODEL_CFG alpha_init={DEFAULT_MODEL_CFG.get('alpha_init')}, wdelta_std={DEFAULT_MODEL_CFG.get('wdelta_std')}",
    )


def check_per_group_wdelta_shared_across_directions() -> Dict[str, Any]:
    from lcmunet.lc_vss import PVM_NUM_GROUPS, LC_PVMLayer

    layer = LC_PVMLayer(input_dim=32, output_dim=32, descriptor_type="contrast")
    if layer.w_deltas is None or len(layer.w_deltas) != PVM_NUM_GROUPS:
        got = None if layer.w_deltas is None else len(layer.w_deltas)
        return _result("per_group_wdelta", "FAIL", f"expected {PVM_NUM_GROUPS} W_delta modules (one per PVM group), got {got}")
    distinct = len({id(m) for m in layer.w_deltas})
    if distinct != PVM_NUM_GROUPS:
        return _result("per_group_wdelta", "FAIL", f"W_delta modules are not all distinct per group ({distinct}/{PVM_NUM_GROUPS} unique)")
    return _result(
        "per_group_wdelta", "PASS",
        f"{PVM_NUM_GROUPS} distinct per-group W_delta modules confirmed. 'Shared across the four scan "
        "directions' is trivially satisfied -- this backbone has ONE flatten order, not four (see "
        "lcmunet/lc_vss.py's ARCHITECTURE NOTE: 'direction' maps onto PVM channel-group here).",
    )


def check_both_descriptors_available() -> Dict[str, Any]:
    import torch

    from lcmunet.lc_vss import VALID_DESCRIPTOR_TYPES, LocalDescriptor

    if not {"contrast", "plain"} <= set(VALID_DESCRIPTOR_TYPES):
        return _result("both_descriptors_available", "FAIL", f"VALID_DESCRIPTOR_TYPES={VALID_DESCRIPTOR_TYPES} missing 'contrast' and/or 'plain'")

    torch.manual_seed(0)
    contrast = LocalDescriptor(channels=8, descriptor_type="contrast")
    plain = LocalDescriptor(channels=8, descriptor_type="plain")
    with torch.no_grad():
        plain.dw_conv.weight.copy_(contrast.dw_conv.weight)  # identical conv weights -> isolate the "-Xn" difference

    x = torch.randn(2, 8, 6, 6)
    with torch.no_grad():
        contrast_out = contrast(x)
        plain_out = plain(x)
    if torch.allclose(contrast_out, plain_out - x, atol=1e-5):
        return _result(
            "both_descriptors_available", "PASS",
            "Both descriptor_types constructible; contrast == DWConv3x3(Xn) - Xn confirmed functionally "
            "(identical conv weights, contrast output equals plain output minus Xn exactly).",
        )
    return _result("both_descriptors_available", "FAIL", "contrast descriptor output != plain - Xn for identical conv weights.")


def check_active_datasets_full_scope() -> Dict[str, Any]:
    from lcmunet.config import ACTIVE_DATASETS
    from lcmunet.data.raw_layout import DATASET_NAMES

    missing = [d for d in DATASET_NAMES if d not in ACTIVE_DATASETS]
    if not missing:
        return _result("active_datasets_full_scope", "PASS", f"ACTIVE_DATASETS covers all {len(DATASET_NAMES)} datasets: {list(ACTIVE_DATASETS)}.")
    return _result(
        "active_datasets_full_scope", "FAIL",
        f"ACTIVE_DATASETS={list(ACTIVE_DATASETS)} excludes {missing} -- a full run over all 4 "
        "datasets is REQUIRED before final submission per the methodology (section 7). Edit "
        "lcmunet/config.py's ACTIVE_DATASETS, commit, push, and re-run colab_runner.ipynb to "
        "bring the excluded dataset(s) into scope (no other manual step needed).",
    )


# ---- artifact-dependent checks (UNKNOWN until the real Colab run exists) --


def check_cvc_sequence_split_zero_overlap(paths) -> Dict[str, Any]:
    import json

    from lcmunet.data.cvc_sequence import assert_no_sequence_leakage

    path = Path(paths.splits) / "cvc_sequence_map.json"
    if not path.is_file():
        return _result("cvc_sequence_split_zero_overlap", "UNKNOWN", f"No {path} yet -- build the CVC split (real CVC-ClinicDB data required) first.")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    try:
        assert_no_sequence_leakage(data["sequence_of_frame"], data["partition_of_frame"])
    except AssertionError as exc:
        return _result("cvc_sequence_split_zero_overlap", "FAIL", str(exc))
    return _result(
        "cvc_sequence_split_zero_overlap", "PASS",
        f"No sequence spans more than one partition ({data.get('n_sequences')} sequences; "
        f"mapping source: {data.get('mapping_source')}).",
    )


def check_same_scan_impl_for_all(paths) -> Dict[str, Any]:
    import pandas as pd

    path = Path(paths.results) / "efficiency.csv"
    if not path.is_file():
        return _result("same_scan_impl_for_all", "UNKNOWN", f"No {path} yet -- run notebooks/08_efficiency.ipynb first.")
    df = pd.read_csv(path)
    if df.empty:
        return _result("same_scan_impl_for_all", "UNKNOWN", f"{path} has no rows yet.")
    impls = sorted(set(df["scan_impl"].dropna().unique().tolist()))
    if len(impls) == 1:
        return _result("same_scan_impl_for_all", "PASS", f"All {len(df)} efficiency.csv rows share scan_impl={impls[0]!r}.")
    return _result(
        "same_scan_impl_for_all", "FAIL",
        f"Multiple scan_impl values found in efficiency.csv: {impls} -- mixed-implementation "
        "efficiency comparisons are invalid (section 5.5).",
    )


def check_ablations_present(paths) -> Dict[str, Any]:
    from lcmunet import experiment_matrix as em
    from lcmunet.results_store import load_results

    scan_impl, _source = em.resolve_scan_impl(paths)
    phase1_rows = dict(em.build_phase1_kvasir(scan_impl))
    df = load_results(paths.results)
    done_ids = set(df["config_id"]) if len(df) else set()

    missing = [role for role, config in phase1_rows.items() if config.config_id not in done_ids]
    if not missing:
        return _result("ablations_present", "PASS", f"All {len(phase1_rows)} Phase-1 rows (A/C/D/Desc + Placement) present in results.csv.")
    return _result("ablations_present", "UNKNOWN", f"{len(missing)}/{len(phase1_rows)} Phase-1 rows not yet DONE: {sorted(missing)}")


def check_mechanism_figures_present(paths) -> Dict[str, Any]:
    required = [
        Path(paths.figures) / "delta_difference_map.png",
        Path(paths.figures) / "region_wise_modulation.png",
        Path(paths.figures) / "per_stage_alpha.png",
        Path(paths.results) / "mechanism_report.md",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if not missing:
        return _result("mechanism_figures_present", "PASS", "All 3 mechanism figures + mechanism_report.md present.")
    return _result("mechanism_figures_present", "UNKNOWN", f"Missing: {missing} -- run notebooks/09_mechanism.ipynb first.")


def check_seeds_present(paths) -> Dict[str, Any]:
    import pandas as pd

    from lcmunet.config import ACTIVE_DATASETS

    path = Path(paths.results) / "phase2_summary.csv"
    if not path.is_file():
        return _result("seeds_present_5_3", "UNKNOWN", f"No {path} yet -- run notebooks/07_phase2.ipynb first.")
    df = pd.read_csv(path)
    if df.empty:
        return _result("seeds_present_5_3", "UNKNOWN", f"{path} has no rows yet.")

    isic_active = any(d in ACTIVE_DATASETS for d in ("isic2017", "isic2018"))
    expected = {"headline": 5, "headline_best_competitor": 3}
    if isic_active:
        expected["isic_generalisation"] = 3

    problems = []
    for scope, expected_n in expected.items():
        rows = df[df["scope"] == scope]
        if rows.empty:
            problems.append(f"{scope}: no rows")
            continue
        bad = rows[rows["n_seeds"] != expected_n]
        if len(bad):
            problems.append(f"{scope}: {len(bad)} row(s) with n_seeds != {expected_n} (found {sorted(bad['n_seeds'].unique().tolist())})")

    scope_note = "" if isic_active else " isic_generalisation: N/A -- not in current scope (ISIC not in ACTIVE_DATASETS)."
    if not problems:
        return _result("seeds_present_5_3", "PASS", "headline rows have 5 seeds; comparator" + ("/ISIC" if isic_active else "") + " rows have 3 seeds." + scope_note)
    return _result("seeds_present_5_3", "FAIL", "; ".join(problems) + scope_note)


def check_stats_computed(paths) -> Dict[str, Any]:
    path = Path(paths.results) / "stats_report.md"
    if not path.is_file():
        return _result("stats_computed", "UNKNOWN", f"No {path} yet -- run notebooks/07_phase2.ipynb first.")
    content = path.read_text(encoding="utf-8")
    if "Wilcoxon" in content and "Cliff" in content:
        return _result("stats_computed", "PASS", f"{path} present and contains Wilcoxon / Cliff's delta content.")
    return _result("stats_computed", "FAIL", f"{path} exists but does not look like a real stats report (missing Wilcoxon/Cliff's delta text).")


def check_efficiency_measured_not_estimated(paths) -> Dict[str, Any]:
    import pandas as pd

    path = Path(paths.results) / "efficiency.csv"
    if not path.is_file():
        return _result("efficiency_measured_not_estimated", "UNKNOWN", f"No {path} yet -- run notebooks/08_efficiency.ipynb first.")
    df = pd.read_csv(path)
    if df.empty:
        return _result("efficiency_measured_not_estimated", "UNKNOWN", f"{path} has no rows yet.")

    problems = []
    for _, row in df.iterrows():
        model_name = row.get("model_name")
        if not bool(row.get("fps_b1_is_gpu_measurement")) or not bool(row.get("fps_b8_is_gpu_measurement")):
            problems.append(f"{model_name}: FPS not GPU-measured (a CPU dry-run number must never be reported as this).")
        if pd.isna(row.get("peak_mem_MB_b8")):
            problems.append(f"{model_name}: peak_mem_MB_b8 missing.")
        if not row.get("gpu_name") or pd.isna(row.get("gpu_name")):
            problems.append(f"{model_name}: gpu_name missing.")
    if not problems:
        return _result("efficiency_measured_not_estimated", "PASS", f"All {len(df)} efficiency.csv rows are real GPU measurements (FPS/peak-mem/gpu_name all present).")
    return _result("efficiency_measured_not_estimated", "FAIL", "; ".join(problems))


# ---- [verify before submission] marker collection ---------------------------


def collect_verify_markers(methodology_path: Path) -> List[str]:
    content = methodology_path.read_text(encoding="utf-8")
    seen: List[str] = []
    for match in VERIFY_MARKER_RE.finditer(content):
        marker = match.group(0)
        if marker not in seen:
            seen.append(marker)
    return seen


# ---- orchestration -------------------------------------------------------


def run_all_checks(paths) -> List[Dict[str, Any]]:
    return [
        check_step0_audit(),
        check_alpha_and_wdelta_defaults(),
        check_per_group_wdelta_shared_across_directions(),
        check_both_descriptors_available(),
        check_active_datasets_full_scope(),
        check_cvc_sequence_split_zero_overlap(paths),
        check_same_scan_impl_for_all(paths),
        check_ablations_present(paths),
        check_mechanism_figures_present(paths),
        check_seeds_present(paths),
        check_stats_computed(paths),
        check_efficiency_measured_not_estimated(paths),
    ]


def print_checklist(checks: List[Dict[str, Any]]) -> None:
    print("=== Final pre-submission checklist (methodology section 16) ===\n")
    for c in checks:
        print(f"  [{c['status']:7s}] {c['name']}: {c['detail']}")
    counts: Dict[str, int] = {}
    for c in checks:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    print()
    print(f"Summary: {counts.get('PASS', 0)} PASS, {counts.get('UNKNOWN', 0)} UNKNOWN, {counts.get('FAIL', 0)} FAIL")
    if counts.get("FAIL"):
        print("FAIL items must be fixed before submission.")
    if counts.get("UNKNOWN"):
        print("UNKNOWN items need the corresponding Colab notebook run before they can be verified.")


def print_verify_markers(methodology_path: Path) -> List[str]:
    markers = collect_verify_markers(methodology_path)
    print(f"\n=== [verify before submission] markers found in {methodology_path.name} ({len(markers)}) ===")
    for m in markers:
        print(f"  TODO: {m}")
    return markers


def main() -> None:
    from lcmunet.paths import get_paths

    paths = get_paths()
    checks = run_all_checks(paths)
    print_checklist(checks)
    print_verify_markers(METHODOLOGY_PATH)


if __name__ == "__main__":
    main()
