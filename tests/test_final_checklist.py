"""scripts/final_checklist.py (methodology section 16)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import final_checklist as fc  # noqa: E402


# ---- code-level checks (always determinable, no Colab needed) --------------


def test_check_step0_audit_passes_on_real_code():
    result = fc.check_step0_audit()
    assert result["status"] == "PASS"


def test_check_step0_audit_fails_when_audit_raises(monkeypatch):
    import lcmunet.audit as audit_mod

    def _boom(verbose=True):
        raise AssertionError("W_delta.weight.grad is all zero")

    monkeypatch.setattr(audit_mod, "run_step0_audit", _boom)
    # check_step0_audit() does `from lcmunet.audit import run_step0_audit`
    # freshly on every call, so patching the source module is what takes effect.
    result = fc.check_step0_audit()
    assert result["status"] == "FAIL"
    assert "W_delta.weight.grad" in result["detail"]


def test_check_alpha_and_wdelta_defaults_passes_on_real_config():
    result = fc.check_alpha_and_wdelta_defaults()
    assert result["status"] == "PASS"


def test_check_alpha_and_wdelta_defaults_fails_on_wrong_value(monkeypatch):
    import lcmunet.config as config_mod

    monkeypatch.setitem(config_mod.DEFAULT_MODEL_CFG, "alpha_init", 0.0)
    result = fc.check_alpha_and_wdelta_defaults()
    assert result["status"] == "FAIL"


def test_check_per_group_wdelta_passes_on_real_model():
    result = fc.check_per_group_wdelta_shared_across_directions()
    assert result["status"] == "PASS"
    assert "4 distinct" in result["detail"]


def test_check_both_descriptors_available_passes_on_real_model():
    result = fc.check_both_descriptors_available()
    assert result["status"] == "PASS"


# ---- artifact-dependent checks: UNKNOWN when missing -----------------------


def test_cvc_check_unknown_when_split_map_missing(paths):
    result = fc.check_cvc_sequence_split_zero_overlap(paths)
    assert result["status"] == "UNKNOWN"


def test_same_scan_impl_check_unknown_when_efficiency_csv_missing(paths):
    result = fc.check_same_scan_impl_for_all(paths)
    assert result["status"] == "UNKNOWN"


def test_ablations_present_unknown_when_no_results(paths):
    result = fc.check_ablations_present(paths)
    assert result["status"] == "UNKNOWN"
    assert "10/10" in result["detail"]


def test_mechanism_figures_present_unknown_when_missing(paths):
    result = fc.check_mechanism_figures_present(paths)
    assert result["status"] == "UNKNOWN"


def test_seeds_present_unknown_when_summary_missing(paths):
    result = fc.check_seeds_present(paths)
    assert result["status"] == "UNKNOWN"


def test_stats_computed_unknown_when_report_missing(paths):
    result = fc.check_stats_computed(paths)
    assert result["status"] == "UNKNOWN"


def test_efficiency_measured_unknown_when_csv_missing(paths):
    result = fc.check_efficiency_measured_not_estimated(paths)
    assert result["status"] == "UNKNOWN"


# ---- artifact-dependent checks: PASS/FAIL once the artifact exists ---------


def _write_cvc_sequence_map(paths, leaky: bool) -> Path:
    sequence_of_frame = {"1": 1, "2": 1, "3": 2, "4": 2}
    partition_of_frame = {"1": "train", "2": "test" if leaky else "train", "3": "val", "4": "val"}
    payload = {
        "n_sequences": 2, "mapping_source": "test fixture",
        "sequence_of_frame": sequence_of_frame, "partition_of_frame": partition_of_frame,
    }
    path = Path(paths.splits) / "cvc_sequence_map.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_cvc_check_passes_on_clean_split(paths):
    _write_cvc_sequence_map(paths, leaky=False)
    result = fc.check_cvc_sequence_split_zero_overlap(paths)
    assert result["status"] == "PASS"


def test_cvc_check_fails_on_leaky_split(paths):
    _write_cvc_sequence_map(paths, leaky=True)
    result = fc.check_cvc_sequence_split_zero_overlap(paths)
    assert result["status"] == "FAIL"


def _write_efficiency_csv(paths, rows) -> Path:
    path = Path(paths.results) / "efficiency.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_same_scan_impl_check_passes_when_uniform(paths):
    _write_efficiency_csv(paths, [{"model_name": "unet", "scan_impl": "cuda"}, {"model_name": "glgf", "scan_impl": "cuda"}])
    result = fc.check_same_scan_impl_for_all(paths)
    assert result["status"] == "PASS"


def test_same_scan_impl_check_fails_when_mixed(paths):
    _write_efficiency_csv(paths, [{"model_name": "unet", "scan_impl": "cuda"}, {"model_name": "glgf", "scan_impl": "ref"}])
    result = fc.check_same_scan_impl_for_all(paths)
    assert result["status"] == "FAIL"


def test_ablations_present_passes_once_all_ten_rows_are_done(paths):
    from lcmunet import experiment_matrix as em
    from lcmunet.results_store import upsert_result

    phase1_rows = dict(em.build_phase1_kvasir(scan_impl="ref"))
    for role, config in phase1_rows.items():
        upsert_result(paths.results, {
            "config_id": config.config_id, "model_name": config.model_name, "dataset": config.dataset,
            "seed": config.seed, "split_file": config.split_file, "dsc": 0.5, "miou": 0.4,
        })
    result = fc.check_ablations_present(paths)
    assert result["status"] == "PASS"


def test_mechanism_figures_present_passes_once_all_files_exist(paths):
    for name in ("delta_difference_map.png", "region_wise_modulation.png", "per_stage_alpha.png"):
        (Path(paths.figures) / name).write_bytes(b"fake png")
    (Path(paths.results) / "mechanism_report.md").write_text("report", encoding="utf-8")
    result = fc.check_mechanism_figures_present(paths)
    assert result["status"] == "PASS"


def _write_phase2_summary(paths, headline_n=5, comp_n=3, isic_n=3) -> Path:
    rows = [
        {"scope": "headline", "dataset": "kvasir_seg", "model_name": "lc_ss2d", "n_seeds": headline_n},
        {"scope": "headline_best_competitor", "dataset": "kvasir_seg", "model_name": "unet", "n_seeds": comp_n},
        {"scope": "isic_generalisation", "dataset": "isic2017", "model_name": "lc_ss2d", "n_seeds": isic_n},
    ]
    path = Path(paths.results) / "phase2_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_seeds_present_passes_on_correct_counts(paths):
    _write_phase2_summary(paths)
    result = fc.check_seeds_present(paths)
    assert result["status"] == "PASS"


def test_seeds_present_fails_on_wrong_counts(paths):
    _write_phase2_summary(paths, headline_n=3)  # should be 5
    result = fc.check_seeds_present(paths)
    assert result["status"] == "FAIL"
    assert "headline" in result["detail"]


def test_seeds_present_passes_when_isic_missing_but_not_in_active_datasets(paths, monkeypatch):
    from lcmunet import config as config_module

    monkeypatch.setattr(config_module, "ACTIVE_DATASETS", ["kvasir_seg", "cvc_clinicdb"])

    rows = [
        {"scope": "headline", "dataset": "kvasir_seg", "model_name": "lc_ss2d", "n_seeds": 5},
        {"scope": "headline_best_competitor", "dataset": "kvasir_seg", "model_name": "unet", "n_seeds": 3},
        # deliberately NO isic_generalisation row at all
    ]
    path = Path(paths.results) / "phase2_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)

    result = fc.check_seeds_present(paths)
    assert result["status"] == "PASS"
    assert "N/A" in result["detail"] or "not in current scope" in result["detail"]


def test_seeds_present_fails_when_isic_missing_and_isic_is_active(paths, monkeypatch):
    from lcmunet import config as config_module

    monkeypatch.setattr(config_module, "ACTIVE_DATASETS", ["kvasir_seg", "cvc_clinicdb", "isic2017", "isic2018"])

    rows = [
        {"scope": "headline", "dataset": "kvasir_seg", "model_name": "lc_ss2d", "n_seeds": 5},
        {"scope": "headline_best_competitor", "dataset": "kvasir_seg", "model_name": "unet", "n_seeds": 3},
        # no isic_generalisation row -- now a real problem since ISIC is active
    ]
    path = Path(paths.results) / "phase2_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)

    result = fc.check_seeds_present(paths)
    assert result["status"] == "FAIL"
    assert "isic_generalisation" in result["detail"]


def test_active_datasets_full_scope_fails_when_isic_excluded(monkeypatch):
    from lcmunet import config as config_module

    monkeypatch.setattr(config_module, "ACTIVE_DATASETS", ["kvasir_seg", "cvc_clinicdb"])

    result = fc.check_active_datasets_full_scope()
    assert result["status"] == "FAIL"
    assert "isic2017" in result["detail"] and "isic2018" in result["detail"]
    assert "full run over all 4" in result["detail"]


def test_active_datasets_full_scope_passes_when_all_four_active(monkeypatch):
    from lcmunet import config as config_module

    monkeypatch.setattr(config_module, "ACTIVE_DATASETS", ["kvasir_seg", "cvc_clinicdb", "isic2017", "isic2018"])

    result = fc.check_active_datasets_full_scope()
    assert result["status"] == "PASS"


def test_stats_computed_passes_when_report_has_expected_content(paths):
    path = Path(paths.results) / "stats_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("## Results\nWilcoxon p=0.01, Cliff's delta=0.4", encoding="utf-8")
    result = fc.check_stats_computed(paths)
    assert result["status"] == "PASS"


def test_stats_computed_fails_when_report_looks_like_a_stub(paths):
    path = Path(paths.results) / "stats_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("TODO", encoding="utf-8")
    result = fc.check_stats_computed(paths)
    assert result["status"] == "FAIL"


def test_efficiency_measured_passes_on_real_gpu_rows(paths):
    _write_efficiency_csv(paths, [{
        "model_name": "unet", "fps_b1_is_gpu_measurement": True, "fps_b8_is_gpu_measurement": True,
        "peak_mem_MB_b8": 512.0, "gpu_name": "Tesla T4",
    }])
    result = fc.check_efficiency_measured_not_estimated(paths)
    assert result["status"] == "PASS"


def test_efficiency_measured_fails_on_cpu_dry_run_row(paths):
    _write_efficiency_csv(paths, [{
        "model_name": "unet", "fps_b1_is_gpu_measurement": False, "fps_b8_is_gpu_measurement": False,
        "peak_mem_MB_b8": None, "gpu_name": None,
    }])
    result = fc.check_efficiency_measured_not_estimated(paths)
    assert result["status"] == "FAIL"
    assert "unet" in result["detail"]


# ---- [verify before submission] marker collection ---------------------------


def test_collect_verify_markers_finds_all_unique_markers(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text(
        "Some text [verify before submission] more text.\n"
        "Another one: [verify HCMUNet or ECM-TransUNet details before citing].\n"
        "Repeated marker [verify before submission] again.\n"
        "Not a marker: [descriptor_type].\n",
        encoding="utf-8",
    )
    markers = fc.collect_verify_markers(path)
    assert markers == ["[verify before submission]", "[verify HCMUNet or ECM-TransUNet details before citing]"]


def test_collect_verify_markers_against_real_methodology_doc_is_nonempty():
    markers = fc.collect_verify_markers(fc.METHODOLOGY_PATH)
    assert len(markers) > 0
    assert all("verify" in m.lower() for m in markers)


# ---- orchestration smoke tests ----------------------------------------------


def test_run_all_checks_returns_one_result_per_check(paths):
    checks = fc.run_all_checks(paths)
    names = [c["name"] for c in checks]
    assert len(names) == len(set(names))  # no duplicate check names
    assert all(c["status"] in ("PASS", "UNKNOWN", "FAIL") for c in checks)


def test_print_checklist_and_verify_markers_do_not_crash(paths, capsys):
    checks = fc.run_all_checks(paths)
    fc.print_checklist(checks)
    fc.print_verify_markers(fc.METHODOLOGY_PATH)
    out = capsys.readouterr().out
    assert "Summary:" in out
    assert "verify before submission" in out.lower()
