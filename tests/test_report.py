"""lcmunet/report.py (methodology sections 8, 9, 15, 16)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

import lcmunet.report as report


# ---- cite-only rows ----------------------------------------------------------


def test_load_cite_only_rows_missing_file_returns_empty_with_note(tmp_path):
    rows, notes = report.load_cite_only_rows(tmp_path / "nope.yaml")
    assert rows == []
    assert len(notes) == 1
    assert "No cite-only numbers file" in notes[0]


def test_load_cite_only_rows_excludes_unverified_and_includes_verified(tmp_path):
    path = tmp_path / "cite_only.yaml"
    path.write_text(yaml.safe_dump([
        {"model_name": "VM-UNet", "dataset": "kvasir_seg", "dsc": 0.8621, "split_verified_match": False},
        {"model_name": "VM-UNetV2", "dataset": "kvasir_seg", "dsc": 0.9075, "split_verified_match": True},
    ]), encoding="utf-8")

    rows, notes = report.load_cite_only_rows(path)
    assert [r["model_name"] for r in rows] == ["VM-UNetV2"]
    assert len(notes) == 1
    assert "VM-UNet" in notes[0] and "OMITTED" in notes[0]


# ---- metric readers ------------------------------------------------------------


def test_row_metric_raises_clearly_when_missing(paths):
    from lcmunet.config import RunConfig

    config = RunConfig(run_name="x", model_name="lc_ss2d", dataset="kvasir_seg", seed=0, split_file="splits/kvasir_seg.json")
    with pytest.raises(RuntimeError, match="missing DONE results.csv row"):
        report._row_metric(paths.results, config, "dsc")


def test_seed_metric_values_reads_named_column(paths):
    from lcmunet.config import RunConfig
    from lcmunet.results_store import upsert_result

    config = RunConfig(run_name="x", model_name="lc_ss2d", dataset="kvasir_seg", seed=0, split_file="splits/kvasir_seg.json")
    upsert_result(paths.results, {
        "config_id": config.config_id, "model_name": "lc_ss2d", "dataset": "kvasir_seg", "seed": 0,
        "split_file": config.split_file, "dsc": 0.9, "miou": 0.8,
    })
    values = report._seed_metric_values(paths.results, [config], "miou")
    assert values == [0.8]


# ---- rendering ------------------------------------------------------------------


def test_df_to_markdown_formats_floats_and_handles_empty():
    df = pd.DataFrame({"model": ["A"], "dsc_mean": [0.91234]})
    md = report._df_to_markdown(df, float_cols=["dsc_mean"])
    assert "0.9123" in md
    assert "A" in md

    assert "no rows" in report._df_to_markdown(pd.DataFrame())


def test_write_table_writes_csv_md_tex(tmp_path):
    df = pd.DataFrame({"model": ["A", "B"], "dsc": [0.9, 0.8]})
    paths_dict = report._write_table(df, tmp_path, "demo", "Demo table", float_cols=["dsc"])
    assert paths_dict["csv"].is_file()
    assert paths_dict["md"].is_file()
    assert paths_dict["tex"].is_file()
    assert "Demo table" in paths_dict["tex"].read_text(encoding="utf-8")


def test_write_table_empty_dataframe_writes_placeholder_tex(tmp_path):
    paths_dict = report._write_table(pd.DataFrame(), tmp_path, "empty", "Empty table")
    assert "no rows available" in paths_dict["tex"].read_text(encoding="utf-8")


# ---- _flatten_comparisons ---------------------------------------------------


def _model_summary(mean, ci_low, ci_high, label="X"):
    return {"label": label, "mean": mean, "std": 0.01, "ci_low": ci_low, "ci_high": ci_high, "n": 5, "model_name": label}


def _comparison(label_a, label_b, mean_diff=0.01, p=0.03, d=0.5, cliff=0.4, n=100):
    return {"label_a": label_a, "label_b": label_b, "mean_diff": mean_diff, "wilcoxon_p": p, "cohens_d": d, "cliffs_delta": cliff, "n_images": n}


def test_flatten_comparisons_headline_report_all_three_pairs():
    report_dict = {
        "models": {
            "hero": _model_summary(0.92, 0.90, 0.94, "LC-SS2D"),
            "baseline_pvm": _model_summary(0.90, 0.88, 0.92, "Baseline"),
            "glgf": _model_summary(0.91, 0.89, 0.93, "GLGF"),
        },
        "comparisons": {
            "hero_vs_baseline": _comparison("LC-SS2D", "Baseline"),
            "hero_vs_glgf": _comparison("LC-SS2D", "GLGF"),
        },
        "best_competitor": {"model_name": "unet", "summary": _model_summary(0.85, 0.83, 0.87, "U-Net")},
        "hero_vs_best_competitor": _comparison("LC-SS2D", "U-Net"),
    }
    rows = report._flatten_comparisons("kvasir_seg", report_dict)
    assert len(rows) == 3
    labels = {r["comparison"] for r in rows}
    assert labels == {"LC-SS2D vs Baseline", "LC-SS2D vs GLGF", "LC-SS2D vs U-Net"}
    for r in rows:
        assert r["dataset"] == "kvasir_seg"
        assert r["mean_dsc_a"] == 0.92  # always the hero


def test_flatten_comparisons_isic_report_hero_vs_baseline_only():
    report_dict = {
        "models": {"hero": _model_summary(0.80, 0.78, 0.82, "LC-SS2D"), "baseline_pvm": _model_summary(0.78, 0.76, 0.80, "Baseline")},
        "comparisons": {"hero_vs_baseline": _comparison("LC-SS2D", "Baseline")},
    }
    rows = report._flatten_comparisons("isic2017", report_dict)
    assert len(rows) == 1
    assert rows[0]["comparison"] == "LC-SS2D vs Baseline"
    assert rows[0]["dataset"] == "isic2017"


# ---- statistics_table: orchestration mocked, phase2_report itself already tested --


def test_statistics_table_collects_rows_across_datasets_and_notes_failures(monkeypatch):
    calls = []

    def fake_headline(paths, dataset, hero_descriptor_type, seeds, comparators_required):
        calls.append(dataset)
        if dataset == "cvc_clinicdb":
            raise RuntimeError("missing DONE results.csv rows for cvc_clinicdb")
        return {
            "models": {"hero": _model_summary(0.9, 0.88, 0.92), "baseline_pvm": _model_summary(0.85, 0.83, 0.87), "glgf": _model_summary(0.87, 0.85, 0.89)},
            "comparisons": {"hero_vs_baseline": _comparison("LC-SS2D", "Baseline"), "hero_vs_glgf": _comparison("LC-SS2D", "GLGF")},
        }

    def fake_isic(paths, dataset, hero_descriptor_type, seeds):
        return {
            "models": {"hero": _model_summary(0.8, 0.78, 0.82), "baseline_pvm": _model_summary(0.78, 0.76, 0.80)},
            "comparisons": {"hero_vs_baseline": _comparison("LC-SS2D", "Baseline")},
        }

    monkeypatch.setattr(report.pr, "dataset_headline_report", fake_headline)
    monkeypatch.setattr(report.pr, "isic_generalisation_report", fake_isic)

    df, notes = report.statistics_table(None)

    assert set(calls) == {"kvasir_seg", "cvc_clinicdb"}
    assert any("cvc_clinicdb" in n and "SKIPPED" in n for n in notes)
    # kvasir_seg (2 comparisons) + isic2017 + isic2018 (1 each) = 4 rows
    assert len(df) == 4
    assert set(df["dataset"]) == {"kvasir_seg", "isic2017", "isic2018"}


# ---- cross_dataset_table: real small CSV, no training needed ----------------


def test_cross_dataset_table_reads_real_csv(paths):
    from lcmunet.cross_dataset_eval import _upsert

    _upsert(paths.results, {
        "model_name": "lc_ss2d", "descriptor_type": "contrast", "train_dataset": "kvasir_seg",
        "test_dataset": "cvc_clinicdb", "seed": 42, "dsc": 0.75, "miou": 0.6,
        "sensitivity": 0.7, "specificity": 0.9, "accuracy": 0.85, "hd95": 10.0, "assd": 3.0,
    })
    df = report.cross_dataset_table(paths)
    assert len(df) == 1
    assert df.iloc[0]["model_name"] == "lc_ss2d"
    assert df.iloc[0]["train_dataset"] == "kvasir_seg"


def test_cross_dataset_table_empty_when_no_results(paths):
    df = report.cross_dataset_table(paths)
    assert df.empty


# ---- ablation_table: real (tiny) Phase-1 training + fresh local efficiency --


def test_ablation_table_real_rows_plus_synthetic_baseline(make_kvasir_raw, monkeypatch):
    from lcmunet import experiment_matrix as em
    from lcmunet.data.splits import build_kvasir_split
    from lcmunet.engine import run_one
    from lcmunet.results_store import upsert_result

    monkeypatch.setattr(em, "EPOCHS", 1)
    monkeypatch.setattr(em, "INPUT_SIZE", 32)
    monkeypatch.setattr(em, "BATCH_SIZE", 4)

    paths = make_kvasir_raw(n=20)
    build_kvasir_split(paths)

    configs = dict(em.build_phase1_kvasir(scan_impl="ref"))
    cpu_runnable_roles = [role for role, _label in report.ABLATION_ROWS if role != "phase1_ablation_A_baseline_pvm"]
    for role in cpu_runnable_roles:
        run_one(configs[role], paths=paths, num_workers=0)

    baseline_config = configs["phase1_ablation_A_baseline_pvm"]
    upsert_result(paths.results, {
        "config_id": baseline_config.config_id, "model_name": baseline_config.model_name,
        "dataset": baseline_config.dataset, "seed": baseline_config.seed, "split_file": baseline_config.split_file,
        "dsc": 0.01, "miou": 0.01, "sensitivity": 0.01, "specificity": 0.5, "accuracy": 0.5,
        "hd95": 50.0, "assd": 30.0, "scan_impl": "cuda",
        "notes": "SYNTHETIC stand-in for ultralight_baseline (GPU-gated) -- test fixture only.",
    })

    df, notes = report.ablation_table(paths, input_size=32)

    assert len(df) == len(report.ABLATION_ROWS)
    assert notes == []
    baseline_row = df[df["role"] == "phase1_ablation_A_baseline_pvm"].iloc[0]
    assert pd.isna(baseline_row["params_M"])
    assert "GPU-gated" in baseline_row["efficiency_source"]

    hero_row = df[df["role"] == "phase1_ablation_A_lc_ss2d_hero"].iloc[0]
    assert hero_row["params_M"] > 0
    assert hero_row["gflops_total"] > 0
    assert "measured fresh" in hero_row["efficiency_source"]

    p0_row = df[df["role"] == "phase1_placement_P0"].iloc[0]
    # P0 has zero active LC-VSS stages -- strictly fewer params than the hero (4 active stages)
    assert p0_row["params_M"] < hero_row["params_M"]


# ---- main_comparison_table: real (tiny, single-seed) training + a synthetic --
# ---- efficiency.csv + a synthetic baseline_pvm row (same GPU-gated pattern) --


def test_main_comparison_table_real_rows_plus_efficiency_join(make_kvasir_raw, monkeypatch):
    from lcmunet import experiment_matrix as em
    from lcmunet.data.splits import build_kvasir_split
    from lcmunet.engine import run_one
    from lcmunet.results_store import upsert_result

    monkeypatch.setattr(em, "EPOCHS", 1)
    monkeypatch.setattr(em, "INPUT_SIZE", 32)
    monkeypatch.setattr(em, "BATCH_SIZE", 4)
    monkeypatch.setattr(em, "HEADLINE_SEEDS", [42])
    monkeypatch.setattr(em, "OTHER_SEEDS", [42])

    paths = make_kvasir_raw(n=20)
    build_kvasir_split(paths)

    scan_impl = "ref"
    headline_rows = dict(em.build_phase2_headline(scan_impl))
    comparator_rows = dict(em.build_phase2_comparators(scan_impl))

    for role in ("phase2_headline_kvasir_seg_glgf_seed42", "phase2_headline_kvasir_seg_lc_ss2d_hero_seed42"):
        run_one(headline_rows[role], paths=paths, num_workers=0)
    for model_name in ("unet", "malunet", "egeunet"):
        run_one(comparator_rows[f"phase2_comparators_kvasir_seg_{model_name}_seed42"], paths=paths, num_workers=0)

    baseline_config = headline_rows["phase2_headline_kvasir_seg_baseline_pvm_seed42"]
    upsert_result(paths.results, {
        "config_id": baseline_config.config_id, "model_name": baseline_config.model_name,
        "dataset": baseline_config.dataset, "seed": baseline_config.seed, "split_file": baseline_config.split_file,
        "dsc": 0.5, "miou": 0.4, "sensitivity": 0.5, "specificity": 0.5, "accuracy": 0.5,
        "hd95": 20.0, "assd": 10.0, "scan_impl": "cuda", "notes": "SYNTHETIC stand-in for ultralight_baseline.",
    })

    eff_df = pd.DataFrame([
        {"model_name": name, "params_M": i + 0.1, "gflops_total": i + 1.0, "fps_b1": 100.0, "fps_b8": 500.0}
        for i, name in enumerate(["ultralight_baseline", "glgf", "lc_ss2d", "unet", "malunet", "egeunet"])
    ])
    (paths.results / "efficiency.csv").parent.mkdir(parents=True, exist_ok=True)
    eff_df.to_csv(paths.results / "efficiency.csv", index=False)

    df, notes = report.main_comparison_table(paths, dataset="kvasir_seg")

    assert len(notes) == 1
    assert f"No cite-only numbers file at {report.DEFAULT_CITE_ONLY_PATH}" in notes[0]
    assert set(df["model_name"]) == {"ultralight_baseline", "glgf", "lc_ss2d", "unet", "malunet", "egeunet"}
    assert (df["cite_only"] == False).all()  # noqa: E712
    hero_row = df[df["model_name"] == "lc_ss2d"].iloc[0]
    assert hero_row["params_M"] == pytest.approx(2.1)
    assert hero_row["n_seeds"] == 1


# ---- generate_paper_tables: orchestration resilience, sub-functions mocked --


def test_generate_paper_tables_writes_successful_tables_and_records_failures(paths, monkeypatch):
    monkeypatch.setattr(report, "main_comparison_table", lambda *a, **k: (pd.DataFrame({"model": ["A"], "dsc_mean": [0.9]}), []))
    monkeypatch.setattr(report, "ablation_table", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("phase-1 not done")))
    monkeypatch.setattr(report, "statistics_table", lambda *a, **k: (pd.DataFrame({"dataset": ["kvasir_seg"], "wilcoxon_p": [0.01]}), ["note-a"]))
    monkeypatch.setattr(report, "cross_dataset_table", lambda *a, **k: pd.DataFrame())

    result = report.generate_paper_tables(paths)

    assert set(result["tables"].keys()) == {"main_comparison", "statistics", "cross_dataset"}
    assert "ablation" in result["errors"]
    assert "phase-1 not done" in result["errors"]["ablation"]

    for name in ("main_comparison", "statistics", "cross_dataset"):
        assert result["tables"][name]["csv"].is_file()
        assert result["tables"][name]["md"].is_file()
        assert result["tables"][name]["tex"].is_file()

    readme = (paths.results / "paper_tables" / "README.md").read_text(encoding="utf-8")
    assert "NOT PRODUCED" in readme
    assert "note-a" in readme
