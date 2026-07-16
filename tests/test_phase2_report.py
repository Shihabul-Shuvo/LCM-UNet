"""End-to-end test for lcmunet/phase2_report.py: real (cheap) training for
every CPU-runnable role across Kvasir-SEG, CVC-ClinicDB, ISIC2017, ISIC2018
+ synthetic stand-ins for the GPU-gated ultralight_baseline rows (same
rationale as tests/test_gate2_report.py) + synthetic cross-dataset rows
(lcmunet/cross_dataset_eval.py's own mechanism is tested separately in
tests/test_cross_dataset_eval.py -- this test exercises phase2_report's
AGGREGATION of already-produced results, not re-proving cross-dataset
evaluation itself).
"""

import numpy as np

from lcmunet import config as config_module
from lcmunet import experiment_matrix as em
from lcmunet import phase2_report as pr
from lcmunet import cross_dataset_eval as cde
from lcmunet.metrics import load_per_image_dice, save_per_image_dice
from lcmunet.results_store import upsert_result


def _fake_baseline_row_and_perimage(paths, config, real_hero_config):
    """Synthesises a results.csv row + per-image Dice file for the
    GPU-gated ultralight_baseline, reusing the REAL test image id set from
    an already-trained (CPU-runnable) config on the same dataset/seed so
    paired comparisons can align -- NOT a real measurement, test fixture
    only (mirrors tests/test_gate2_report.py's identical rationale)."""
    real = load_per_image_dice(real_hero_config.config_id, real_hero_config.seed, paths.results)
    ids = [str(i) for i in real["ids"]]
    rng = np.random.default_rng(config.seed)
    dsc = np.clip(0.55 + rng.normal(0, 0.05, size=len(ids)), 0, 1)
    save_per_image_dice(config.config_id, config.seed, ids, dsc, paths.results)
    upsert_result(paths.results, {
        "config_id": config.config_id, "model_name": config.model_name, "dataset": config.dataset,
        "seed": config.seed, "split_file": config.split_file, "dsc": float(dsc.mean()),
        "miou": float(dsc.mean()) * 0.9, "sensitivity": float(dsc.mean()), "specificity": 0.9,
        "accuracy": 0.9, "hd95": 20.0, "assd": 10.0, "scan_impl": "ref",
        "notes": "SYNTHETIC stand-in for ultralight_baseline -- test fixture only.",
    })


def _write_isic_override(paths, version, ids):
    """build_isic_split() requires EXACTLY the real dataset's image count
    (2000/2594) unless an ultralight_splits override file exactly partitions
    whatever ids actually exist -- writing a tiny override lets this test
    use a small n instead of generating thousands of synthetic images."""
    import json

    train, val, test = ids[:16], ids[16:18], ids[18:20]
    override_dir = paths.data_raw / "ultralight_splits"
    override_dir.mkdir(parents=True, exist_ok=True)
    with open(override_dir / f"{version}.json", "w", encoding="utf-8") as f:
        json.dump({"train": train, "val": val, "test": test}, f)


def test_generate_phase2_summary_end_to_end(make_kvasir_raw, make_cvc_raw, make_isic_raw, monkeypatch):
    from lcmunet.data import raw_layout as rl
    from lcmunet.data.splits import build_kvasir_split, build_cvc_split, build_isic_split
    from lcmunet.engine import run_one

    # This test exercises the FULL 4-dataset report (including ISIC), so all
    # 4 must be in scope regardless of the current ACTIVE_DATASETS default
    # (lcmunet/config.py) -- see test_phase2_report_isic_not_in_scope below
    # for the "ISIC excluded" behaviour this same module now also supports.
    monkeypatch.setattr(config_module, "ACTIVE_DATASETS", ["kvasir_seg", "cvc_clinicdb", "isic2017", "isic2018"])

    monkeypatch.setattr(em, "EPOCHS", 1)
    monkeypatch.setattr(em, "INPUT_SIZE", 64)
    monkeypatch.setattr(em, "BATCH_SIZE", 4)
    monkeypatch.setattr(em, "HEADLINE_SEEDS", [42, 43])
    monkeypatch.setattr(em, "OTHER_SEEDS", [42, 43])

    # Gate-2 is exhaustively tested in tests/test_gate2_report.py already --
    # here we only need a fixed, known-PROCEED result so phase2_report's OWN
    # logic (not Gate-2's) is what's under test.
    monkeypatch.setattr(pr, "require_gate2_proceed", lambda paths: {"desc_winner": "contrast"})

    paths = make_kvasir_raw(n=20)
    paths = make_cvc_raw(n=612)  # sequence-level splitter needs enough frames per sequence for a non-empty test partition
    paths = make_isic_raw(version="isic2017", n=20)
    paths = make_isic_raw(version="isic2018", n=20)
    build_kvasir_split(paths)
    build_cvc_split(paths)

    for version in ("isic2017", "isic2018"):
        ids = sorted(p.id for p in rl.list_isic_pairs(paths.data_raw, version))
        _write_isic_override(paths, version, ids)
    build_isic_split(paths, "isic2017")
    build_isic_split(paths, "isic2018")

    headline_rows = dict(em.build_phase2_headline(scan_impl="ref"))
    isic_rows = dict(em.build_phase2_isic(scan_impl="ref"))
    comparator_rows = dict(em.build_phase2_comparators(scan_impl="ref"))

    # --- Kvasir: glgf + hero + all 3 (REQUIRED) comparators, real training ---
    for seed in em.HEADLINE_SEEDS:
        run_one(headline_rows[f"phase2_headline_kvasir_seg_glgf_seed{seed}"], paths=paths, num_workers=0)
        run_one(headline_rows[f"phase2_headline_kvasir_seg_lc_ss2d_hero_seed{seed}"], paths=paths, num_workers=0)
    for seed in em.OTHER_SEEDS:
        for model_name in ("unet", "malunet", "egeunet"):
            run_one(comparator_rows[f"phase2_comparators_kvasir_seg_{model_name}_seed{seed}"], paths=paths, num_workers=0)

    # --- CVC: glgf + hero real; NO comparators (exercises the optional_if_time skip path) ---
    for seed in em.HEADLINE_SEEDS:
        run_one(headline_rows[f"phase2_headline_cvc_clinicdb_glgf_seed{seed}"], paths=paths, num_workers=0)
        run_one(headline_rows[f"phase2_headline_cvc_clinicdb_lc_ss2d_hero_seed{seed}"], paths=paths, num_workers=0)

    # --- ISIC2017/2018: hero real ---
    for dataset in ("isic2017", "isic2018"):
        for seed in em.OTHER_SEEDS:
            run_one(isic_rows[f"phase2_{dataset}_lc_ss2d_hero_seed{seed}"], paths=paths, num_workers=0)

    # --- synthetic ultralight_baseline stand-ins (GPU-gated, cannot run here) ---
    for dataset in ("kvasir_seg", "cvc_clinicdb"):
        hero_role = f"phase2_headline_{dataset}_lc_ss2d_hero_seed"
        for seed in em.HEADLINE_SEEDS:
            baseline_config = headline_rows[f"phase2_headline_{dataset}_baseline_pvm_seed{seed}"]
            hero_config = headline_rows[f"{hero_role}{seed}"]
            _fake_baseline_row_and_perimage(paths, baseline_config, hero_config)
    for dataset in ("isic2017", "isic2018"):
        for seed in em.OTHER_SEEDS:
            baseline_config = isic_rows[f"phase2_{dataset}_baseline_pvm_seed{seed}"]
            hero_config = isic_rows[f"phase2_{dataset}_lc_ss2d_hero_seed{seed}"]
            _fake_baseline_row_and_perimage(paths, baseline_config, hero_config)

    # --- synthetic cross-dataset rows (cross_dataset_eval's own mechanism tested elsewhere) ---
    for train_ds, test_ds in [("kvasir_seg", "cvc_clinicdb"), ("cvc_clinicdb", "kvasir_seg")]:
        for model_name in ("ultralight_baseline", "lc_ss2d"):
            for seed in em.HEADLINE_SEEDS:
                cde._upsert(paths.results, {
                    "model_name": model_name, "descriptor_type": "contrast" if model_name == "lc_ss2d" else None,
                    "train_dataset": train_ds, "test_dataset": test_ds, "seed": seed,
                    "dsc": 0.5, "miou": 0.4, "sensitivity": 0.5, "specificity": 0.9,
                    "accuracy": 0.8, "hd95": 15.0, "assd": 8.0,
                })

    result_paths = pr.generate_phase2_summary(paths, headline_datasets=("kvasir_seg", "cvc_clinicdb"), isic_datasets=("isic2017", "isic2018"))

    assert result_paths["summary_csv"].is_file()
    assert result_paths["stats_report_md"].is_file()

    import pandas as pd
    csv_df = pd.read_csv(result_paths["summary_csv"])
    assert set(csv_df["scope"]) >= {"headline", "isic_generalisation", "cross_dataset", "headline_best_competitor"}
    # kvasir headline has a best competitor row; cvc headline does not (no comparators run there)
    kvasir_best = csv_df[(csv_df["scope"] == "headline_best_competitor") & (csv_df["dataset"] == "kvasir_seg")]
    assert len(kvasir_best) == 1
    cvc_best = csv_df[(csv_df["scope"] == "headline_best_competitor") & (csv_df["dataset"] == "cvc_clinicdb")]
    assert len(cvc_best) == 0

    md = result_paths["stats_report_md"].read_text(encoding="utf-8")
    assert "kvasir_seg" in md and "cvc_clinicdb" in md and "isic2017" in md and "isic2018" in md
    assert "Best competitor" in md
    assert "not available" in md.lower() or "optional_if_time" in md.lower()  # CVC's graceful-degradation note
    assert "Cross-dataset generalisation" in md


def test_isic_generalisation_report_not_in_scope_returns_stub(paths, monkeypatch):
    """With ACTIVE_DATASETS excluding ISIC (the current production default),
    isic_generalisation_report must return an in_scope=False stub -- no
    results.csv/per-image lookup at all, no crash, no blank/broken row."""
    monkeypatch.setattr(config_module, "ACTIVE_DATASETS", ["kvasir_seg", "cvc_clinicdb"])

    report = pr.isic_generalisation_report(paths, "isic2017", "contrast", [42, 43, 44])

    assert report == {
        "dataset": "isic2017",
        "in_scope": False,
        "note": f"isic2017: {pr.NOT_IN_SCOPE_NOTE}",
    }
    assert "N/A" in report["note"]
    assert "not in current scope" in report["note"]


def test_dataset_headline_report_not_in_scope_returns_stub(paths, monkeypatch):
    monkeypatch.setattr(config_module, "ACTIVE_DATASETS", ["kvasir_seg"])  # cvc_clinicdb excluded

    report = pr.dataset_headline_report(paths, "cvc_clinicdb", "contrast", [42], comparators_required=False)

    assert report["in_scope"] is False
    assert "cvc_clinicdb" in report["note"]


def test_render_sections_show_note_instead_of_crashing_when_not_in_scope():
    stub = pr._not_in_scope_report("isic2018")
    lines = pr._render_isic_section(stub)
    assert any("N/A" in line for line in lines)
    assert any("not in current scope" in line for line in lines)

    headline_lines = pr._render_headline_section(stub)
    assert any("N/A" in line for line in headline_lines)


def test_write_summary_csv_omits_out_of_scope_reports(tmp_path):
    stub = pr._not_in_scope_report("isic2017")
    out_path = tmp_path / "phase2_summary.csv"
    pr._write_summary_csv(out_path, headline_reports=[], isic_reports=[stub], cross_rows=[])

    import pandas as pd

    df = pd.read_csv(out_path)
    assert len(df) == 0  # no rows at all for a fully-out-of-scope report set


def test_render_stats_report_md_notes_partial_scope():
    md = pr.render_stats_report_md(
        desc_winner="contrast",
        headline_reports=[],
        isic_reports=[pr._not_in_scope_report("isic2017"), pr._not_in_scope_report("isic2018")],
        cross_rows=[],
    )
    assert "SCOPE: ACTIVE_DATASETS" in md
    assert "N/A" in md
    assert "full run over all 4 datasets is required" in md
