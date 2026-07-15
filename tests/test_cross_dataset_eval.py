import pandas as pd
import pytest

from lcmunet import cross_dataset_eval as cde
from lcmunet import experiment_matrix as em


def test_evaluate_cross_dataset_row_raises_when_no_checkpoint(paths):
    configs = dict(em.build_phase2_headline(scan_impl="ref"))
    hero_config = configs["phase2_headline_kvasir_seg_lc_ss2d_hero_seed42"]

    with pytest.raises(RuntimeError, match="No trained checkpoint"):
        cde.evaluate_cross_dataset_row(paths, hero_config, test_dataset="cvc_clinicdb")


def test_evaluate_cross_dataset_row_real_checkpoint_real_cross_eval(make_kvasir_raw, make_cvc_raw, monkeypatch):
    """Trains a real (cheap) lc_ss2d checkpoint on Kvasir, then evaluates it
    on CVC-ClinicDB's test split -- exercises the actual mechanism this
    module exists for (reusing an in-domain checkpoint cross-dataset), no
    mocking of the model/loader/metrics pipeline.
    """
    from lcmunet.data.splits import build_kvasir_split, build_cvc_split
    from lcmunet.engine import run_one

    monkeypatch.setattr(em, "EPOCHS", 1)
    monkeypatch.setattr(em, "INPUT_SIZE", 64)
    monkeypatch.setattr(em, "BATCH_SIZE", 4)

    paths = make_kvasir_raw(n=20)
    paths = make_cvc_raw(n=612)  # full count -- the sequence-level splitter needs enough frames per sequence for a non-empty test partition
    build_kvasir_split(paths)
    build_cvc_split(paths)

    configs = dict(em.build_phase2_headline(scan_impl="ref"))
    hero_config = configs["phase2_headline_kvasir_seg_lc_ss2d_hero_seed42"]
    run_one(hero_config, paths=paths, num_workers=0)

    row = cde.evaluate_cross_dataset_row(paths, hero_config, test_dataset="cvc_clinicdb")

    assert row["model_name"] == "lc_ss2d"
    assert row["train_dataset"] == "kvasir_seg"
    assert row["test_dataset"] == "cvc_clinicdb"
    assert row["seed"] == 42
    assert 0.0 <= row["dsc"] <= 1.0


def test_cross_dataset_upsert_is_idempotent_no_duplicates(paths):
    row = dict(
        model_name="lc_ss2d", descriptor_type="contrast", train_dataset="kvasir_seg",
        test_dataset="cvc_clinicdb", seed=42, dsc=0.5, miou=0.4, sensitivity=0.5,
        specificity=0.9, accuracy=0.8, hd95=10.0, assd=5.0,
    )
    cde._upsert(paths.results, row)
    row2 = dict(row, dsc=0.6)  # re-run with a different value -- must replace, not append
    cde._upsert(paths.results, row2)

    df = cde.load_cross_dataset_results(paths.results)
    assert len(df) == 1
    assert df.iloc[0]["dsc"] == 0.6


def test_run_cross_dataset_suite_evaluates_all_pairs_and_seeds(paths, monkeypatch):
    """Orchestration test: mocks evaluate_cross_dataset_row (no real
    training/GPU needed) to verify run_cross_dataset_suite calls it with
    the correct (config, test_dataset) for every direction/model/seed
    combination and upserts every result."""
    calls = []

    def fake_evaluate(paths_arg, in_domain_config, test_dataset):
        calls.append((in_domain_config.model_name, in_domain_config.dataset, test_dataset, in_domain_config.seed))
        return {
            "model_name": in_domain_config.model_name,
            "descriptor_type": in_domain_config.model_cfg.get("descriptor_type"),
            "train_dataset": in_domain_config.dataset,
            "test_dataset": test_dataset,
            "seed": in_domain_config.seed,
            "dsc": 0.5, "miou": 0.4, "sensitivity": 0.5, "specificity": 0.9,
            "accuracy": 0.8, "hd95": 10.0, "assd": 5.0,
        }

    monkeypatch.setattr(cde, "evaluate_cross_dataset_row", fake_evaluate)

    df = cde.run_cross_dataset_suite(paths, hero_descriptor_type="contrast", seeds=[42, 43])

    # 2 directions x 2 models (baseline_pvm, lc_ss2d hero) x 2 seeds = 8 rows
    assert len(calls) == 8
    assert len(df) == 8
    assert set(df["train_dataset"]) == {"kvasir_seg", "cvc_clinicdb"}
    assert set(df["model_name"]) == {"ultralight_baseline", "lc_ss2d"}
    assert set(df["seed"]) == {42, 43}
    for train_ds, test_ds, *_ in [(c[1], c[2]) for c in calls]:
        assert train_ds != test_ds  # never "cross-dataset" to itself
