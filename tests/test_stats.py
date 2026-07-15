import numpy as np
import pytest

from lcmunet import stats
from lcmunet.config import RunConfig
from lcmunet.metrics import save_per_image_dice
from lcmunet.results_store import upsert_result


def _configs(model_name, seeds, dataset="kvasir_seg"):
    """Real RunConfig objects (distinct config_id per seed, exactly like
    production) -- NOT a single hand-picked config_id reused across seeds,
    which cannot happen in results.csv (seed is part of config_id's hash)."""
    return [
        RunConfig(
            run_name=f"{model_name}_{dataset}_seed{seed}", model_name=model_name, dataset=dataset,
            seed=seed, split_file=f"splits/{dataset}.json",
        )
        for seed in seeds
    ]


def _result_row(config, dsc):
    return dict(
        config_id=config.config_id, model_name=config.model_name, dataset=config.dataset, seed=config.seed,
        split_file=config.split_file, dsc=dsc, miou=dsc * 0.9, sensitivity=dsc,
        specificity=0.9, accuracy=0.9, hd95=10.0, assd=5.0, scan_impl="ref",
    )


# ---- mean_std_ci ------------------------------------------------------------


def test_mean_std_ci_basic():
    result = stats.mean_std_ci([0.70, 0.72, 0.71, 0.73, 0.69])
    assert result["n"] == 5
    assert result["mean"] == pytest.approx(0.71)
    assert result["std"] > 0
    assert result["ci_low"] < result["mean"] < result["ci_high"]


def test_mean_std_ci_single_value_has_nan_ci():
    result = stats.mean_std_ci([0.70])
    assert result["n"] == 1
    assert result["std"] == 0.0
    assert np.isnan(result["ci_low"]) and np.isnan(result["ci_high"])


def test_mean_std_ci_raises_on_empty():
    with pytest.raises(ValueError):
        stats.mean_std_ci([])


# ---- seed_dsc_values ---------------------------------------------------------


def test_seed_dsc_values_reads_results_csv_across_distinct_config_ids(paths):
    configs = _configs("lc_ss2d", [42, 43, 44])
    assert len({c.config_id for c in configs}) == 3  # each seed really is a distinct config_id
    for config, dsc in zip(configs, [0.70, 0.71, 0.72]):
        upsert_result(paths.results, _result_row(config, dsc))

    values = stats.seed_dsc_values(paths.results, configs)
    assert values == [0.70, 0.71, 0.72]


def test_seed_dsc_values_raises_on_missing_seed(paths):
    configs = _configs("lc_ss2d", [42, 43])
    upsert_result(paths.results, _result_row(configs[0], 0.70))
    with pytest.raises(RuntimeError, match="missing DONE results.csv row"):
        stats.seed_dsc_values(paths.results, configs)


# ---- aligned_mean_per_image_dsc ----------------------------------------------


def test_aligned_mean_per_image_dsc_averages_across_seeds(paths):
    configs = _configs("lc_ss2d", [42, 43])
    ids = ["img_a", "img_b", "img_c"]
    save_per_image_dice(configs[0].config_id, configs[0].seed, ids, np.array([0.8, 0.6, 0.9]), paths.results)
    save_per_image_dice(configs[1].config_id, configs[1].seed, ids, np.array([0.6, 0.8, 0.7]), paths.results)

    mean_dsc, ref_ids = stats.aligned_mean_per_image_dsc(paths.results, configs)
    assert ref_ids == sorted(ids)
    expected = {"img_a": 0.7, "img_b": 0.7, "img_c": 0.8}
    for img_id, val in zip(ref_ids, mean_dsc):
        assert val == pytest.approx(expected[img_id])


def test_aligned_mean_per_image_dsc_raises_on_id_mismatch_across_seeds(paths):
    configs = _configs("lc_ss2d", [42, 43])
    save_per_image_dice(configs[0].config_id, configs[0].seed, ["img_a", "img_b"], np.array([0.8, 0.6]), paths.results)
    save_per_image_dice(configs[1].config_id, configs[1].seed, ["img_a", "img_x"], np.array([0.8, 0.6]), paths.results)

    with pytest.raises(RuntimeError, match="id set"):
        stats.aligned_mean_per_image_dsc(paths.results, configs)


# ---- cohens_d_paired / cliffs_delta -------------------------------------------


def test_cohens_d_and_cliffs_delta_zero_when_identical():
    a = np.array([0.7, 0.8, 0.6, 0.9])
    assert stats.cohens_d_paired(a, a) == 0.0
    assert stats.cliffs_delta(a, a) == 0.0


def test_cliffs_delta_is_plus_one_when_a_always_greater():
    a = np.array([0.9, 0.85, 0.7, 0.6])
    b = np.array([0.5, 0.4, 0.3, 0.2])
    assert stats.cliffs_delta(a, b) == 1.0
    assert stats.cliffs_delta(b, a) == -1.0


def test_cohens_d_paired_matches_manual_formula():
    a = np.array([0.9, 0.7, 0.8, 0.6])
    b = np.array([0.5, 0.6, 0.3, 0.5])
    diff = a - b
    expected = diff.mean() / diff.std(ddof=1)
    assert stats.cohens_d_paired(a, b) == pytest.approx(expected)


# ---- paired_comparison --------------------------------------------------------


def test_paired_comparison_computes_wilcoxon_and_effect_sizes(paths):
    hero_configs = _configs("lc_ss2d", [42])
    baseline_configs = _configs("ultralight_baseline", [42])

    ids = [f"img_{i}" for i in range(20)]
    rng = np.random.default_rng(0)
    hero_dsc = np.clip(0.75 + rng.normal(0, 0.05, size=20), 0, 1)
    baseline_dsc = np.clip(0.65 + rng.normal(0, 0.05, size=20), 0, 1)

    save_per_image_dice(hero_configs[0].config_id, 42, ids, hero_dsc, paths.results)
    save_per_image_dice(baseline_configs[0].config_id, 42, ids, baseline_dsc, paths.results)

    result = stats.paired_comparison(paths.results, hero_configs, baseline_configs, "hero", "baseline")

    assert result["n_images"] == 20
    assert 0.0 <= result["wilcoxon_p"] <= 1.0
    assert result["mean_diff"] == pytest.approx(result["mean_dsc_a"] - result["mean_dsc_b"])
    assert result["cliffs_delta"] > 0  # hero constructed to be higher on average


def test_paired_comparison_identical_values_gives_p_one_no_crash(paths):
    configs_a = _configs("lc_ss2d", [42])
    configs_b = _configs("glgf", [42])
    ids = ["img_a", "img_b", "img_c"]
    same = np.array([0.7, 0.8, 0.6])
    save_per_image_dice(configs_a[0].config_id, 42, ids, same, paths.results)
    save_per_image_dice(configs_b[0].config_id, 42, ids, same, paths.results)

    result = stats.paired_comparison(paths.results, configs_a, configs_b, "a", "b")
    assert result["wilcoxon_p"] == 1.0
    assert result["cohens_d"] == 0.0
    assert result["cliffs_delta"] == 0.0


def test_paired_comparison_raises_on_mismatched_test_sets(paths):
    configs_a = _configs("lc_ss2d", [42])
    configs_b = _configs("glgf", [42])
    save_per_image_dice(configs_a[0].config_id, 42, ["img_a", "img_b"], np.array([0.7, 0.8]), paths.results)
    save_per_image_dice(configs_b[0].config_id, 42, ["img_x", "img_y"], np.array([0.7, 0.8]), paths.results)

    with pytest.raises(RuntimeError, match="different test image id sets"):
        stats.paired_comparison(paths.results, configs_a, configs_b, "a", "b")


# ---- best_competitor -----------------------------------------------------------


def test_best_competitor_picks_highest_mean_dsc(paths):
    unet_configs = _configs("unet", [42, 43])
    malunet_configs = _configs("malunet", [42, 43])
    egeunet_configs = _configs("egeunet", [42, 43])

    for config, dsc in zip(unet_configs, [0.60, 0.62]):
        upsert_result(paths.results, _result_row(config, dsc))
    for config, dsc in zip(malunet_configs, [0.70, 0.72]):
        upsert_result(paths.results, _result_row(config, dsc))
    for config, dsc in zip(egeunet_configs, [0.65, 0.66]):
        upsert_result(paths.results, _result_row(config, dsc))

    comparators = {"unet": unet_configs, "malunet": malunet_configs, "egeunet": egeunet_configs}
    best_model, configs = stats.best_competitor(paths.results, comparators)
    assert best_model == "malunet"
    assert configs == malunet_configs


def test_best_competitor_raises_on_empty():
    with pytest.raises(ValueError):
        stats.best_competitor(None, {})
