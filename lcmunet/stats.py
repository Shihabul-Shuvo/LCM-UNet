"""Statistical reporting for headline rows (methodology section 8).

"5 seeds -> mean +/- std", "95% CI on mean Dice", "Paired Wilcoxon
signed-rank on per-image Dice (ours vs baseline; ours vs GLGF; ours vs best
competitor). Paired = same test images in both runs", and an effect size
("Cliff's delta or Cohen's d"; this module reports BOTH, for robustness,
rather than picking one). "Report honestly even if small" -- nothing here
is gated on the result looking good; every number is computed the same way
regardless of sign or magnitude.

IDENTITY NOTE: RunConfig.seed is a hashed field of RunConfig.config_id
(lcmunet/config.py) -- a single config_id therefore corresponds to exactly
ONE seed, never several. Every function below that spans multiple seeds of
"the same experiment" (same model_name/dataset/model_cfg) accordingly takes
a LIST of RunConfig objects (one per seed, distinct config_ids), never a
single config_id paired with a seed list -- that combination cannot exist
in results.csv/per-image storage and would silently look up nothing (or
the wrong thing) for every seed after the first.

PAIRING (flagged interpretation -- the methodology doesn't spell out the
exact multi-seed aggregation for the paired test): per-image Dice is
averaged ACROSS SEEDS first (5 seeds for headline rows, 3 for comparators),
producing one mean-per-image-Dice vector per model per dataset, which is
what gets paired and compared. This uses the full multi-seed protocol
rather than arbitrarily picking one seed's predictions, and every seed's
per-image id set is asserted identical before averaging (same split_file
-> same test partition is a Fairness-rule invariant, not just a hope).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats as scipy_stats

from lcmunet.config import RunConfig
from lcmunet.metrics import load_per_image_dice
from lcmunet.results_store import load_results


def mean_std_ci(values: List[float], confidence: float = 0.95) -> Dict[str, float]:
    """mean +/- std (ddof=1) and a CI on the mean (section 8: "95% CI on
    mean Dice"). Uses a t-distribution critical value (appropriate, and more
    honest than a normal z-value, at small n like 5 seeds). n=1 returns a
    NaN half-width (a CI is undefined from a single observation) rather
    than fabricating a width of 0.
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        raise ValueError("mean_std_ci: no values given")
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        se = std / math.sqrt(n)
        t_crit = float(scipy_stats.t.ppf(1 - (1 - confidence) / 2, df=n - 1))
        half_width = t_crit * se
    else:
        half_width = float("nan")
    return {"n": n, "mean": mean, "std": std, "ci_low": mean - half_width, "ci_high": mean + half_width}


def seed_dsc_values(results_dir, configs: List[RunConfig]) -> List[float]:
    """DSC (results.csv) for each config in `configs` -- one RunConfig per
    seed of the SAME experiment (same model_name/dataset/model_cfg); each
    has its own distinct config_id (seed is part of the hash). Raises
    listing every missing (config_id, seed) if any row isn't DONE yet."""
    df = load_results(results_dir)
    values = []
    missing = []
    for config in configs:
        match = df[(df["config_id"] == config.config_id) & (df["seed"] == config.seed)]
        if len(match) == 0:
            missing.append((config.config_id, config.seed))
            continue
        values.append(float(match.iloc[-1]["dsc"]))
    if missing:
        raise RuntimeError(f"missing DONE results.csv row(s) for (config_id, seed): {missing}")
    return values


def aligned_mean_per_image_dsc(results_dir, configs: List[RunConfig]) -> Tuple[np.ndarray, List[str]]:
    """Loads per-image Dice for each RunConfig in `configs` (one per seed,
    distinct config_ids), asserts every seed has the IDENTICAL image id set
    (same split -> same test partition, Fairness rule), and returns
    (mean-across-seeds array, sorted id list).
    """
    if not configs:
        raise ValueError("aligned_mean_per_image_dsc: no configs given")
    per_seed_series = []
    ref_ids: List[str] | None = None
    for config in configs:
        payload = load_per_image_dice(config.config_id, config.seed, results_dir)
        ids = [str(i) for i in payload["ids"]]
        dsc = payload["dsc"]
        order = sorted(range(len(ids)), key=lambda i: ids[i])
        ids_sorted = [ids[i] for i in order]
        dsc_sorted = np.asarray([dsc[i] for i in order], dtype=float)
        if ref_ids is None:
            ref_ids = ids_sorted
        elif ids_sorted != ref_ids:
            raise RuntimeError(
                f"config_id={config.config_id} (seed={config.seed}): per-image id set "
                "differs from an earlier seed of the same experiment -- the test "
                "partition must be IDENTICAL across seeds (Fairness rule, same "
                "split_file). Cannot average across seeds for statistics."
            )
        per_seed_series.append(dsc_sorted)
    stacked = np.stack(per_seed_series, axis=0)  # (n_seeds, n_images)
    return stacked.mean(axis=0), ref_ids  # type: ignore[return-value]


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired/related samples: mean(diff) / std(diff, ddof=1)."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sd = float(diff.std(ddof=1))
    if sd == 0.0:
        return 0.0 if float(diff.mean()) == 0.0 else math.copysign(float("inf"), diff.mean())
    return float(diff.mean() / sd)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Classical two-sample Cliff's delta: (#{a_i>b_j} - #{a_i<b_j}) / (n*m)
    over all cross pairs. O(n*m); trivial at test-set sizes here (<=1000
    images per dataset -> <=1e6 pairs).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    greater = int((a[:, None] > b[None, :]).sum())
    less = int((a[:, None] < b[None, :]).sum())
    return float((greater - less) / (len(a) * len(b)))


def paired_comparison(
    results_dir,
    configs_a: List[RunConfig],
    configs_b: List[RunConfig],
    label_a: str,
    label_b: str,
) -> Dict[str, Any]:
    """Paired Wilcoxon signed-rank + both effect sizes between two models'
    mean-across-seeds per-image Dice, aligned and asserted-identical by
    image id (section 8: "Paired = same test images in both runs")."""
    mean_a, ids_a = aligned_mean_per_image_dsc(results_dir, configs_a)
    mean_b, ids_b = aligned_mean_per_image_dsc(results_dir, configs_b)
    if ids_a != ids_b:
        raise RuntimeError(
            f"Cannot pair {label_a!r} with {label_b!r}: different test image id sets -- "
            "these must share the same split_file/test partition to be paired (section 8: "
            "'Paired = same test images in both runs')."
        )

    if np.allclose(mean_a, mean_b):
        # scipy.stats.wilcoxon raises on an all-zero difference vector rather
        # than returning a (statistic, p=1.0) no-effect result -- report that
        # explicitly instead of letting an exception surface as a crash.
        statistic, p_value = 0.0, 1.0
    else:
        statistic, p_value = scipy_stats.wilcoxon(mean_a, mean_b)

    return {
        "label_a": label_a,
        "label_b": label_b,
        "n_images": len(ids_a),
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_p": float(p_value),
        "cohens_d": cohens_d_paired(mean_a, mean_b),
        "cliffs_delta": cliffs_delta(mean_a, mean_b),
        "mean_dsc_a": float(mean_a.mean()),
        "mean_dsc_b": float(mean_b.mean()),
        "mean_diff": float(mean_a.mean() - mean_b.mean()),
    }


def best_competitor(results_dir, comparator_configs: Dict[str, List[RunConfig]]) -> Tuple[str, List[RunConfig]]:
    """comparator_configs: {model_name: [RunConfig, ...]} (one RunConfig per
    seed) for the reproduced comparators on one dataset (methodology
    section 9: U-Net, EGE-UNet, MALUNet). Returns (model_name, configs) for
    whichever has the highest mean DSC across its seeds -- determined from
    actual results, never assumed. Raises if comparator_configs is empty.
    """
    if not comparator_configs:
        raise ValueError("best_competitor: no comparator configs given")
    means = {}
    for model_name, configs in comparator_configs.items():
        values = seed_dsc_values(results_dir, configs)
        means[model_name] = float(np.mean(values))
    best_model = max(means, key=means.get)
    return best_model, comparator_configs[best_model]
