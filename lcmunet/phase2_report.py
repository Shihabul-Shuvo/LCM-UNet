"""Phase-2 headline statistics + reporting (methodology section 8, 9, 14).

Orchestrates lcmunet.stats over the Phase-2 headline rows (Kvasir-SEG +
CVC-ClinicDB: baseline/GLGF/hero, 5 seeds), the ISIC generalisation rows
(hero + baseline, 3 seeds, no GLGF per this prompt's spec), the reproduced
comparators (U-Net/EGE-UNet/MALUNet), and results/cross_dataset_results.csv
(lcmunet/cross_dataset_eval.py) into results/phase2_summary.csv (a flat
table) and results/stats_report.md (the full narrative report).

Hard-gated: generate_phase2_summary() calls
lcmunet.gate2_report.require_gate2_proceed() itself (defence in depth --
the primary gate belongs BEFORE any Phase-2 GPU-hours are spent, in
notebooks/07_phase2.ipynb, but this function re-checks so it can never be
called standalone against a PIVOT'd Gate-2 state and produce a report that
looks like a green light).

Comparator availability: required on Kvasir-SEG (methodology section 9),
marked optional_if_time on CVC-ClinicDB (lcmunet/experiment_matrix.py). A
missing Kvasir comparator row is a hard failure (fail loud); a missing CVC
comparator row degrades the "vs best competitor" comparison to "not
available" in the report rather than crashing the whole thing.

DATASET SCOPE (lcmunet.config.ACTIVE_DATASETS): dataset_headline_report and
isic_generalisation_report check this UP FRONT and return a
{"in_scope": False, "note": ...} stub for any dataset not currently active,
rather than attempting a results.csv/per-image lookup that would raise
(nothing was ever trained for it). This renders as "N/A -- not in current
scope" in the stats report, never a crash or a blank/broken row. A full run
over all 4 datasets (ACTIVE_DATASETS = all of
lcmunet.data.raw_layout.DATASET_NAMES) is required before final submission
per the methodology.
"""

from __future__ import annotations

import csv as _csv
from pathlib import Path
from typing import Any, Dict, List

from lcmunet import config as config_module
from lcmunet import cross_dataset_eval as cde
from lcmunet import experiment_matrix as em
from lcmunet import stats
from lcmunet.config import RunConfig
from lcmunet.gate2_report import require_gate2_proceed

COMPARATOR_MODEL_NAMES = ("unet", "malunet", "egeunet")

NOT_IN_SCOPE_NOTE = (
    "N/A -- not in current scope (dataset not in lcmunet.config.ACTIVE_DATASETS). "
    "A full run over all 4 datasets is required before final submission per the methodology."
)


def _not_in_scope_report(dataset: str) -> Dict[str, Any]:
    return {"dataset": dataset, "in_scope": False, "note": f"{dataset}: {NOT_IN_SCOPE_NOTE}"}


def _headline_configs(headline_rows: Dict[str, RunConfig], dataset: str, model_role: str, seeds: List[int]) -> List[RunConfig]:
    return [headline_rows[f"phase2_headline_{dataset}_{model_role}_seed{seed}"] for seed in seeds]


def _comparator_configs(comparator_rows: Dict[str, RunConfig], dataset: str, role_prefix: str, seeds: List[int]) -> Dict[str, List[RunConfig]]:
    return {
        model_name: [comparator_rows[f"{role_prefix}_{model_name}_seed{seed}"] for seed in seeds]
        for model_name in COMPARATOR_MODEL_NAMES
    }


def _model_summary(paths, label: str, configs: List[RunConfig]) -> Dict[str, Any]:
    values = stats.seed_dsc_values(paths.results, configs)
    summary = stats.mean_std_ci(values)
    summary["label"] = label
    summary["model_name"] = configs[0].model_name
    return summary


def dataset_headline_report(
    paths,
    dataset: str,
    hero_descriptor_type: str,
    headline_seeds: List[int],
    comparators_required: bool,
) -> Dict[str, Any]:
    """One in-domain dataset's headline report: mean+/-std/CI for
    baseline/GLGF/hero DSC across seeds, plus 3 paired comparisons (hero vs
    baseline, hero vs GLGF, hero vs best competitor -- 'best competitor' is
    only included if comparator results are available; see module
    docstring for the required-vs-optional policy).

    Returns {"in_scope": False, "note": ...} immediately, without any
    results.csv lookup, if `dataset` is not currently in ACTIVE_DATASETS.
    """
    if dataset not in config_module.ACTIVE_DATASETS:
        return _not_in_scope_report(dataset)

    scan_impl, _source = em.resolve_scan_impl(paths)
    headline_rows = dict(em.build_phase2_headline(scan_impl, hero_descriptor_type=hero_descriptor_type))

    baseline_configs = _headline_configs(headline_rows, dataset, "baseline_pvm", headline_seeds)
    glgf_configs = _headline_configs(headline_rows, dataset, "glgf", headline_seeds)
    hero_configs = _headline_configs(headline_rows, dataset, "lc_ss2d_hero", headline_seeds)

    report: Dict[str, Any] = {
        "dataset": dataset,
        "in_scope": True,
        "models": {
            "baseline_pvm": _model_summary(paths, "Baseline PVM", baseline_configs),
            "glgf": _model_summary(paths, "GLGF late-fusion", glgf_configs),
            "hero": _model_summary(paths, "LC-SS2D (ours)", hero_configs),
        },
        "comparisons": {
            "hero_vs_baseline": stats.paired_comparison(paths.results, hero_configs, baseline_configs, "LC-SS2D", "Baseline PVM"),
            "hero_vs_glgf": stats.paired_comparison(paths.results, hero_configs, glgf_configs, "LC-SS2D", "GLGF"),
        },
        "best_competitor": None,
        "hero_vs_best_competitor": None,
    }

    comparator_rows = dict(em.build_phase2_comparators(scan_impl))
    role_prefix = "phase2_comparators_kvasir_seg" if dataset == "kvasir_seg" else "phase2_comparators_optional_if_time_cvc_clinicdb"
    comparator_seeds = list(em.OTHER_SEEDS)
    try:
        comparator_configs = _comparator_configs(comparator_rows, dataset, role_prefix, comparator_seeds)
        best_model, best_configs = stats.best_competitor(paths.results, comparator_configs)
    except (RuntimeError, KeyError) as exc:
        if comparators_required:
            raise RuntimeError(f"dataset={dataset}: required comparator results missing -- {exc}") from exc
        report["comparator_note"] = f"Comparators not available for {dataset} (optional_if_time, apparently skipped): {exc}"
        return report

    report["best_competitor"] = {"model_name": best_model, "summary": _model_summary(paths, best_model, best_configs)}
    report["hero_vs_best_competitor"] = stats.paired_comparison(paths.results, hero_configs, best_configs, "LC-SS2D", best_model)
    return report


def isic_generalisation_report(paths, dataset: str, hero_descriptor_type: str, seeds: List[int]) -> Dict[str, Any]:
    """Hero vs baseline only (no GLGF, no comparators, per this prompt's
    spec) on one ISIC version -- still a full paired comparison (same
    mechanism as the in-domain headline tables), just a smaller table.

    Returns {"in_scope": False, "note": ...} immediately, without any
    results.csv/per-image lookup, if `dataset` is not currently in
    ACTIVE_DATASETS (see module docstring).
    """
    if dataset not in config_module.ACTIVE_DATASETS:
        return _not_in_scope_report(dataset)

    scan_impl, _source = em.resolve_scan_impl(paths)
    isic_rows = dict(em.build_phase2_isic(scan_impl, hero_descriptor_type=hero_descriptor_type))

    baseline_configs = [isic_rows[f"phase2_{dataset}_baseline_pvm_seed{seed}"] for seed in seeds]
    hero_configs = [isic_rows[f"phase2_{dataset}_lc_ss2d_hero_seed{seed}"] for seed in seeds]

    return {
        "dataset": dataset,
        "in_scope": True,
        "models": {
            "baseline_pvm": _model_summary(paths, "Baseline PVM", baseline_configs),
            "hero": _model_summary(paths, "LC-SS2D (ours)", hero_configs),
        },
        "comparisons": {
            "hero_vs_baseline": stats.paired_comparison(paths.results, hero_configs, baseline_configs, "LC-SS2D", "Baseline PVM"),
        },
    }


def cross_dataset_report(paths) -> List[Dict[str, Any]]:
    """One row per (model_name, train_dataset, test_dataset), mean+/-std/CI
    of DSC over whatever seeds are present in results/cross_dataset_results.csv."""
    df = cde.load_cross_dataset_results(paths.results)
    rows = []
    for (model_name, train_dataset, test_dataset), group in df.groupby(["model_name", "train_dataset", "test_dataset"]):
        summary = stats.mean_std_ci(group["dsc"].astype(float).tolist())
        rows.append({"model_name": model_name, "train_dataset": train_dataset, "test_dataset": test_dataset, **summary})
    return rows


# ---- writers ----------------------------------------------------------------


def _write_summary_csv(path: Path, headline_reports: List[Dict[str, Any]], isic_reports: List[Dict[str, Any]], cross_rows: List[Dict[str, Any]]) -> None:
    columns = ["scope", "dataset", "train_dataset", "test_dataset", "model_name", "n_seeds", "mean_dsc", "std_dsc", "ci_low", "ci_high"]
    rows: List[Dict[str, Any]] = []

    for report in headline_reports:
        if not report.get("in_scope", True):
            continue  # N/A -- not in current scope; omitted from the CSV, noted in the .md instead
        for key, summary in report["models"].items():
            rows.append({
                "scope": "headline", "dataset": report["dataset"], "train_dataset": "", "test_dataset": "",
                "model_name": summary["model_name"], "n_seeds": summary["n"], "mean_dsc": summary["mean"],
                "std_dsc": summary["std"], "ci_low": summary["ci_low"], "ci_high": summary["ci_high"],
            })
        if report.get("best_competitor"):
            summary = report["best_competitor"]["summary"]
            rows.append({
                "scope": "headline_best_competitor", "dataset": report["dataset"], "train_dataset": "", "test_dataset": "",
                "model_name": summary["model_name"], "n_seeds": summary["n"], "mean_dsc": summary["mean"],
                "std_dsc": summary["std"], "ci_low": summary["ci_low"], "ci_high": summary["ci_high"],
            })

    for report in isic_reports:
        if not report.get("in_scope", True):
            continue  # N/A -- not in current scope; omitted from the CSV, noted in the .md instead
        for key, summary in report["models"].items():
            rows.append({
                "scope": "isic_generalisation", "dataset": report["dataset"], "train_dataset": "", "test_dataset": "",
                "model_name": summary["model_name"], "n_seeds": summary["n"], "mean_dsc": summary["mean"],
                "std_dsc": summary["std"], "ci_low": summary["ci_low"], "ci_high": summary["ci_high"],
            })

    for row in cross_rows:
        rows.append({
            "scope": "cross_dataset", "dataset": "", "train_dataset": row["train_dataset"], "test_dataset": row["test_dataset"],
            "model_name": row["model_name"], "n_seeds": row["n"], "mean_dsc": row["mean"],
            "std_dsc": row["std"], "ci_low": row["ci_low"], "ci_high": row["ci_high"],
        })

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"


def _render_headline_section(report: Dict[str, Any]) -> List[str]:
    if not report.get("in_scope", True):
        return [f"### {report['dataset']}", "", report["note"], ""]

    lines = [f"### {report['dataset']}", ""]
    lines.append("| Model | n seeds | mean DSC | std | 95% CI |")
    lines.append("|:--|--:|--:|--:|:--|")
    for key in ("baseline_pvm", "glgf", "hero"):
        s = report["models"][key]
        lines.append(f"| {s['label']} | {s['n']} | {_fmt_pct(s['mean'])} | {_fmt_pct(s['std'])} | [{_fmt_pct(s['ci_low'])}, {_fmt_pct(s['ci_high'])}] |")
    if report.get("best_competitor"):
        s = report["best_competitor"]["summary"]
        lines.append(f"| Best competitor: {s['label']} | {s['n']} | {_fmt_pct(s['mean'])} | {_fmt_pct(s['std'])} | [{_fmt_pct(s['ci_low'])}, {_fmt_pct(s['ci_high'])}] |")
    lines.append("")

    lines.append("| Comparison | mean diff (Dice pts) | Wilcoxon p | Cohen's d | Cliff's delta | n images |")
    lines.append("|:--|--:|--:|--:|--:|--:|")
    for comp in ("hero_vs_baseline", "hero_vs_glgf"):
        c = report["comparisons"][comp]
        lines.append(f"| {c['label_a']} vs {c['label_b']} | {c['mean_diff']*100:+.2f} | {c['wilcoxon_p']:.4g} | {c['cohens_d']:.3f} | {c['cliffs_delta']:.3f} | {c['n_images']} |")
    if report.get("hero_vs_best_competitor"):
        c = report["hero_vs_best_competitor"]
        lines.append(f"| {c['label_a']} vs {c['label_b']} (best competitor) | {c['mean_diff']*100:+.2f} | {c['wilcoxon_p']:.4g} | {c['cohens_d']:.3f} | {c['cliffs_delta']:.3f} | {c['n_images']} |")
    elif report.get("comparator_note"):
        lines.append("")
        lines.append(f"*{report['comparator_note']}*")
    lines.append("")
    return lines


def _render_isic_section(report: Dict[str, Any]) -> List[str]:
    if not report.get("in_scope", True):
        return [f"### {report['dataset']}", "", report["note"], ""]

    lines = [f"### {report['dataset']}", ""]
    lines.append("| Model | n seeds | mean DSC | std | 95% CI |")
    lines.append("|:--|--:|--:|--:|:--|")
    for key in ("baseline_pvm", "hero"):
        s = report["models"][key]
        lines.append(f"| {s['label']} | {s['n']} | {_fmt_pct(s['mean'])} | {_fmt_pct(s['std'])} | [{_fmt_pct(s['ci_low'])}, {_fmt_pct(s['ci_high'])}] |")
    lines.append("")
    c = report["comparisons"]["hero_vs_baseline"]
    lines.append("| Comparison | mean diff (Dice pts) | Wilcoxon p | Cohen's d | Cliff's delta | n images |")
    lines.append("|:--|--:|--:|--:|--:|--:|")
    lines.append(f"| {c['label_a']} vs {c['label_b']} | {c['mean_diff']*100:+.2f} | {c['wilcoxon_p']:.4g} | {c['cohens_d']:.3f} | {c['cliffs_delta']:.3f} | {c['n_images']} |")
    lines.append("")
    return lines


def render_stats_report_md(
    desc_winner: str,
    headline_reports: List[Dict[str, Any]],
    isic_reports: List[Dict[str, Any]],
    cross_rows: List[Dict[str, Any]],
) -> str:
    lines = ["# Phase-2 statistics report (methodology section 8, 9, 14)", ""]
    lines.append(
        f"Gate-2 PROCEED confirmed; Desc ablation winner **`{desc_winner}`** used as the fixed "
        "hero descriptor for every Phase-2 hero row (section 10.1: \"winner used in all later results\")."
    )
    lines.append("")
    lines.append(
        "Per-image Dice is averaged ACROSS SEEDS before pairing (flagged interpretation -- see "
        "lcmunet/stats.py's module docstring); Cohen's d and Cliff's delta are both reported "
        "(section 8 allows either)."
    )
    lines.append("")
    if set(config_module.ACTIVE_DATASETS) != {"kvasir_seg", "cvc_clinicdb", "isic2017", "isic2018"}:
        lines.append(
            f"**SCOPE: ACTIVE_DATASETS = {list(config_module.ACTIVE_DATASETS)}** (lcmunet/config.py). Any dataset "
            "not listed here is reported \"N/A -- not in current scope\" below, not an error. A full "
            "run over all 4 datasets is required before final submission per the methodology."
        )
        lines.append("")

    lines.append("## Headline results (Kvasir-SEG, CVC-ClinicDB)")
    lines.append("")
    for report in headline_reports:
        lines.extend(_render_headline_section(report))

    lines.append("## ISIC generalisation (hero + baseline only, no GLGF)")
    lines.append("")
    for report in isic_reports:
        lines.extend(_render_isic_section(report))

    lines.append("## Cross-dataset generalisation")
    lines.append("")
    if cross_rows:
        lines.append("| Model | Train | Test | n seeds | mean DSC | std | 95% CI |")
        lines.append("|:--|:--|:--|--:|--:|--:|:--|")
        for row in cross_rows:
            lines.append(
                f"| {row['model_name']} | {row['train_dataset']} | {row['test_dataset']} | {row['n']} | "
                f"{_fmt_pct(row['mean'])} | {_fmt_pct(row['std'])} | [{_fmt_pct(row['ci_low'])}, {_fmt_pct(row['ci_high'])}] |"
            )
    else:
        lines.append("*No cross-dataset results found in results/cross_dataset_results.csv -- run "
                      "lcmunet.cross_dataset_eval.run_cross_dataset_suite first.*")
    lines.append("")

    return "\n".join(lines)


def generate_phase2_summary(paths, headline_datasets=("kvasir_seg", "cvc_clinicdb"), isic_datasets=("isic2017", "isic2018")) -> Dict[str, Path]:
    """Requires Gate-2 PROCEED (re-checked here, not just trusted from the
    caller -- see module docstring). Writes results/phase2_summary.csv and
    results/stats_report.md; returns their paths.
    """
    gate2_rules = require_gate2_proceed(paths)
    desc_winner = gate2_rules["desc_winner"]

    headline_reports = [
        dataset_headline_report(
            paths, dataset, desc_winner, list(em.HEADLINE_SEEDS),
            comparators_required=(dataset == "kvasir_seg"),
        )
        for dataset in headline_datasets
    ]
    isic_reports = [
        isic_generalisation_report(paths, dataset, desc_winner, list(em.OTHER_SEEDS))
        for dataset in isic_datasets
    ]
    cross_rows = cross_dataset_report(paths)

    csv_path = Path(paths.results) / "phase2_summary.csv"
    _write_summary_csv(csv_path, headline_reports, isic_reports, cross_rows)

    md = render_stats_report_md(desc_winner, headline_reports, isic_reports, cross_rows)
    md_path = Path(paths.results) / "stats_report.md"
    md_path.write_text(md, encoding="utf-8")

    return {"summary_csv": csv_path, "stats_report_md": md_path}
