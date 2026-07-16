"""Paper-ready table assembly (methodology sections 8, 9, 15, 16).

Reads ALREADY-PRODUCED artifacts under results/ (results.csv, efficiency.csv,
per-image Dice .npy files, cross_dataset_results.csv) and reformats them into
results/paper_tables/*.csv + *.md + *.tex. "Mostly local" (this prompt): the
network/GPU-touching prerequisites (real datasets, real training, real
Colab-measured FPS/memory) already happened in earlier prompts; this module
only reads what they wrote, plus a small amount of fresh LOCAL, CPU-only
Params/GFLOPs measurement for ablation-row model_cfgs that efficiency.csv
never covered (see ablation_table's docstring) -- it never retrains, and
never recomputes headline statistics from scratch: the "statistics" table
reuses lcmunet.phase2_report's already-tested paired-comparison functions
rather than re-deriving Wilcoxon/effect-size logic a second time.

Four tables (main comparison / ablation / statistics / cross-dataset), each
produced independently in its own try/except -- a missing prerequisite for
one (e.g. Phase-2 not yet run) does not block the others (same
orchestration-resilience pattern as lcmunet.efficiency_report.build_all_models
/ lcmunet.data.splits.build_all_splits). Every skipped table/row is named in
a returned "notes"/"errors" structure and in results/paper_tables/README.md
-- never silently omitted without a trace.

CITE-ONLY ROWS (section 9, docs/CITE_ONLY_MODELS.md). VM-UNet/VM-UNetV2/
LightM-UNet/H-VMUNet are never reproduced locally and their numbers are
NEVER fabricated here. An optional external file (default
docs/cite_only_numbers.yaml) may supply them; a row is included in the main
comparison table ONLY if it is present AND explicitly marked
`split_verified_match: true` (with the mandated footnote); every other row
-- including the entire file being absent, which is the current state, per
docs/CITE_ONLY_MODELS.md: "No number from this group has been verified
against our split as of this document" -- is OMITTED and named in the notes
as pending verification, never included with a guessed or third-party-
aggregator number.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from lcmunet import experiment_matrix as em
from lcmunet import phase2_report as pr
from lcmunet.config import RunConfig
from lcmunet.results_store import load_results

CITE_ONLY_FOOTNOTE = "Results from original paper; split may differ from ours."
DEFAULT_CITE_ONLY_PATH = Path("docs/cite_only_numbers.yaml")

MAIN_TABLE_LABELS: Dict[str, str] = {
    "unet": "U-Net (reproduced)",
    "ultralight_baseline": "UltraLight VM-UNet (reproduced)",
    "malunet": "MALUNet (reproduced)",
    "egeunet": "EGE-UNet (reproduced)",
    "glgf": "GLGF late-fusion (reproduced)",
    "lc_ss2d": "LC-SS2D / LCM-UNet (ours)",
}

# (role, label) -- methodology sections 10.1/10.2's full ablation + placement
# matrix. Placement="hero" is the Ablation-A hero row itself, not repeated.
ABLATION_ROWS: List[Tuple[str, str]] = [
    ("phase1_ablation_A_baseline_pvm", "Baseline PVM (reproduced)"),
    ("phase1_ablation_A_glgf", "GLGF late-fusion"),
    ("phase1_ablation_A_lc_ss2d_hero", "LC-SS2D (ours, hero placement)"),
    ("phase1_ablation_C_k1_control", "Ablation C: 1x1 capacity-matched control"),
    ("phase1_ablation_D_inject_input", "Ablation D: DWConv -> input u"),
    ("phase1_desc_plain", "Desc ablation: plain DWConv3x3"),
    ("phase1_placement_P0", "Placement: P0 (no LC-VSS)"),
    ("phase1_placement_E4E5", "Placement: E4+E5"),
    ("phase1_placement_plusE6", "Placement: hero+E6"),
    ("phase1_placement_E4D4", "Placement: E4+D4"),
]


# ---- shared metric readers (results.csv) ------------------------------------


def _row_metric(results_dir, config: RunConfig, metric: str) -> float:
    df = load_results(results_dir)
    match = df[(df["config_id"] == config.config_id) & (df["seed"] == config.seed)]
    if len(match) == 0:
        raise RuntimeError(
            f"missing DONE results.csv row for config_id={config.config_id} "
            f"seed={config.seed} ({config.model_name}/{config.dataset})"
        )
    return float(match.iloc[-1][metric])


def _seed_metric_values(results_dir, configs: List[RunConfig], metric: str) -> List[float]:
    """Generalises lcmunet.stats.seed_dsc_values to an arbitrary results.csv
    metric column (mIoU here; stats.py itself stays dsc-specific since it is
    the module methodology section 8's paired-significance machinery reads,
    and mIoU never needs a paired test, only mean+/-std -- methodology
    section 9)."""
    df = load_results(results_dir)
    values: List[float] = []
    missing = []
    for config in configs:
        match = df[(df["config_id"] == config.config_id) & (df["seed"] == config.seed)]
        if len(match) == 0:
            missing.append((config.config_id, config.seed))
            continue
        values.append(float(match.iloc[-1][metric]))
    if missing:
        raise RuntimeError(f"missing DONE results.csv row(s) for (config_id, seed): {missing}")
    return values


def _load_efficiency_rows(paths) -> Dict[str, Dict[str, Any]]:
    path = Path(paths.results) / "efficiency.csv"
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    return {row["model_name"]: row.to_dict() for _, row in df.iterrows()}


# ---- cite-only rows (section 9) ---------------------------------------------


def load_cite_only_rows(path: Optional[Path] = None) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Returns (included_rows, notes). See module docstring's CITE-ONLY ROWS
    section -- only explicitly `split_verified_match: true` rows are ever
    included; everything else is named in `notes`, never silently dropped."""
    path = Path(path) if path is not None else DEFAULT_CITE_ONLY_PATH
    if not path.is_file():
        return [], [
            f"No cite-only numbers file at {path} -- main comparison table has "
            "ZERO cite-only rows (VM-UNet/VM-UNetV2/LightM-UNet/H-VMUNet); see "
            "docs/CITE_ONLY_MODELS.md."
        ]

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or []

    included: List[Dict[str, Any]] = []
    notes: List[str] = []
    for entry in raw:
        if entry.get("split_verified_match") is True:
            included.append(entry)
        else:
            notes.append(
                f"Cite-only row OMITTED (not verified against our split): "
                f"{entry.get('model_name')} / {entry.get('dataset')} -- see docs/CITE_ONLY_MODELS.md."
            )
    return included, notes


# ---- main comparison table (section 9) --------------------------------------


def main_comparison_table(
    paths,
    dataset: str = "kvasir_seg",
    hero_descriptor_type: str = "contrast",
    cite_only_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Reproduced comparators + baseline/GLGF/hero, DSC/mIoU mean+/-std
    (across methodology section 6's seeds) joined with efficiency.csv's
    Params/GFLOPs/FPS, plus any verified cite-only rows for `dataset`."""
    scan_impl, _source = em.resolve_scan_impl(paths)
    headline_rows = dict(em.build_phase2_headline(scan_impl, hero_descriptor_type=hero_descriptor_type))
    comparator_rows = dict(em.build_phase2_comparators(scan_impl))
    efficiency_by_model = _load_efficiency_rows(paths)

    role_prefix = "phase2_comparators_kvasir_seg" if dataset == "kvasir_seg" else "phase2_comparators_optional_if_time_cvc_clinicdb"

    entries: List[Tuple[str, List[RunConfig]]] = [
        ("ultralight_baseline", [headline_rows[f"phase2_headline_{dataset}_baseline_pvm_seed{s}"] for s in em.HEADLINE_SEEDS]),
        ("glgf", [headline_rows[f"phase2_headline_{dataset}_glgf_seed{s}"] for s in em.HEADLINE_SEEDS]),
        ("lc_ss2d", [headline_rows[f"phase2_headline_{dataset}_lc_ss2d_hero_seed{s}"] for s in em.HEADLINE_SEEDS]),
    ]
    for model_name in ("unet", "malunet", "egeunet"):
        entries.append((model_name, [comparator_rows[f"{role_prefix}_{model_name}_seed{s}"] for s in em.OTHER_SEEDS]))

    rows: List[Dict[str, Any]] = []
    notes: List[str] = []
    for model_name, configs in entries:
        label = MAIN_TABLE_LABELS[model_name]
        try:
            dsc_vals = _seed_metric_values(paths.results, configs, "dsc")
            miou_vals = _seed_metric_values(paths.results, configs, "miou")
        except RuntimeError as exc:
            notes.append(f"{label}: SKIPPED -- {exc}")
            continue
        eff = efficiency_by_model.get(model_name, {})
        rows.append({
            "model": label, "model_name": model_name, "n_seeds": len(dsc_vals),
            "dsc_mean": float(np.mean(dsc_vals)), "dsc_std": float(np.std(dsc_vals, ddof=1)) if len(dsc_vals) > 1 else 0.0,
            "miou_mean": float(np.mean(miou_vals)), "miou_std": float(np.std(miou_vals, ddof=1)) if len(miou_vals) > 1 else 0.0,
            "params_M": eff.get("params_M"), "gflops_total": eff.get("gflops_total"),
            "fps_b1": eff.get("fps_b1"), "fps_b8": eff.get("fps_b8"),
            "cite_only": False, "footnote": "",
        })

    cite_rows, cite_notes = load_cite_only_rows(cite_only_path)
    notes.extend(cite_notes)
    for entry in cite_rows:
        if entry.get("dataset") != dataset:
            continue
        rows.append({
            "model": entry.get("model_name"), "model_name": entry.get("model_name"), "n_seeds": None,
            "dsc_mean": entry.get("dsc"), "dsc_std": None, "miou_mean": entry.get("miou"), "miou_std": None,
            "params_M": entry.get("params_M"), "gflops_total": entry.get("gflops"),
            "fps_b1": None, "fps_b8": None,
            "cite_only": True, "footnote": CITE_ONLY_FOOTNOTE,
        })

    return pd.DataFrame(rows), notes


# ---- ablation table (section 10) --------------------------------------------


def _measure_ablation_efficiency(config: RunConfig, efficiency_by_model: Dict[str, Dict[str, Any]], input_size: int = 256) -> Dict[str, Any]:
    """Params/GFLOPs per ABLATION ROW's own model_cfg -- results/efficiency.csv
    only has ONE row per canonical model_name (6 total), not per ablation
    permutation (e.g. kernel_size=1 vs 3, or a placement with zero LC-VSS
    stages have genuinely different params/GFLOPs from the hero config), so
    it cannot answer this table. Every ablation row except baseline_pvm is
    lc_ss2d/glgf (CPU-buildable, structural measurement -- device/scan_impl-
    independent, see lcmunet/efficiency.py's module docstring) -- measured
    FRESH, locally, here. baseline_pvm (ultralight_baseline) is GPU-gated and
    falls back to results/efficiency.csv's real Colab-measured row if present.
    """
    if config.model_name == "ultralight_baseline":
        eff = efficiency_by_model.get("ultralight_baseline")
        if eff is None:
            return {"params_M": None, "gflops_total": None, "efficiency_source": "N/A (GPU-gated; run notebooks/08_efficiency.ipynb first)"}
        return {"params_M": eff.get("params_M"), "gflops_total": eff.get("gflops_total"), "efficiency_source": "results/efficiency.csv (Colab-measured)"}

    import torch

    from lcmunet import efficiency as eff_mod
    from lcmunet.engine import build_model

    model = build_model(config)
    device = torch.device("cpu")
    params_M = eff_mod.measure_params_millions(model)
    module_g = eff_mod.measure_module_level_gflops(model, input_size=input_size, batch=1, device=device)
    scan_g = eff_mod.measure_scan_gflops(model, input_size=input_size, batch=1, device=device)
    return {
        "params_M": params_M,
        "gflops_total": module_g["gflops_module_thop"] + scan_g["gflops_scan_supplementary"],
        "efficiency_source": "measured fresh, locally, CPU (structural: device/scan_impl-independent)",
    }


def ablation_table(paths, input_size: int = 256) -> Tuple[pd.DataFrame, List[str]]:
    """A/C/D/Desc + Placement rows: DSC/mIoU/HD95 (Phase-1, single seed=42)
    + Params/GFLOPs (see _measure_ablation_efficiency)."""
    scan_impl, _source = em.resolve_scan_impl(paths)
    phase1_rows = dict(em.build_phase1_kvasir(scan_impl))
    efficiency_by_model = _load_efficiency_rows(paths)

    rows: List[Dict[str, Any]] = []
    notes: List[str] = []
    for role, label in ABLATION_ROWS:
        config = phase1_rows[role]
        try:
            dsc = _row_metric(paths.results, config, "dsc")
            miou = _row_metric(paths.results, config, "miou")
            hd95 = _row_metric(paths.results, config, "hd95")
        except RuntimeError as exc:
            notes.append(f"{label}: SKIPPED -- {exc}")
            continue
        eff = _measure_ablation_efficiency(config, efficiency_by_model, input_size=input_size)
        rows.append({
            "row": label, "role": role, "model_name": config.model_name, "config_id": config.config_id,
            "dsc": dsc, "miou": miou, "hd95": hd95,
            "params_M": eff["params_M"], "gflops_total": eff["gflops_total"], "efficiency_source": eff["efficiency_source"],
        })
    return pd.DataFrame(rows), notes


# ---- statistics table (section 8) -------------------------------------------


def _flatten_comparisons(dataset: str, report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flattens lcmunet.phase2_report.dataset_headline_report / _isic_
    generalisation_report's "models"+"comparisons" dicts into one row per
    paired comparison (hero vs baseline / hero vs GLGF / hero vs best
    competitor -- whichever are present for this report)."""
    models = report["models"]
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []
    if "hero_vs_baseline" in report["comparisons"]:
        pairs.append((report["comparisons"]["hero_vs_baseline"], models["hero"], models["baseline_pvm"]))
    if "hero_vs_glgf" in report["comparisons"]:
        pairs.append((report["comparisons"]["hero_vs_glgf"], models["hero"], models["glgf"]))
    if report.get("hero_vs_best_competitor"):
        pairs.append((report["hero_vs_best_competitor"], models["hero"], report["best_competitor"]["summary"]))

    out = []
    for comp, a, b in pairs:
        out.append({
            "dataset": dataset, "comparison": f"{comp['label_a']} vs {comp['label_b']}",
            "mean_dsc_a": a["mean"], "ci_low_a": a["ci_low"], "ci_high_a": a["ci_high"],
            "mean_dsc_b": b["mean"], "ci_low_b": b["ci_low"], "ci_high_b": b["ci_high"],
            "mean_diff": comp["mean_diff"], "wilcoxon_p": comp["wilcoxon_p"],
            "cohens_d": comp["cohens_d"], "cliffs_delta": comp["cliffs_delta"], "n_images": comp["n_images"],
        })
    return out


def statistics_table(
    paths,
    hero_descriptor_type: str = "contrast",
    headline_datasets: Tuple[str, ...] = ("kvasir_seg", "cvc_clinicdb"),
    isic_datasets: Tuple[str, ...] = ("isic2017", "isic2018"),
) -> Tuple[pd.DataFrame, List[str]]:
    """95% CI (per model), Wilcoxon p, effect size -- reuses
    lcmunet.phase2_report's already-tested paired-comparison functions
    directly (never re-derives the statistics)."""
    rows: List[Dict[str, Any]] = []
    notes: List[str] = []

    for dataset in headline_datasets:
        try:
            report = pr.dataset_headline_report(
                paths, dataset, hero_descriptor_type, list(em.HEADLINE_SEEDS),
                comparators_required=(dataset == "kvasir_seg"),
            )
        except RuntimeError as exc:
            notes.append(f"{dataset} (headline): SKIPPED -- {exc}")
            continue
        if not report.get("in_scope", True):
            notes.append(report["note"])
            continue
        rows.extend(_flatten_comparisons(dataset, report))

    for dataset in isic_datasets:
        try:
            report = pr.isic_generalisation_report(paths, dataset, hero_descriptor_type, list(em.OTHER_SEEDS))
        except RuntimeError as exc:
            notes.append(f"{dataset}: SKIPPED -- {exc}")
            continue
        if not report.get("in_scope", True):
            notes.append(report["note"])
            continue
        rows.extend(_flatten_comparisons(dataset, report))

    return pd.DataFrame(rows), notes


# ---- cross-dataset table (section 7) ----------------------------------------


def cross_dataset_table(paths) -> pd.DataFrame:
    """Directly reuses lcmunet.phase2_report.cross_dataset_report -- never
    empty-vs-missing ambiguous: an empty results/cross_dataset_results.csv
    yields an empty (but correctly headered) DataFrame, not an error."""
    return pd.DataFrame(pr.cross_dataset_report(paths))


# ---- rendering ---------------------------------------------------------------


def _df_to_markdown(df: pd.DataFrame, float_cols: Optional[List[str]] = None) -> str:
    if df.empty:
        return "*(no rows -- see notes)*"
    float_cols = float_cols or []
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join([":--"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c in float_cols and isinstance(v, (int, float, np.floating)) and not pd.isna(v):
                cells.append(f"{float(v):.4f}")
            else:
                cells.append("" if pd.isna(v) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_table(df: pd.DataFrame, out_dir: Path, name: str, caption: str, float_cols: Optional[List[str]] = None) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    md_path = out_dir / f"{name}.md"
    tex_path = out_dir / f"{name}.tex"

    df.to_csv(csv_path, index=False)
    md_path.write_text(f"## {caption}\n\n{_df_to_markdown(df, float_cols)}\n", encoding="utf-8")
    if df.empty:
        tex_path.write_text(f"% {caption}: no rows available -- see the accompanying .md notes\n", encoding="utf-8")
    else:
        tex = df.to_latex(index=False, caption=caption, label=f"tab:{name}", na_rep="--", float_format="%.4f")
        tex_path.write_text(tex, encoding="utf-8")
    return {"csv": csv_path, "md": md_path, "tex": tex_path}


def _write_readme(out_dir: Path, result: Dict[str, Any]) -> Path:
    lines = ["# Paper tables (methodology sections 8, 9, 15, 16)", ""]
    for name in ("main_comparison", "ablation", "statistics", "cross_dataset"):
        lines.append(f"## {name}")
        if name in result["errors"]:
            lines.append(f"- **NOT PRODUCED**: {result['errors'][name]}")
        else:
            paths_dict = result["tables"][name]
            lines.append(f"- csv: `{paths_dict['csv']}`")
            lines.append(f"- md: `{paths_dict['md']}`")
            lines.append(f"- tex: `{paths_dict['tex']}`")
            for note in result["notes"].get(name, []):
                lines.append(f"- note: {note}")
        lines.append("")
    path = out_dir / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---- orchestration -----------------------------------------------------------


def generate_paper_tables(
    paths,
    dataset: str = "kvasir_seg",
    hero_descriptor_type: str = "contrast",
    cite_only_path: Optional[Path] = None,
    ablation_input_size: int = 256,
) -> Dict[str, Any]:
    """Produces every table it can into results/paper_tables/; a missing
    prerequisite for one table does not block the others (see module
    docstring). Returns {"tables": {name: {"csv","md","tex"}}, "notes":
    {name: [str,...]}, "errors": {name: str}}.
    """
    out_dir = Path(paths.results) / "paper_tables"
    result: Dict[str, Any] = {"tables": {}, "notes": {}, "errors": {}}

    try:
        main_df, main_notes = main_comparison_table(paths, dataset=dataset, hero_descriptor_type=hero_descriptor_type, cite_only_path=cite_only_path)
        result["tables"]["main_comparison"] = _write_table(
            main_df, out_dir, "main_comparison", f"Main comparison table ({dataset})",
            float_cols=["dsc_mean", "dsc_std", "miou_mean", "miou_std", "params_M", "gflops_total", "fps_b1", "fps_b8"],
        )
        result["notes"]["main_comparison"] = main_notes
    except Exception as exc:  # noqa: BLE001 -- orchestration boundary, see module docstring
        result["errors"]["main_comparison"] = repr(exc)

    try:
        abl_df, abl_notes = ablation_table(paths, input_size=ablation_input_size)
        result["tables"]["ablation"] = _write_table(
            abl_df, out_dir, "ablation", "Ablation table (Kvasir-SEG, Phase-1, sections 10.1/10.2)",
            float_cols=["dsc", "miou", "hd95", "params_M", "gflops_total"],
        )
        result["notes"]["ablation"] = abl_notes
    except Exception as exc:  # noqa: BLE001
        result["errors"]["ablation"] = repr(exc)

    try:
        stats_df, stats_notes = statistics_table(paths, hero_descriptor_type=hero_descriptor_type)
        result["tables"]["statistics"] = _write_table(
            stats_df, out_dir, "statistics", "Statistics table (95% CI, Wilcoxon p, effect size -- section 8)",
            float_cols=["mean_dsc_a", "ci_low_a", "ci_high_a", "mean_dsc_b", "ci_low_b", "ci_high_b", "mean_diff", "wilcoxon_p", "cohens_d", "cliffs_delta"],
        )
        result["notes"]["statistics"] = stats_notes
    except Exception as exc:  # noqa: BLE001
        result["errors"]["statistics"] = repr(exc)

    try:
        cross_df = cross_dataset_table(paths)
        result["tables"]["cross_dataset"] = _write_table(
            cross_df, out_dir, "cross_dataset", "Cross-dataset generalisation table (section 7)",
            float_cols=["mean", "std", "ci_low", "ci_high"],
        )
        result["notes"]["cross_dataset"] = [] if len(cross_df) else ["results/cross_dataset_results.csv has no rows yet -- run lcmunet.cross_dataset_eval.run_cross_dataset_suite first."]
    except Exception as exc:  # noqa: BLE001
        result["errors"]["cross_dataset"] = repr(exc)

    _write_readme(out_dir, result)
    return result


if __name__ == "__main__":
    import argparse

    from lcmunet.paths import get_paths

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="kvasir_seg", choices=["kvasir_seg", "cvc_clinicdb"])
    parser.add_argument("--hero-descriptor", default="contrast", choices=["contrast", "plain"])
    parser.add_argument("--cite-only-path", default=None)
    args = parser.parse_args()

    paths = get_paths()
    result = generate_paper_tables(
        paths, dataset=args.dataset, hero_descriptor_type=args.hero_descriptor,
        cite_only_path=args.cite_only_path,
    )
    for name in ("main_comparison", "ablation", "statistics", "cross_dataset"):
        if name in result["errors"]:
            print(f"{name}: NOT PRODUCED -- {result['errors'][name]}")
        else:
            print(f"{name}: {result['tables'][name]['csv']}")
        for note in result["notes"].get(name, []):
            print(f"  note: {note}")
    print(f"\nSee {Path(paths.results) / 'paper_tables' / 'README.md'} for the full summary.")
