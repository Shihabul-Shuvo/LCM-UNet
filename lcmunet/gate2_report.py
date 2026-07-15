"""Gate-2 report (methodology section 13 Phase-1 decision rules).

Reads Phase-1 Kvasir-SEG results (produced by a real run of run_queue over
ONLY the Phase-1 config_ids -- see lcmunet/experiment_matrix.py's
"phase1_*" roles and this module's docstring for how those jobs are
selected) and renders results/gate2_report.md: each section 13 rule's
pass/fail, the Desc ablation winner, a Delta-difference sanity check
(lcmunet/delta_diff.py), and a single PROCEED/PIVOT recommendation
following section 13's exact pivot table.

All thresholds here are HEURISTICS -- section 13's own words ("Thresholds
are heuristics -- always pair with significance and effect size at the
final multi-seed stage") -- and every number in the emitted report is
explicitly labelled as such. This module NEVER auto-proceeds to Phase-2;
`decide()` only ever returns a recommendation string, and nothing in this
codebase acts on it without the user enqueuing Phase-2 themselves.

FAIL LOUD: if a required Phase-1 row is not yet DONE, load_phase1_rows
raises a clear RuntimeError naming exactly what's missing, rather than
producing a partial or silently-wrong report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from lcmunet import experiment_matrix as em
from lcmunet.delta_diff import delta_difference_report
from lcmunet.results_store import load_results

# Section 13 thresholds. Stated in Dice-score units on the [0, 1] scale
# ("~0.4%"/"~0.2%" read as 0.4/0.2 *percentage points* of Dice, i.e.
# 0.004/0.002 on this scale) -- HEURISTICS, confirm with multi-seed later.
MAIN_THRESHOLD = 0.004  # LC-SS2D beats baseline PVM by >= ~0.4% Dice
ABLATION_A_THRESHOLD = 0.002  # LC-SS2D beats GLGF by >= ~0.2% Dice
# Ablation C ("3x3 beats 1x1 control") and D ("DWConv->Delta beats
# DWConv->input") have no numeric threshold in section 13, just "beats" --
# read here as a strict positive Dice difference (flagged, not fabricated).

REQUIRED_ROLES = {
    "baseline_pvm": "phase1_ablation_A_baseline_pvm",
    "glgf": "phase1_ablation_A_glgf",
    "hero": "phase1_ablation_A_lc_ss2d_hero",
    "c_control": "phase1_ablation_C_k1_control",
    "d_inject_input": "phase1_ablation_D_inject_input",
    "desc_plain": "phase1_desc_plain",
}

LABELS = {
    "baseline_pvm": "Baseline PVM (reproduced)",
    "glgf": "GLGF late-fusion",
    "hero": "LC-SS2D (ours, hero: contrast, k=3, inject=delta, placement=hero)",
    "c_control": "Ablation C: 1x1 capacity-matched control",
    "d_inject_input": "Ablation D: DWConv -> input u",
    "desc_plain": "Desc ablation: plain DWConv3x3",
}


def phase1_config_ids(paths) -> set:
    """The set of config_ids that belong to ANY phase1_* role -- the
    job_filter driving `run_queue` should restrict to exactly this set
    (see notebooks/06_phase1_gate2.ipynb).
    """
    scan_impl, _source = em.resolve_scan_impl(paths)
    rows = em.build_phase1_kvasir(scan_impl)
    return {config.config_id for _role, config in rows}


def load_phase1_rows(paths) -> Dict[str, Dict[str, Any]]:
    """{label: {"config": RunConfig, **results.csv row}} for the 6 rows
    Gate-2 needs. Raises RuntimeError naming every missing row if Phase-1
    hasn't finished (or hasn't started) -- fail loud, not a partial report.
    """
    scan_impl, _source = em.resolve_scan_impl(paths)
    all_rows = dict(em.build_phase1_kvasir(scan_impl))
    configs = {label: all_rows[role] for label, role in REQUIRED_ROLES.items()}

    results_df = load_results(paths.results)

    rows: Dict[str, Dict[str, Any]] = {}
    missing = []
    for label, config in configs.items():
        match = results_df[(results_df["config_id"] == config.config_id) & (results_df["seed"] == config.seed)]
        if len(match) == 0:
            missing.append(f"{label} ({REQUIRED_ROLES[label]}, config_id={config.config_id})")
            continue
        row = match.iloc[-1].to_dict()
        rows[label] = {"config": config, **row}

    if missing:
        raise RuntimeError(
            "Gate-2 report cannot be generated -- missing DONE results.csv rows for: "
            + "; ".join(missing)
            + ". Run Phase-1 first: drive lcmunet.run_manifest.run_queue with "
            "job_filter=lambda job: job['config_id'] in lcmunet.gate2_report."
            "phase1_config_ids(paths) (see notebooks/06_phase1_gate2.ipynb)."
        )
    return rows


def evaluate_gate2_rules(rows: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    baseline_dsc = float(rows["baseline_pvm"]["dsc"])
    glgf_dsc = float(rows["glgf"]["dsc"])
    hero_dsc = float(rows["hero"]["dsc"])
    c_control_dsc = float(rows["c_control"]["dsc"])
    d_input_dsc = float(rows["d_inject_input"]["dsc"])
    desc_plain_dsc = float(rows["desc_plain"]["dsc"])

    main_diff = hero_dsc - baseline_dsc
    a_diff = hero_dsc - glgf_dsc
    c_diff = hero_dsc - c_control_dsc
    d_diff = hero_dsc - d_input_dsc
    desc_diff = hero_dsc - desc_plain_dsc  # hero IS the contrast-descriptor row

    return {
        "main_pass": main_diff >= MAIN_THRESHOLD,
        "main_diff": main_diff,
        "A_pass": a_diff >= ABLATION_A_THRESHOLD,
        "A_diff": a_diff,
        "C_pass": c_diff > 0.0,
        "C_diff": c_diff,
        "D_pass": d_diff > 0.0,
        "D_diff": d_diff,
        "desc_winner": "contrast" if desc_diff >= 0.0 else "plain",
        "desc_diff": desc_diff,
    }


def decide(rules: Dict[str, Any], delta_non_constant: bool) -> Tuple[str, str]:
    """Section 13's exact pivot table, checked in the same order the table
    presents its rows. Returns (decision, reason) with decision in
    {"PROCEED", "PIVOT"} -- never anything else, never auto-acted-on.
    """
    if not rules["main_pass"]:
        return "PIVOT", (
            f"LC-SS2D ~= baseline (hero - baseline = {rules['main_diff']*100:+.2f} pts Dice, "
            f"heuristic threshold >= +0.40 pts not met). Per section 13: verify the "
            "injection is actually working (hook on dts -- lcmunet/audit.py step-0 items "
            "1/2/5, lcmunet/delta_diff.py); check alpha convergence (per-stage alpha CSV); "
            "if still no gain -> Fallback 1 (retreat to the GLGF late-fusion paper)."
        )
    if not rules["A_pass"]:
        return "PIVOT", (
            f"Beats baseline but ~= GLGF (Ablation A: hero - glgf = {rules['A_diff']*100:+.2f} pts, "
            f"heuristic threshold >= +0.20 pts not met). Per section 13: try the contrast "
            f"descriptor if plain DWConv was used (current Desc winner: {rules['desc_winner']!r}); "
            "increase seeds to 10; if A still fails -> retreat to the GLGF late-fusion paper."
        )
    if not rules["C_pass"]:
        return "PIVOT", (
            f"Beats GLGF but does not beat the 1x1 capacity-matched control (Ablation C: "
            f"hero - control = {rules['C_diff']*100:+.2f} pts, not > 0). Per section 13: switch "
            f"descriptor to the contrast form (or vice versa; current Desc winner: "
            f"{rules['desc_winner']!r}); if C still fails -> retreat to GLGF."
        )
    if not rules["D_pass"]:
        return "PIVOT", (
            f"Beats baseline, A and C pass, but Ablation D does not (hero - inject_input = "
            f"{rules['D_diff']*100:+.2f} pts, not > 0). Per section 13: DWConv-to-input works -- "
            "pivot the claim to 'locality injection' rather than 'dynamics conditioning'; "
            "adjust framing accordingly."
        )
    if not delta_non_constant:
        return "PIVOT", (
            "All Dice-based rules (main, A, C, D) pass, but the Delta-difference sanity "
            "check found a CONSTANT (not spatially-varying) modulation on every LC-VSS "
            "stage. This is not one of section 13's named result patterns, but strongly "
            "suggests the mechanism is not genuinely neighbourhood-conditioned despite "
            "passing on Dice alone -- treat like the 'verify injection is working' pivot "
            "above before proceeding."
        )
    return "PROCEED", (
        "All section 13 Phase-1 decision rules pass on this single seed (main threshold, "
        "Ablation A, Ablation C, Ablation D, and the Delta-difference sanity check). This "
        "is a single-seed HEURISTIC screen (Gate 2) -- confirm with multi-seed statistics "
        "(5 seeds, paired Wilcoxon, 95% CI, effect size; methodology section 8) at the "
        "Phase-2 headline stage before treating this as a final result. Do NOT auto-enqueue "
        "Phase-2 -- the user must confirm."
    )


def _load_hero_checkpoint_state_dict(paths, hero_config) -> dict:
    from lcmunet.engine import checkpoint_dir

    ckpt_path = checkpoint_dir(paths, hero_config) / "best.pt"
    if not ckpt_path.is_file():
        raise RuntimeError(
            f"No trained checkpoint at {ckpt_path} for the hero config "
            f"(config_id={hero_config.config_id}). Phase-1 must be run (via run_queue) "
            "before the Delta-difference section of the Gate-2 report can be computed."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt["model"]


def compute_hero_delta_diff(paths, hero_config, n_images: int = 8) -> Dict[str, Dict[str, Any]]:
    """Loads the REAL trained hero checkpoint and REAL validation images
    (via the standard dataloader, sanity=True truncates to n_images), then
    runs lcmunet.delta_diff.delta_difference_report on them."""
    from lcmunet.data.loaders import build_dataloaders
    from lcmunet.engine import build_model

    model = build_model(hero_config)
    model.load_state_dict(_load_hero_checkpoint_state_dict(paths, hero_config))
    model.eval()

    _train_loader, val_loader, _test_loader = build_dataloaders(hero_config, paths, sanity=True, num_workers=0)
    images, _masks, _ids = next(iter(val_loader))
    images = images[:n_images]

    return delta_difference_report(model, images)


def render_report_md(
    rows: Dict[str, Dict[str, Any]],
    rules: Dict[str, Any],
    delta_report: Dict[str, Dict[str, Any]],
    decision: str,
    reason: str,
) -> str:
    seed = rows["hero"]["config"].seed
    lines = []
    lines.append("# Gate-2 report (methodology section 13, Phase-1 decision rules)")
    lines.append("")
    lines.append(
        f"**Single-seed (seed={seed}) HEURISTIC screen on Kvasir-SEG.** Every threshold "
        "below is a heuristic per section 13's own wording -- confirm with multi-seed "
        "statistics (5 seeds, Wilcoxon, 95% CI, effect size) at the Phase-2 headline stage."
    )
    lines.append("")
    lines.append(f"## DECISION: {decision}")
    lines.append("")
    lines.append(reason)
    lines.append("")
    lines.append(
        "**This report does not auto-enqueue Phase-2.** Phase-2 is many multi-hour runs "
        "on Colab free tier -- the user must review this report and confirm before "
        "`scripts/enqueue_all.py` Phase-2 jobs are run."
    )
    lines.append("")

    lines.append("## Phase-1 Ablation A / C / D / Desc rows")
    lines.append("")
    lines.append("| Row | model_name | DSC | mIoU | HD95 | config_id |")
    lines.append("|:--|:--|--:|--:|--:|:--|")
    for label in ("baseline_pvm", "glgf", "hero", "c_control", "d_inject_input", "desc_plain"):
        r = rows[label]
        lines.append(
            f"| {LABELS[label]} | {r['config'].model_name} | {float(r['dsc']):.4f} | "
            f"{float(r['miou']):.4f} | {float(r['hd95']):.2f} | `{r['config'].config_id}` |"
        )
    lines.append("")

    lines.append("## Section 13 rule evaluation (heuristics)")
    lines.append("")
    lines.append("| Rule | Diff (Dice pts) | Threshold | Result |")
    lines.append("|:--|--:|:--|:--:|")
    lines.append(f"| Main: LC-SS2D beats baseline PVM | {rules['main_diff']*100:+.2f} | >= +0.40 | {'PASS' if rules['main_pass'] else 'FAIL'} |")
    lines.append(f"| Ablation A: LC-SS2D beats GLGF | {rules['A_diff']*100:+.2f} | >= +0.20 | {'PASS' if rules['A_pass'] else 'FAIL'} |")
    lines.append(f"| Ablation C: 3x3 beats 1x1 control | {rules['C_diff']*100:+.2f} | > 0 (no explicit threshold in section 13) | {'PASS' if rules['C_pass'] else 'FAIL'} |")
    lines.append(f"| Ablation D: DWConv->Delta beats DWConv->input | {rules['D_diff']*100:+.2f} | > 0 (no explicit threshold in section 13) | {'PASS' if rules['D_pass'] else 'FAIL'} |")
    lines.append("")
    lines.append(
        f"**Desc ablation winner: `{rules['desc_winner']}`** (contrast - plain = "
        f"{rules['desc_diff']*100:+.2f} Dice pts). Per section 10.1: \"winner used in all "
        "later results\" -- if 'plain' wins, the paper's causal language and the hero "
        "config's default descriptor_type should be revisited before Phase-2."
    )
    lines.append("")

    lines.append("## Delta-difference sanity check (Delta(ours) - Delta(baseline), section 11.1)")
    lines.append("")
    lines.append(
        "Computed on the REAL trained hero checkpoint by zeroing each LC-VSS stage's "
        "learned alpha for one extra forward pass on the same real validation images, "
        "then restoring it (see lcmunet/delta_diff.py) -- no separate baseline model "
        "needed."
    )
    lines.append("")
    lines.append("| Stage | mean\\|diff\\| | std(diff) | max\\|diff\\| | non-constant |")
    lines.append("|:--|--:|--:|--:|:--:|")
    for stage in ("E4", "E5", "D5", "D4"):
        s = delta_report[stage]
        lines.append(
            f"| {stage} | {s['mean_abs_diff']:.6f} | {s['std_diff']:.6f} | "
            f"{s['max_abs_diff']:.6f} | {'yes' if s['non_constant'] else 'NO'} |"
        )
    lines.append("")

    return "\n".join(lines)


def compute_gate2_decision(paths, n_delta_images: int = 8) -> Dict[str, Any]:
    """Re-derives the full Gate-2 decision FRESH from current results.csv
    state -- never from a possibly-stale gate2_report.md file that might be
    sitting on disk from an earlier, incomplete run. Returns
    {"rows", "rules", "delta_report", "decision", "reason"}.
    """
    rows = load_phase1_rows(paths)
    rules = evaluate_gate2_rules(rows)

    hero_config = rows["hero"]["config"]
    delta_report = compute_hero_delta_diff(paths, hero_config, n_images=n_delta_images)
    delta_non_constant = all(stats["non_constant"] for stats in delta_report.values())

    decision, reason = decide(rules, delta_non_constant)
    return {"rows": rows, "rules": rules, "delta_report": delta_report, "decision": decision, "reason": reason}


def generate_gate2_report(paths, n_delta_images: int = 8) -> Path:
    result = compute_gate2_decision(paths, n_delta_images=n_delta_images)
    md = render_report_md(result["rows"], result["rules"], result["delta_report"], result["decision"], result["reason"])

    path = Path(paths.results) / "gate2_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    return path


def require_gate2_proceed(paths, n_delta_images: int = 8) -> Dict[str, Any]:
    """Re-derives the Gate-2 decision FRESH from current results.csv state
    and RAISES unless the decision is PROCEED. This is the hard runtime
    gate Phase-2 orchestration (notebooks/07_phase2.ipynb,
    lcmunet/phase2_report.py) MUST call before enqueuing or driving ANY
    Phase-2 job -- "do not auto-proceed to Phase-2" (this prompt's own
    instruction) is enforced here in code, not left as a human-trusted
    checkbox. Returns the full rules dict (including desc_winner --
    methodology section 10.1: "winner used in all later results") on
    success, so callers can thread the winning descriptor into Phase-2's
    hero configs (lcmunet.experiment_matrix.build_phase2_headline's
    hero_descriptor_type).
    """
    result = compute_gate2_decision(paths, n_delta_images=n_delta_images)
    if result["decision"] != "PROCEED":
        raise RuntimeError(
            f"Gate-2 decision is {result['decision']}, not PROCEED -- refusing to run "
            f"Phase-2. Reason: {result['reason']}"
        )
    return result["rules"]
