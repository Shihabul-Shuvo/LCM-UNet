import re

import pytest

from lcmunet import experiment_matrix as em
from lcmunet import gate2_report as gr


def _rows(baseline=0.700, glgf=0.720, hero=0.730, c_control=0.715, d_input=0.712, desc_plain=0.718):
    """A fully synthetic rows dict -- exercises evaluate_gate2_rules/decide
    without needing any real training. dsc values chosen to (by default)
    pass every section 13 rule."""
    configs = dict(em.build_phase1_kvasir(scan_impl="ref"))
    label_to_config = {label: configs[role] for label, role in gr.REQUIRED_ROLES.items()}
    values = {
        "baseline_pvm": baseline,
        "glgf": glgf,
        "hero": hero,
        "c_control": c_control,
        "d_inject_input": d_input,
        "desc_plain": desc_plain,
    }
    return {
        label: {"config": label_to_config[label], "dsc": v, "miou": v * 0.9, "hd95": 20.0}
        for label, v in values.items()
    }


# ---- evaluate_gate2_rules -------------------------------------------------


def test_evaluate_gate2_rules_all_pass():
    rules = gr.evaluate_gate2_rules(_rows())
    assert rules["main_pass"] and rules["A_pass"] and rules["C_pass"] and rules["D_pass"]
    assert rules["desc_winner"] == "contrast"


def test_evaluate_gate2_rules_main_threshold_boundary():
    # hero - baseline exactly at the 0.004 heuristic threshold -> PASS (>=)
    rows = _rows(baseline=0.700, hero=0.704)
    rules = gr.evaluate_gate2_rules(rows)
    assert rules["main_pass"] is True

    rows_fail = _rows(baseline=0.700, hero=0.7039)
    assert gr.evaluate_gate2_rules(rows_fail)["main_pass"] is False


def test_evaluate_gate2_rules_desc_winner_flips_to_plain():
    rows = _rows(hero=0.720, desc_plain=0.730)  # plain descriptor scored higher than the hero's contrast
    rules = gr.evaluate_gate2_rules(rows)
    assert rules["desc_winner"] == "plain"


# ---- decide(): every branch of the section 13 pivot table -----------------


def test_decide_proceed_when_everything_passes():
    rules = gr.evaluate_gate2_rules(_rows())
    decision, reason = gr.decide(rules, delta_non_constant=True)
    assert decision == "PROCEED"
    assert "multi-seed" in reason


def test_decide_pivot_main_threshold_fails():
    rules = gr.evaluate_gate2_rules(_rows(hero=0.701, baseline=0.700))  # +0.1pt, below 0.4pt heuristic
    decision, reason = gr.decide(rules, delta_non_constant=True)
    assert decision == "PIVOT"
    assert "Fallback 1" in reason
    assert "hook on dts" in reason


def test_decide_pivot_ablation_a_fails():
    # main passes (hero far above baseline) but hero ~= glgf (A fails)
    rules = gr.evaluate_gate2_rules(_rows(baseline=0.700, hero=0.720, glgf=0.719))
    decision, reason = gr.decide(rules, delta_non_constant=True)
    assert decision == "PIVOT"
    assert "retreat to the GLGF" in reason
    assert "increase seeds to 10" in reason


def test_decide_pivot_ablation_c_fails():
    # main + A pass, but hero does not beat the 1x1 control
    rules = gr.evaluate_gate2_rules(_rows(baseline=0.700, hero=0.720, glgf=0.710, c_control=0.725))
    decision, reason = gr.decide(rules, delta_non_constant=True)
    assert decision == "PIVOT"
    assert "1x1 capacity-matched control" in reason
    assert "switch" in reason.lower()


def test_decide_pivot_ablation_d_fails():
    # main + A + C pass, but hero does not beat inject-to-input
    rules = gr.evaluate_gate2_rules(_rows(baseline=0.700, hero=0.720, glgf=0.710, c_control=0.712, d_input=0.725))
    decision, reason = gr.decide(rules, delta_non_constant=True)
    assert decision == "PIVOT"
    assert "locality injection" in reason


def test_decide_pivot_delta_constant_despite_all_dice_rules_passing():
    rules = gr.evaluate_gate2_rules(_rows())  # all Dice rules pass
    decision, reason = gr.decide(rules, delta_non_constant=False)
    assert decision == "PIVOT"
    assert "CONSTANT" in reason


# ---- load_phase1_rows: fail loud on missing rows ---------------------------


def test_load_phase1_rows_raises_clear_error_when_nothing_is_done(paths):
    with pytest.raises(RuntimeError, match="missing DONE results"):
        gr.load_phase1_rows(paths)


def test_load_phase1_rows_error_names_every_missing_role(paths):
    with pytest.raises(RuntimeError) as excinfo:
        gr.load_phase1_rows(paths)
    for role in gr.REQUIRED_ROLES.values():
        assert role in str(excinfo.value)


# ---- render_report_md: smoke test -----------------------------------------


def test_render_report_md_contains_decision_and_table():
    rows = _rows()
    rules = gr.evaluate_gate2_rules(rows)
    delta_report = {
        stage: {"mean_abs_diff": 0.01, "std_diff": 0.02, "max_abs_diff": 0.05, "non_constant": True}
        for stage in ("E4", "E5", "D5", "D4")
    }
    decision, reason = gr.decide(rules, delta_non_constant=True)
    md = gr.render_report_md(rows, rules, delta_report, decision, reason)

    assert "## DECISION: PROCEED" in md
    assert "Desc ablation winner: `contrast`" in md
    assert re.search(r"\|\s*E4\s*\|", md)
    assert "HEURISTIC" in md.upper()


# ---- phase1_config_ids ------------------------------------------------------


def test_phase1_config_ids_matches_build_phase1_kvasir(paths):
    ids = gr.phase1_config_ids(paths)
    expected = {config.config_id for _role, config in em.build_phase1_kvasir(scan_impl="ref")}
    assert ids == expected
    assert len(ids) == 10


# ---- require_gate2_proceed: the hard runtime gate Phase-2 must call ------


def test_require_gate2_proceed_raises_when_decision_is_pivot(monkeypatch):
    monkeypatch.setattr(gr, "load_phase1_rows", lambda paths: _rows(hero=0.701, baseline=0.700))  # main fails
    monkeypatch.setattr(gr, "compute_hero_delta_diff", lambda paths, hero_config, n_images=8: {
        s: {"non_constant": True} for s in ("E4", "E5", "D5", "D4")
    })

    with pytest.raises(RuntimeError, match="not PROCEED"):
        gr.require_gate2_proceed(paths=None)


def test_require_gate2_proceed_returns_rules_when_decision_is_proceed(monkeypatch):
    monkeypatch.setattr(gr, "load_phase1_rows", lambda paths: _rows())  # all-pass defaults
    monkeypatch.setattr(gr, "compute_hero_delta_diff", lambda paths, hero_config, n_images=8: {
        s: {"non_constant": True} for s in ("E4", "E5", "D5", "D4")
    })

    rules = gr.require_gate2_proceed(paths=None)
    assert rules["desc_winner"] == "contrast"
    assert rules["main_pass"] and rules["A_pass"] and rules["C_pass"] and rules["D_pass"]


# ---- end-to-end: real (cheap) training for every CPU-runnable role,      --
# ---- a synthetic stand-in for the GPU-gated ultralight_baseline row      --


def test_generate_gate2_report_end_to_end(make_kvasir_raw, monkeypatch):
    """Real run_one() training for every role that can actually run on this
    CPU dev machine (hero/c_control/d_inject_input/desc_plain/glgf --
    exactly like tests/test_glgf.py's sanity-training test), plus a
    synthetic results.csv row standing in for 'ultralight_baseline' (which
    requires mamba-ssm's CUDA build and cannot run here -- same rationale
    as tests/test_comparators.py testing its fail-loud behaviour instead of
    the real model). Verifies the FULL pipeline (rule evaluation, the real
    Delta-difference check on a real trained checkpoint, and markdown
    rendering) runs end-to-end without error.
    """
    from lcmunet.data.splits import build_kvasir_split
    from lcmunet.engine import run_one
    from lcmunet.results_store import upsert_result

    # Keep this test fast (real training, just tiny): config identity still
    # matches gr's own lookup because both call the same em.build_phase1_kvasir,
    # which reads these same monkeypatched module constants.
    monkeypatch.setattr(em, "EPOCHS", 1)
    monkeypatch.setattr(em, "INPUT_SIZE", 64)
    monkeypatch.setattr(em, "BATCH_SIZE", 4)

    paths = make_kvasir_raw(n=20)
    build_kvasir_split(paths)

    configs = dict(em.build_phase1_kvasir(scan_impl="ref"))
    cpu_runnable_roles = [
        "phase1_ablation_A_glgf",
        "phase1_ablation_A_lc_ss2d_hero",
        "phase1_ablation_C_k1_control",
        "phase1_ablation_D_inject_input",
        "phase1_desc_plain",
    ]
    for role in cpu_runnable_roles:
        run_one(configs[role], paths=paths, num_workers=0)

    baseline_config = configs["phase1_ablation_A_baseline_pvm"]
    upsert_result(
        paths.results,
        {
            "config_id": baseline_config.config_id,
            "model_name": baseline_config.model_name,
            "dataset": baseline_config.dataset,
            "seed": baseline_config.seed,
            "split_file": baseline_config.split_file,
            "dsc": 0.01,  # deliberately low/synthetic -- NOT a real measurement, see docstring
            "miou": 0.01,
            "sensitivity": 0.01,
            "specificity": 0.5,
            "accuracy": 0.5,
            "hd95": 50.0,
            "assd": 30.0,
            "scan_impl": "cuda",
            "notes": "SYNTHETIC stand-in for ultralight_baseline (GPU-gated, cannot run on this CPU dev machine) -- test fixture only, not a real result.",
        },
    )

    report_path = gr.generate_gate2_report(paths, n_delta_images=2)

    assert report_path.is_file()
    content = report_path.read_text(encoding="utf-8")
    assert "## DECISION:" in content
    assert "PROCEED" in content or "PIVOT" in content
    for stage in ("E4", "E5", "D5", "D4"):
        assert stage in content
