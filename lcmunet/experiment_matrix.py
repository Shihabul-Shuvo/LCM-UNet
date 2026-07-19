"""Full experiment matrix (methodology sections 9, 10, 13, 14) as RunConfig
objects, ready to enqueue into the run manifest (lcmunet/run_manifest.py).

No new modeling here -- every row uses an existing model_name
("ultralight_baseline" | "glgf" | "lc_ss2d" | "unet" | "malunet" | "egeunet")
with a RunConfig.model_cfg toggle combination that already exists
(lcmunet/config.py DEFAULT_MODEL_CFG, lcmunet/lc_vss.py resolve_placement_stages).

Identity vs. bookkeeping: a RunConfig's config_id is a sha256 hash of ALL of
its own fields (including run_name) -- it carries no phase/ablation label.
The SAME underlying experiment can legitimately appear in more than one
methodology table (e.g. the Kvasir hero row at seed=42 is simultaneously
Phase-1's Ablation A/C/D/Desc/Placement reference row AND Phase-2's headline
seed=42 run) -- run_name is therefore derived purely from the semantic
identity (model_name, dataset, seed, model_cfg), never from which table
asked for it, so two asks for "the same experiment" always produce the same
RunConfig and the same config_id. build_all() below merges those occurrences
into ONE physical job and records every table/role it plays under, so the
manifest never carries two jobs for one experiment and no compute is wasted
re-running it.

FLAGGED GAP (GLOBAL RULES rule 1 -- report, don't silently redesign):
cross-dataset generalisation (train Kvasir/test CVC and reverse; section 7,
section 14 week 5) is an EVALUATION step over an already-trained in-domain
checkpoint (lcmunet.data.loaders.build_cross_dataset_loader already exists
for exactly this), not a new training config -- lcmunet.engine.run_one only
trains and tests on the SAME config.dataset, so there is no RunConfig field
that expresses "train on A, test on B" today. Enqueuing a fabricated
RunConfig for this would silently train+test on one dataset while
mislabeling it as a cross-dataset row, which is worse than not enqueuing it.
NOT enqueued here; needs a small evaluate-only entry point (reusing the
Phase-2 hero/baseline checkpoints built below) in a future prompt.

"glgf" is implemented (lcmunet/glgf.py, GLGFUNet -- the late-fusion
baseline/ablation variant, methodology sections 1/3.5/10.1 Ablation A); its
rows are real, trainable jobs like every other model_name here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from lcmunet.config import DEFAULT_MODEL_CFG, RunConfig

# Seeds (methodology section 6): 5 for headline rows (baseline, GLGF,
# LC-SS2D on Kvasir-SEG/CVC-ClinicDB); 3 elsewhere (ISIC, comparators).
# Phase-1's single seed is intentionally the FIRST headline seed so the
# Phase-1 hero/baseline/GLGF rows on Kvasir double as headline seed 1 --
# see module docstring.
PHASE1_SEED = 42
HEADLINE_SEEDS: List[int] = [42, 43, 44, 45, 46]
OTHER_SEEDS: List[int] = [42, 43, 44]

# Methodology section 6: "Epochs | 250-300". One fixed value so every table
# is directly comparable (Fairness rule) rather than a per-row choice.
#
# TEMPORARILY 50 (was 250): a fast quick-look pass to see results sooner,
# not the paper's final epoch budget. Every RunConfig's config_id hashes
# `epochs` (lcmunet/config.py), so this produces a DIFFERENT set of
# config_ids/checkpoints than the 250-epoch runs already in
# results/manifest.json -- it does NOT resume or overwrite any in-progress
# 250-epoch checkpoint (that checkpoint is untouched on Drive and still
# resumable later by changing this back to 250). Before any real Phase-1/
# Phase-2 table, set this back to 250 (or whatever's chosen in [250, 300])
# so every compared row uses the same epoch budget.
EPOCHS = 50

# Methodology section 6: "Input | 256x256", "Batch | 8". Module-level (like
# EPOCHS above) so tests can monkeypatch a cheap end-to-end configuration
# without duplicating _row()'s construction logic -- production callers
# never override these, so real experiment configs are unaffected.
INPUT_SIZE = 256
BATCH_SIZE = 8

KVASIR_SPLIT = "splits/kvasir_seg.json"
CVC_SPLIT = "splits/cvc_clinicdb.json"
ISIC2017_SPLIT = "splits/isic2017.json"
ISIC2018_SPLIT = "splits/isic2018.json"

HERO_MODEL_CFG: Dict[str, Any] = dict(DEFAULT_MODEL_CFG)  # descriptor=contrast, k=3, inject=delta, placement=hero


def _model_cfg(**overrides: Any) -> Dict[str, Any]:
    cfg = dict(DEFAULT_MODEL_CFG)
    cfg.update(overrides)
    return cfg


def _run_name(model_name: str, dataset: str, seed: int, model_cfg: Dict[str, Any]) -> str:
    """Derived purely from semantic identity -- NEVER from which table/role
    asked for this config -- so the same experiment always hashes to the
    same config_id no matter how many methodology tables reference it.
    """
    if model_name != "lc_ss2d":
        return f"{model_name}_{dataset}_seed{seed}"
    mc = model_cfg
    return (
        f"lc_ss2d_{dataset}_{mc['placement']}_{mc['descriptor_type']}"
        f"_k{mc['kernel_size']}_{mc['inject_target']}"
        f"_e6{int(mc['use_e6'])}_seed{seed}"
    )


def _row(
    role: str,
    model_name: str,
    dataset: str,
    split_file: str,
    seed: int,
    scan_impl: str,
    model_cfg: Dict[str, Any] | None = None,
) -> Tuple[str, RunConfig]:
    cfg = model_cfg if model_cfg is not None else dict(DEFAULT_MODEL_CFG)
    config = RunConfig(
        run_name=_run_name(model_name, dataset, seed, cfg),
        model_name=model_name,
        dataset=dataset,
        seed=seed,
        split_file=split_file,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        input_size=INPUT_SIZE,
        scan_impl=scan_impl,
        model_cfg=cfg,
    )
    return role, config


# ---- Phase-1: Kvasir-SEG, 1 seed, cheap go/no-go (section 10.1/10.2, 13) ----


def build_phase1_kvasir(scan_impl: str) -> List[Tuple[str, RunConfig]]:
    rows: List[Tuple[str, RunConfig]] = []
    seed = PHASE1_SEED

    # Ablation A: baseline PVM (reproduced) -> GLGF late-fusion -> LC-SS2D (ours)
    rows.append(_row("phase1_ablation_A_baseline_pvm", "ultralight_baseline", "kvasir_seg", KVASIR_SPLIT, seed, scan_impl))
    rows.append(_row("phase1_ablation_A_glgf", "glgf", "kvasir_seg", KVASIR_SPLIT, seed, scan_impl))
    rows.append(_row("phase1_ablation_A_lc_ss2d_hero", "lc_ss2d", "kvasir_seg", KVASIR_SPLIT, seed, scan_impl, _model_cfg()))

    # Ablation C: 1x1 capacity-matched control (hero/3x3 is the row directly above)
    rows.append(_row("phase1_ablation_C_k1_control", "lc_ss2d", "kvasir_seg", KVASIR_SPLIT, seed, scan_impl, _model_cfg(kernel_size=1)))

    # Ablation D: DWConv -> input u (hero/DWConv->delta is the row above)
    rows.append(_row("phase1_ablation_D_inject_input", "lc_ss2d", "kvasir_seg", KVASIR_SPLIT, seed, scan_impl, _model_cfg(inject_target="input")))

    # Desc ablation: plain DWConv (hero/contrast is the row above)
    rows.append(_row("phase1_desc_plain", "lc_ss2d", "kvasir_seg", KVASIR_SPLIT, seed, scan_impl, _model_cfg(descriptor_type="plain")))

    # Placement ablation: P0 / E4E5 / hero (above) / +E6 / E4D4
    for tag, placement in (("P0", "P0"), ("E4E5", "E4E5"), ("plusE6", "+E6"), ("E4D4", "E4D4")):
        rows.append(_row(f"phase1_placement_{tag}", "lc_ss2d", "kvasir_seg", KVASIR_SPLIT, seed, scan_impl, _model_cfg(placement=placement)))

    return rows


# ---- Phase-2: headline, multi-seed (section 14 week 4-5) --------------------


def build_phase2_headline(scan_impl: str, hero_descriptor_type: str = "contrast") -> List[Tuple[str, RunConfig]]:
    """Hero + baseline + GLGF on Kvasir-SEG and CVC-ClinicDB, 5 seeds.

    hero_descriptor_type: the Desc ablation winner (methodology section
    10.1: "winner used in all later results"), determined by Gate-2 on
    Phase-1 results (see lcmunet/gate2_report.py). Defaults to 'contrast'
    (the methodology's a-priori default) so this function -- and
    scripts/enqueue_all.py, which calls it before Gate-2 has ever run --
    remains usable before Phase-1 completes. If the winner differs from
    'contrast', these hero rows get a DIFFERENT config_id than Phase-1's
    fixed hero reference row (phase1_ablation_A_lc_ss2d_hero, which is
    ALWAYS 'contrast' by construction -- it IS the Desc ablation's contrast
    arm and must never move) -- that is correct, not a bug: Phase-2's
    headline claim is defined only after Desc is decided.
    """
    rows: List[Tuple[str, RunConfig]] = []
    for dataset, split_file in (("kvasir_seg", KVASIR_SPLIT), ("cvc_clinicdb", CVC_SPLIT)):
        for seed in HEADLINE_SEEDS:
            rows.append(_row(f"phase2_headline_{dataset}_baseline_pvm_seed{seed}", "ultralight_baseline", dataset, split_file, seed, scan_impl))
            rows.append(_row(f"phase2_headline_{dataset}_glgf_seed{seed}", "glgf", dataset, split_file, seed, scan_impl))
            rows.append(_row(f"phase2_headline_{dataset}_lc_ss2d_hero_seed{seed}", "lc_ss2d", dataset, split_file, seed, scan_impl, _model_cfg(descriptor_type=hero_descriptor_type)))
    return rows


def build_phase2_isic(scan_impl: str, hero_descriptor_type: str = "contrast") -> List[Tuple[str, RunConfig]]:
    """Hero + baseline on ISIC2017 and ISIC2018, 3 seeds (no GLGF per spec).
    hero_descriptor_type: see build_phase2_headline."""
    rows: List[Tuple[str, RunConfig]] = []
    for dataset, split_file in (("isic2017", ISIC2017_SPLIT), ("isic2018", ISIC2018_SPLIT)):
        for seed in OTHER_SEEDS:
            rows.append(_row(f"phase2_{dataset}_baseline_pvm_seed{seed}", "ultralight_baseline", dataset, split_file, seed, scan_impl))
            rows.append(_row(f"phase2_{dataset}_lc_ss2d_hero_seed{seed}", "lc_ss2d", dataset, split_file, seed, scan_impl, _model_cfg(descriptor_type=hero_descriptor_type)))
    return rows


def build_phase2_comparators(scan_impl: str) -> List[Tuple[str, RunConfig]]:
    """Reproduced U-Net/EGE-UNet/MALUNet on Kvasir (required) and
    CVC-ClinicDB (marked optional_if_time in the role tag; still enqueued
    as PENDING per this prompt's instructions, just clearly labelled so the
    user can choose to skip them under time pressure), 3 seeds.
    """
    rows: List[Tuple[str, RunConfig]] = []
    datasets = (
        ("kvasir_seg", KVASIR_SPLIT, "phase2_comparators_kvasir_seg"),
        ("cvc_clinicdb", CVC_SPLIT, "phase2_comparators_optional_if_time_cvc_clinicdb"),
    )
    for dataset, split_file, role_prefix in datasets:
        for seed in OTHER_SEEDS:
            for model_name in ("unet", "malunet", "egeunet"):
                rows.append(_row(f"{role_prefix}_{model_name}_seed{seed}", model_name, dataset, split_file, seed, scan_impl))
    return rows


# ---- scan_impl resolution (GLOBAL RULES: record which selective-scan ran) --


def resolve_scan_impl(paths) -> Tuple[str, str]:
    """Returns (scan_impl, source). Prefers results/env.json, written by
    notebooks/01_env.ipynb on the real Colab target (GLOBAL RULES rule 5:
    the GPU/mamba-ssm build is verified there, not here). Falls back to this
    process's own live lcmunet.scan.SCAN_IMPL with an explicit provenance
    tag when no env.json exists yet (e.g. first run on a local CPU dev box,
    before Colab has ever run notebooks/01_env.ipynb).
    """
    env_json_path = Path(paths.results) / "env.json"
    if env_json_path.is_file():
        with open(env_json_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        if "scan_impl" in info:
            return info["scan_impl"], f"results/env.json (timestamp={info.get('timestamp')})"

    from lcmunet.scan import SCAN_IMPL

    return (
        SCAN_IMPL,
        "lcmunet.scan.SCAN_IMPL live fallback -- no results/env.json yet; "
        "run notebooks/01_env.ipynb in Colab first to record the real target "
        "value, then re-run this script.",
    )


# ---- assembly: merge by config_id, track every role a job plays ------------


def build_all(scan_impl: str, hero_descriptor_type: str = "contrast") -> Dict[str, Tuple[RunConfig, List[str]]]:
    """Returns {config_id: (RunConfig, [role, ...])}. A role is a
    human-readable tag for one methodology-table row; a single config_id can
    carry multiple roles when the same experiment is referenced by more than
    one table (see module docstring) -- that is the dedup mechanism, not a
    collision to guard against.

    hero_descriptor_type: see build_phase2_headline -- only affects Phase-2
    hero rows; Phase-1's ablation matrix (including its own fixed 'contrast'
    hero reference row) never changes.
    """
    all_rows: List[Tuple[str, RunConfig]] = (
        build_phase1_kvasir(scan_impl)
        + build_phase2_headline(scan_impl, hero_descriptor_type=hero_descriptor_type)
        + build_phase2_isic(scan_impl, hero_descriptor_type=hero_descriptor_type)
        + build_phase2_comparators(scan_impl)
    )

    merged: Dict[str, Tuple[RunConfig, List[str]]] = {}
    for role, config in all_rows:
        cid = config.config_id
        if cid not in merged:
            merged[cid] = (config, [])
        merged[cid][1].append(role)

    return merged


def phase2_config_ids(scan_impl: str, hero_descriptor_type: str = "contrast") -> set:
    """The set of config_ids belonging to ANY phase2_* role -- the job_filter
    driving Phase-2's run_queue should restrict to exactly this set (see
    notebooks/07_phase2.ipynb), mirroring lcmunet.gate2_report.phase1_config_ids.
    """
    rows = (
        build_phase2_headline(scan_impl, hero_descriptor_type=hero_descriptor_type)
        + build_phase2_isic(scan_impl, hero_descriptor_type=hero_descriptor_type)
        + build_phase2_comparators(scan_impl)
    )
    return {config.config_id for _role, config in rows}


def phase_of(role: str) -> str:
    return "phase1" if role.startswith("phase1") else "phase2"
