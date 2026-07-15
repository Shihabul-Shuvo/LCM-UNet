"""Mechanism analysis and visualisation (methodology section 11) -- exactly
the three deliverables section 11 asks for, no more:

  1. Delta-difference map: Delta(ours) - Delta(baseline), on the SAME real
     test images, for each LC-VSS stage (E4/E5/D5/D4). Essential -- rebuts
     "did you really change the scan?"
  2. Region-wise modulation statistics: mean +/- std of the injected term
     alpha*tanh(W_delta.m) for boundary/interior/background GT-mask regions,
     plus a significance test that boundary != background. Essential --
     rebuts "the modulation is constant and meaningless."
  3. Per-stage learned alpha, from the training-time alpha log
     (lcmunet/engine.py's _log_alphas).

"ours" is the REAL trained LC-SS2D hero checkpoint; "baseline" is the REAL
trained ultralight_baseline (vendored UltraLight_VM_UNet) checkpoint -- NOT
delta_diff.py's alpha=0 proxy (that is explicitly a quick single-model
sanity check, methodology section 13; this module is the full, later
mechanism-analysis deliverable section 13's own docstring already points
to). Both checkpoints come from Phase-1's fixed Ablation-A reference rows
(lcmunet.gate2_report.REQUIRED_ROLES["hero"]/["baseline_pvm"]) by default --
that row's descriptor_type is ALWAYS 'contrast' by construction (Phase-1's
own fixed Desc-ablation reference row never changes regardless of the Gate-2
Desc-ablation winner used later for Phase-2 hero rows), so this module does
not need Gate-2 to have run, let alone PROCEED -- it only needs Phase-1
training to have completed. This is a deliberate choice, not an oversight:
mechanism evidence is informative regardless of the Gate-2 decision (indeed,
if Gate-2 PIVOTs on a constant Delta-difference, this module's deeper
region-wise analysis is exactly the diagnostic that would explain why).

BASELINE CAPTURE, A REAL HOOK-TARGET SUBTLETY (get this wrong and item 1
silently produces a meaningless or empty result -- verified against the
actual mamba_ssm source, not assumed). The vendored PVMLayer's `self.mamba`
is `mamba_ssm.Mamba`, whose `forward()` by default takes a FUSED path
(`mamba_inner_fn`) that never materialises a separate `dt` tensor at all
(the fused kernel hides it -- see lcmunet/lc_vss.py's module docstring).
Forcing `use_fast_path=False` on every Mamba instance makes it take the
slow path instead, which mamba_ssm's own source confirms is line-for-line
identical to lcmunet/lc_vss.py's LCVSSMambaCore reimplementation:
`dt = rearrange(self.dt_proj.weight @ dt.t(), "d (b l) -> b d l", l=seqlen)`
then `y = selective_scan_fn(x, dt, A, B, C, ..., delta_bias=..., delta_softplus=True)`.
Crucially, `mamba_simple.py` imports this via
`from mamba_ssm.ops.selective_scan_interface import selective_scan_fn`, a
`from X import Y` that binds a SEPARATE name into mamba_simple's own module
namespace -- so the hook must patch `mamba_ssm.modules.mamba_simple.
selective_scan_fn` (the name Mamba.forward() actually resolves), not
`mamba_ssm.ops.selective_scan_interface.selective_scan_fn` (where the
function is merely DEFINED). This is the same bug class found and fixed in
lcmunet/efficiency.py's `_capture_scan_shapes` while building this module --
see that module's docstring.

Because captured `dt` has the SAME (B, d_inner, L) shape and the SAME
pre-softplus/pre-delta_bias semantics on both sides (mamba_ssm's slow path
and lcmunet/lc_vss.py's reimplementation), one shared un-flatten/softplus
helper (_dts_calls_to_2d_map) applies to both hero and baseline captures.

GROUP-VS-DIRECTION AVERAGING (flagged, not silently picked -- same call
lcmunet/delta_diff.py and lcmunet/lc_vss.py's ARCHITECTURE NOTE already
made). Section 11 item 1 says "average over scan directions or show one
representative direction". This backbone's PVMLayer has no true
multi-direction SS2D cross-scan -- one flatten order, 4 CHANNEL-GROUP
copies of one shared Mamba/core (see lcmunet/lc_vss.py's ARCHITECTURE
NOTE). "Average over the four directions" is read here as "average over
the four PVM channel groups" -- the axis this backbone actually has four
of -- consistent with how delta_diff.py already resolved the identical
ambiguity for the Gate-2 sanity check.

REGION DEFINITION (flagged). boundary/interior/background are computed AT
EACH STAGE'S OWN TOKEN RESOLUTION -- confirmed by direct capture, NOT by
assuming encoder/decoder symmetry: at the default input_size=256, E4=32x32,
E5=16x16, D5=8x8, D4=8x8 (D5 and D4 share the coarser resolution -- the
decoder's bilinear upsample happens AFTER each PVM-family stage's own
forward, not before, so D4's stage input is D5's OUTPUT resolution, not a
mirror of E4's). Not at the GT mask's native resolution: the GT mask is nearest-neighbour
downsampled to the stage's (H, W) grid first, then partitioned via one
iteration of 3x3 binary dilation/erosion at THAT resolution -- boundary =
dilated XOR eroded, interior = eroded, background = NOT dilated (a clean
partition of every token into exactly one region). This is deliberately
matched to the local descriptor's own 3x3 receptive field (methodology
section 3.2) rather than an arbitrary full-resolution pixel-count band, and
avoids the ambiguity of mixing full-resolution morphology with a coarse
per-token feature map.

Every figure this module writes carries, verbatim, the methodology section
11 disclaimer.
"""

from __future__ import annotations

import csv as _csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.ndimage import binary_dilation, binary_erosion  # noqa: E402
from scipy.stats import kruskal  # noqa: E402

import lcmunet.delta_diff as dd  # noqa: E402

STAGES: Tuple[str, ...] = ("E4", "E5", "D5", "D4")

DISCLAIMER = (
    "These visualisations demonstrate that the proposed modulation is "
    "structure-localised, consistent with the hypothesis of "
    "neighbourhood-conditioned scan dynamics. They do not constitute "
    "clinical evidence and are not intended as such."
)

SIGNIFICANCE_ALPHA = 0.05


# ---- stage-module lookup (hero: LC_PVMLayer; baseline: vendored PVMLayer) --


def _baseline_stage_modules(model: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    """Same 4 positional slots (encoder4[0]/encoder5[0]/decoder1[0]/
    decoder2[0]) as lcmunet.delta_diff._stage_modules, but for the vendored
    PVMLayer (baseline model) -- both the vendored UltraLight_VM_UNet and
    lcmunet.lcm_unet.LCMUNet place their PVM-family module at index 0 of
    each of these four nn.Sequential slots (confirmed by direct reading of
    both source files)."""
    return {
        "E4": model.encoder4[0],
        "E5": model.encoder5[0],
        "D5": model.decoder1[0],
        "D4": model.decoder2[0],
    }


# ---- Delta capture: hero side (reuses lcmunet.delta_diff's helpers) --------


def _hero_stage_delta_dts(hero_model: torch.nn.Module, images: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
    """Real (no alpha-zeroing) per-stage `dts` calls for the trained hero
    model on `images` -- reuses lcmunet.delta_diff's private capture helpers
    (same project, same author, avoids re-deriving the identical hook
    logic) but WITHOUT the alpha=0 toggle delta_diff.delta_difference_report
    does; this just wants the real "ours" Delta."""
    stage_modules = dd._stage_modules(hero_model)
    if not stage_modules:
        raise ValueError(
            "mechanism_analysis: no LC-VSS stage on the hero model has an active "
            "alpha (nothing to visualise) -- pass the trained LC-SS2D hero "
            "checkpoint, not a baseline/GLGF model."
        )
    stage_inputs = dd._capture_stage_inputs(hero_model, images, stage_modules)
    return {stage: dd._capture_dts(stage_modules[stage], stage_inputs[stage]) for stage in STAGES}


# ---- Delta capture: baseline side (vendored mamba_ssm.Mamba) --------------


def _import_mamba_simple():
    """Indirection point so tests can substitute a fake mamba_ssm-shaped
    module tree without needing the real mamba_ssm installed. See this
    module's docstring for why mamba_ssm.modules.mamba_simple (not
    mamba_ssm.ops.selective_scan_interface) is the correct hook target."""
    import mamba_ssm.modules.mamba_simple as mamba_simple_module  # type: ignore

    return mamba_simple_module


def _capture_baseline_dts(stage_module: torch.nn.Module, stage_input: torch.Tensor) -> List[torch.Tensor]:
    """Replays one vendored PVMLayer stage (the trained baseline model's
    E4/E5/D5/D4), forcing mamba_ssm.Mamba's slow forward path so its `dt` is
    visible, then hooks mamba_ssm.modules.mamba_simple.selective_scan_fn to
    record the exact `dt` tensor for each of the 4 PVM-group calls -- see
    this module's docstring for why this specific target and this specific
    forcing are both required, and for the shape/semantics match with the
    hero side's captured `dts`.

    UNTESTABLE end-to-end on this project's local CPU dev machine (mamba_ssm
    is not installed -- GLOBAL RULES rule 5); this exact function only ever
    runs in Colab. The hook-target/use_fast_path LOGIC is unit-tested with a
    fake mamba_ssm-shaped module tree (tests/test_mechanism_analysis.py).
    """
    mamba = stage_module.mamba
    if not hasattr(mamba, "use_fast_path"):
        raise AttributeError(
            "stage_module.mamba has no 'use_fast_path' attribute -- mamba_ssm's "
            "Mamba API may have changed; this capture relies on forcing the slow "
            "forward path (see lcmunet/mechanism_analysis.py module docstring)."
        )

    mamba_simple_module = _import_mamba_simple()
    real_scan = mamba_simple_module.selective_scan_fn
    calls: List[torch.Tensor] = []

    def spy(u, dts, *args, **kwargs):
        calls.append(dts.detach().clone())
        return real_scan(u, dts, *args, **kwargs)

    original_use_fast_path = mamba.use_fast_path
    mamba_simple_module.selective_scan_fn = spy
    mamba.use_fast_path = False
    try:
        with torch.no_grad():
            stage_module(stage_input)
    finally:
        mamba_simple_module.selective_scan_fn = real_scan
        mamba.use_fast_path = original_use_fast_path

    return calls


# ---- per-stage capture, one seam per side (independently mockable) --------


def _hero_stage_capture(hero_model: torch.nn.Module, images: torch.Tensor) -> Dict[str, Dict[str, Any]]:
    """{stage: {"input_hw": (H, W), "dts": [one (B,d_inner,L) tensor per PVM group]}}."""
    stage_modules = dd._stage_modules(hero_model)
    if not stage_modules:
        raise ValueError(
            "hero model has no active LC-VSS alpha -- pass the trained LC-SS2D "
            "hero checkpoint, not a baseline/GLGF model."
        )
    stage_inputs = dd._capture_stage_inputs(hero_model, images, stage_modules)
    return {
        stage: {
            "input_hw": (int(stage_inputs[stage].shape[2]), int(stage_inputs[stage].shape[3])),
            "dts": dd._capture_dts(stage_modules[stage], stage_inputs[stage]),
        }
        for stage in STAGES
    }


def _baseline_stage_capture(baseline_model: torch.nn.Module, images: torch.Tensor) -> Dict[str, Dict[str, Any]]:
    """Same shape as _hero_stage_capture, for the vendored baseline model."""
    stage_modules = _baseline_stage_modules(baseline_model)
    stage_inputs = dd._capture_stage_inputs(baseline_model, images, stage_modules)
    return {
        stage: {
            "input_hw": (int(stage_inputs[stage].shape[2]), int(stage_inputs[stage].shape[3])),
            "dts": _capture_baseline_dts(stage_modules[stage], stage_inputs[stage]),
        }
        for stage in STAGES
    }


# ---- shared un-flatten / reduction math (fully unit-testable) -------------


def _dts_calls_to_2d_map(dts_calls: List[torch.Tensor], h: int, w: int) -> np.ndarray:
    """dts_calls: one (B, d_inner, L) tensor per PVM group (4 groups),
    PRE-softplus (delta_softplus=True is applied INSIDE selective_scan) --
    so this applies softplus first (methodology section 3.1's literal
    Delta_k = softplus(...)), un-flattens L -> (H, W) via a plain reshape
    (L was produced by a row-major merge of H, W with no permutation -- see
    lcmunet/lc_vss.py's pvm_flatten and the vendored PVMLayer's identical
    x.reshape(B, C, n_tokens); un-flattening is the exact inverse, no
    transpose needed), averages over d_inner (there is no single canonical
    per-token scalar in a d_inner-wide SSM step size, only a per-channel
    one), then averages over the 4 PVM groups (this module's docstring:
    "average over directions" reads as "average over groups" for this
    backbone). Returns (B, H, W) numpy, batch-first.
    """
    per_group = []
    for dts in dts_calls:
        b, d_inner, l = dts.shape
        if l != h * w:
            raise ValueError(f"un-flatten shape mismatch: L={l} != H*W={h*w}")
        delta = F.softplus(dts).reshape(b, d_inner, h, w)
        per_group.append(delta.mean(dim=1))  # (B, H, W)
    stacked = torch.stack(per_group, dim=0)  # (num_groups, B, H, W)
    return stacked.mean(dim=0).detach().cpu().numpy()


# ---- item 1: Delta-difference maps -----------------------------------------


def delta_difference_maps(hero_model: torch.nn.Module, baseline_model: torch.nn.Module, images: torch.Tensor) -> Dict[str, Dict[str, np.ndarray]]:
    """Delta(ours) - Delta(baseline) on the SAME real images, per LC-VSS
    stage. Returns {stage: {"hero": (B,H,W), "baseline": (B,H,W), "diff":
    (B,H,W)}}. Fails loud if either side observes zero scan calls for any
    stage (a silent zero-call capture would otherwise look like "no
    modulation" instead of "the hook did not fire"), or if hero/baseline
    stage-input resolutions disagree (they must, since both models share
    the identical conv/pooling prefix ahead of E4 -- methodology section 4).

    This function's own assembly/fail-loud logic is unit-tested by
    monkeypatching _hero_stage_capture/_baseline_stage_capture directly
    (tests/test_mechanism_analysis.py); those two functions' REAL capture
    mechanics are separately tested against a real LCMUNet (hero side) and a
    fake mamba_ssm module tree (baseline side, mirroring
    tests/test_efficiency.py's regression test for the identical hook-target
    bug class).
    """
    hero_model.eval()
    baseline_model.eval()

    hero_capture = _hero_stage_capture(hero_model, images)
    baseline_capture = _baseline_stage_capture(baseline_model, images)

    maps: Dict[str, Dict[str, np.ndarray]] = {}
    for stage in STAGES:
        h_hw = hero_capture[stage]["input_hw"]
        b_hw = baseline_capture[stage]["input_hw"]
        if h_hw != b_hw:
            raise ValueError(f"{stage}: hero/baseline stage-input resolution mismatch {h_hw} vs {b_hw}")
        h, w = h_hw

        hero_dts = hero_capture[stage]["dts"]
        baseline_dts = baseline_capture[stage]["dts"]
        if len(hero_dts) == 0 or len(baseline_dts) == 0:
            raise RuntimeError(
                f"{stage}: captured zero scan calls (hero={len(hero_dts)}, baseline="
                f"{len(baseline_dts)}) -- the hook did not fire; see this module's "
                "docstring on the mamba_ssm hook target/use_fast_path forcing."
            )

        hero_map = _dts_calls_to_2d_map(hero_dts, h, w)
        baseline_map = _dts_calls_to_2d_map(baseline_dts, h, w)
        maps[stage] = {"hero": hero_map, "baseline": baseline_map, "diff": hero_map - baseline_map}
    return maps


def save_delta_difference_figure(maps: Dict[str, Dict[str, np.ndarray]], out_path: Path, image_index: int = 0) -> Path:
    """3 rows (hero Delta, baseline Delta, hero-baseline diff) x 4 stage
    columns. The diff row is the essential content (methodology section 11:
    "do not show raw Delta heatmaps alone") -- hero/baseline rows are
    supplementary context, never presented alone."""
    fig, axes = plt.subplots(3, len(STAGES), figsize=(3.2 * len(STAGES), 9.5))
    row_labels = ("Delta (ours, hero)", "Delta (baseline)", "Delta(ours) - Delta(baseline)")

    for col, stage in enumerate(STAGES):
        hero_img = maps[stage]["hero"][image_index]
        baseline_img = maps[stage]["baseline"][image_index]
        diff_img = maps[stage]["diff"][image_index]
        diff_abs_max = float(np.abs(diff_img).max()) or 1.0

        im0 = axes[0, col].imshow(hero_img, cmap="viridis")
        axes[0, col].set_title(stage)
        plt.colorbar(im0, ax=axes[0, col], fraction=0.046)

        im1 = axes[1, col].imshow(baseline_img, cmap="viridis")
        plt.colorbar(im1, ax=axes[1, col], fraction=0.046)

        im2 = axes[2, col].imshow(diff_img, cmap="RdBu_r", vmin=-diff_abs_max, vmax=diff_abs_max)
        plt.colorbar(im2, ax=axes[2, col], fraction=0.046)

        for row in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=9)

    fig.suptitle(
        "Delta-difference map (methodology section 11 item 1): averaged over the 4 PVM\n"
        "channel groups (this backbone's single flatten order -- see module docstring), "
        f"test image #{image_index}",
        fontsize=10,
    )
    _add_disclaimer_caption(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---- item 2: region-wise modulation statistics -----------------------------


def _capture_modulation_calls(stage_module: torch.nn.Module) -> Tuple[List[torch.Tensor], Any]:
    """Forward-pre-hook on the LC-VSS stage's shared `core` submodule,
    capturing the EXACT (m_k, w_delta, alpha) LC_PVMLayer.forward passes
    into each of the 4 PVM-group LCVSSMambaCore.forward(...) calls, then
    recomputing alpha*tanh(w_delta(m_k)) from those captured tensors --
    mathematically IDENTICAL to what forward() computes internally
    (methodology section 3.1's injected term), since it is never returned
    as a separate value. Returns (calls, handle) -- caller must remove the
    handle."""
    core = stage_module.core
    calls: List[torch.Tensor] = []

    def hook(_module, args, kwargs):
        m_k = kwargs.get("m_k", args[1] if len(args) > 1 else None)
        w_delta = kwargs.get("w_delta")
        alpha = kwargs.get("alpha")
        if m_k is None or w_delta is None or alpha is None:
            return  # descriptor_type='none' stage -- nothing injected, nothing to capture
        calls.append((alpha * torch.tanh(w_delta(m_k))).detach().clone())

    handle = core.register_forward_pre_hook(hook, with_kwargs=True)
    return calls, handle


def _modulation_calls_to_2d_map(calls: List[torch.Tensor], h: int, w: int) -> np.ndarray:
    """calls: one (B, L, K) tensor per PVM group (K = dt_rank for
    inject_target='delta', d_inner for 'input') -- averages over the
    injected term's own channel-like axis (same reduction convention as
    _dts_calls_to_2d_map), then over the 4 groups, then un-flattens L ->
    (H, W) (no transpose: L is already the leading spatial axis here,
    unlike dts's (B, d_inner, L) layout)."""
    per_group = []
    for mod in calls:
        b, l, _k = mod.shape
        if l != h * w:
            raise ValueError(f"un-flatten shape mismatch: L={l} != H*W={h*w}")
        per_group.append(mod.mean(dim=2).reshape(b, h, w))
    stacked = torch.stack(per_group, dim=0)
    return stacked.mean(dim=0).detach().cpu().numpy()


def hero_modulation_maps(hero_model: torch.nn.Module, images: torch.Tensor) -> Dict[str, np.ndarray]:
    """Per-stage (B, H, W) map of the injected term alpha*tanh(W_delta.m)
    (methodology section 3.1) -- NOT the full Delta, the modulation ONLY."""
    stage_modules = dd._stage_modules(hero_model)
    if not stage_modules:
        raise ValueError(
            "hero_modulation_maps: hero model has no active LC-VSS alpha -- "
            "pass the trained LC-SS2D hero checkpoint."
        )
    stage_inputs = dd._capture_stage_inputs(hero_model, images, stage_modules)

    maps: Dict[str, np.ndarray] = {}
    for stage in STAGES:
        mod = stage_modules[stage]
        xin = stage_inputs[stage]
        h, w = xin.shape[2], xin.shape[3]
        calls, handle = _capture_modulation_calls(mod)
        try:
            with torch.no_grad():
                mod(xin)
        finally:
            handle.remove()
        if not calls:
            raise RuntimeError(f"{stage}: captured zero modulation-term calls -- hook did not fire.")
        maps[stage] = _modulation_calls_to_2d_map(calls, h, w)
    return maps


def _region_masks_at_resolution(gt_mask_full_res: np.ndarray, h: int, w: int) -> Dict[str, np.ndarray]:
    """See module docstring "REGION DEFINITION". Returns boolean (H, W)
    masks for "boundary"/"interior"/"background" that exactly partition
    every token into one region."""
    mask_t = torch.from_numpy(gt_mask_full_res.astype(np.float32))[None, None]
    mask_small = (F.interpolate(mask_t, size=(h, w), mode="nearest")[0, 0].numpy() >= 0.5)

    struct = np.ones((3, 3), dtype=bool)
    dilated = binary_dilation(mask_small, structure=struct, iterations=1)
    eroded = binary_erosion(mask_small, structure=struct, iterations=1, border_value=0)
    return {
        "boundary": dilated & ~eroded,
        "interior": eroded,
        "background": ~dilated,
    }


def _region_stat(values: np.ndarray) -> Dict[str, Any]:
    if values.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(values.mean()), "std": float(values.std()), "n": int(values.size)}


def region_wise_modulation_stats(hero_model: torch.nn.Module, images: torch.Tensor, gt_masks: torch.Tensor) -> Dict[str, Dict[str, Any]]:
    """images/gt_masks: SAME batch, SAME order (gt_masks: (B,1,H,W) or
    (B,H,W), float {0,1}). Per-stage {"boundary"/"interior"/"background":
    {mean,std,n}, "kruskal_h", "kruskal_p", "boundary_ne_background"} --
    the significance test compares POOLED boundary vs background pixel
    values (methodology section 11 item 2 offers Kruskal-Wallis or a paired
    t-test; Kruskal-Wallis is used here since boundary/background pixel
    COUNTS differ per image, so the two populations are not naturally
    one-to-one pairable -- a non-parametric rank test on pooled pixels
    avoids that mismatch and makes no distributional assumption on the
    per-pixel modulation values).
    """
    modulation_maps = hero_modulation_maps(hero_model, images)

    gt_np = gt_masks.detach().cpu().numpy()
    if gt_np.ndim == 4:
        gt_np = gt_np[:, 0]

    report: Dict[str, Dict[str, Any]] = {}
    for stage in STAGES:
        mod_map = modulation_maps[stage]  # (B, H, W)
        h, w = mod_map.shape[1], mod_map.shape[2]

        boundary_vals, interior_vals, background_vals = [], [], []
        for b in range(mod_map.shape[0]):
            regions = _region_masks_at_resolution(gt_np[b], h, w)
            boundary_vals.append(mod_map[b][regions["boundary"]])
            interior_vals.append(mod_map[b][regions["interior"]])
            background_vals.append(mod_map[b][regions["background"]])
        boundary_vals = np.concatenate(boundary_vals) if boundary_vals else np.array([])
        interior_vals = np.concatenate(interior_vals) if interior_vals else np.array([])
        background_vals = np.concatenate(background_vals) if background_vals else np.array([])

        if boundary_vals.size and background_vals.size:
            h_stat, p_val = kruskal(boundary_vals, background_vals)
            h_stat, p_val = float(h_stat), float(p_val)
        else:
            h_stat, p_val = float("nan"), float("nan")

        report[stage] = {
            "boundary": _region_stat(boundary_vals),
            "interior": _region_stat(interior_vals),
            "background": _region_stat(background_vals),
            "kruskal_h": h_stat,
            "kruskal_p": p_val,
            "boundary_ne_background": bool(p_val < SIGNIFICANCE_ALPHA) if not np.isnan(p_val) else False,
        }
    return report


def pooled_boundary_vs_background(region_stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """A single headline boundary != background statistic pooling every
    stage's pixels together, for the report's top-line summary (the
    per-stage breakdown lives in the full table/figure)."""
    # region_stats only carries summary stats, not raw pixel arrays, so this
    # combines per-stage Kruskal H statistics via Fisher's method-free simple
    # summary instead of re-pooling raw pixels (not retained to keep the
    # report deterministic from region_stats alone): report whichever stage
    # has the weakest (largest) p-value as the conservative headline number.
    worst_stage = max(STAGES, key=lambda s: region_stats[s]["kruskal_p"] if not np.isnan(region_stats[s]["kruskal_p"]) else 1.0)
    return {
        "conservative_worst_stage": worst_stage,
        "conservative_kruskal_p": region_stats[worst_stage]["kruskal_p"],
        "all_stages_significant": all(region_stats[s]["boundary_ne_background"] for s in STAGES),
    }


def _write_region_stats_csv(path: Path, region_stats: Dict[str, Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["stage", "region", "mean", "std", "n", "kruskal_h", "kruskal_p", "boundary_ne_background"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for stage in STAGES:
            s = region_stats[stage]
            for region in ("boundary", "interior", "background"):
                writer.writerow({
                    "stage": stage, "region": region,
                    "mean": s[region]["mean"], "std": s[region]["std"], "n": s[region]["n"],
                    "kruskal_h": s["kruskal_h"], "kruskal_p": s["kruskal_p"],
                    "boundary_ne_background": s["boundary_ne_background"],
                })
    return path


def save_region_wise_figure(region_stats: Dict[str, Dict[str, Any]], out_path: Path) -> Path:
    fig, axes = plt.subplots(1, len(STAGES), figsize=(3.6 * len(STAGES), 4.2), sharey=False)
    regions = ("boundary", "interior", "background")
    for col, stage in enumerate(STAGES):
        s = region_stats[stage]
        means = [s[r]["mean"] for r in regions]
        stds = [s[r]["std"] for r in regions]
        axes[col].bar(regions, means, yerr=stds, capsize=4, color=["#d62728", "#2ca02c", "#1f77b4"])
        sig = "boundary != background" if s["boundary_ne_background"] else "n.s."
        axes[col].set_title(f"{stage}\nKruskal p={s['kruskal_p']:.3g} ({sig})", fontsize=9)
        axes[col].tick_params(axis="x", labelrotation=30)
    axes[0].set_ylabel("alpha * tanh(W_delta . m)")
    fig.suptitle("Region-wise modulation statistics (methodology section 11 item 2)", fontsize=10)
    _add_disclaimer_caption(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---- item 3: per-stage alpha ------------------------------------------------


def load_alpha_log(paths, hero_config) -> pd.DataFrame:
    from lcmunet.engine import _alpha_csv_path

    path = _alpha_csv_path(paths, hero_config)
    if not path.is_file():
        raise RuntimeError(
            f"No alpha log at {path} -- the hero model must be trained (via "
            "lcmunet.engine.run_one) before per-stage alpha can be plotted; "
            "alpha is logged every lcmunet.engine.ALPHA_LOG_EVERY epochs."
        )
    return pd.read_csv(path)


def save_alpha_figure(alpha_df: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    stage_cols = [c for c in alpha_df.columns if c != "epoch"]
    for stage in stage_cols:
        ax.plot(alpha_df["epoch"], alpha_df[stage], marker="o", label=stage)
    ax.set_xlabel("epoch")
    ax.set_ylabel("alpha")
    ax.set_title("Per-stage learned alpha (methodology section 11 item 3)")
    ax.legend()
    _add_disclaimer_caption(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---- shared caption helper --------------------------------------------------


def _add_disclaimer_caption(fig) -> None:
    fig.text(0.5, -0.02, DISCLAIMER, wrap=True, ha="center", va="top", fontsize=7, style="italic")


# ---- orchestration -----------------------------------------------------------


def _load_checkpoint_state_dict(paths, config) -> dict:
    from lcmunet.engine import checkpoint_dir

    ckpt_path = checkpoint_dir(paths, config) / "best.pt"
    if not ckpt_path.is_file():
        raise RuntimeError(
            f"No trained checkpoint at {ckpt_path} for config_id={config.config_id} "
            f"({config.model_name} on {config.dataset}, seed={config.seed}). Phase-1 "
            "training must complete before mechanism analysis can run."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt["model"]


def _default_hero_baseline_configs(paths):
    from lcmunet import experiment_matrix as em
    from lcmunet.gate2_report import REQUIRED_ROLES

    scan_impl, _source = em.resolve_scan_impl(paths)
    rows = dict(em.build_phase1_kvasir(scan_impl))
    return rows[REQUIRED_ROLES["hero"]], rows[REQUIRED_ROLES["baseline_pvm"]]


def render_mechanism_report_md(
    hero_config,
    baseline_config,
    image_ids: List[str],
    delta_maps: Dict[str, Dict[str, np.ndarray]],
    region_stats: Dict[str, Dict[str, Any]],
    alpha_df: pd.DataFrame,
    figure_paths: Dict[str, Path],
) -> str:
    lines = ["# Mechanism analysis report (methodology section 11)", ""]
    lines.append(
        f"Hero (LC-SS2D): `{hero_config.model_name}`, descriptor_type="
        f"`{hero_config.model_cfg['descriptor_type']}`, dataset={hero_config.dataset}, "
        f"seed={hero_config.seed}, config_id=`{hero_config.config_id}`."
    )
    lines.append(
        f"Baseline (reproduced PVM): `{baseline_config.model_name}`, dataset="
        f"{baseline_config.dataset}, seed={baseline_config.seed}, config_id="
        f"`{baseline_config.config_id}`."
    )
    lines.append(f"Test images used ({len(image_ids)}): {', '.join(image_ids)}.")
    lines.append("")
    lines.append(f"> {DISCLAIMER}")
    lines.append("")

    lines.append("## 1. Delta-difference map (essential)")
    lines.append("")
    lines.append(f"Figure: `{figure_paths['delta_difference_figure']}`")
    lines.append("")
    lines.append("Averaged over the 4 PVM channel groups (this backbone's single flatten order -- see module docstring).")
    lines.append("")
    lines.append("| Stage | mean\\|diff\\| | max\\|diff\\| |")
    lines.append("|:--|--:|--:|")
    for stage in STAGES:
        diff = delta_maps[stage]["diff"]
        lines.append(f"| {stage} | {float(np.abs(diff).mean()):.6f} | {float(np.abs(diff).max()):.6f} |")
    lines.append("")

    lines.append("## 2. Region-wise modulation statistics (essential)")
    lines.append("")
    lines.append(f"Figure: `{figure_paths['region_wise_figure']}`; table: `{figure_paths['region_wise_csv']}`")
    lines.append("")
    lines.append(
        "Significance test: Kruskal-Wallis on pooled boundary vs background pixel "
        "values (see lcmunet/mechanism_analysis.py's module docstring for why "
        "Kruskal-Wallis over a paired t-test)."
    )
    lines.append("")
    lines.append("| Stage | Boundary mean+/-std (n) | Interior mean+/-std (n) | Background mean+/-std (n) | Kruskal p | boundary != background |")
    lines.append("|:--|:--|:--|:--|--:|:--:|")
    for stage in STAGES:
        s = region_stats[stage]
        b, i, bg = s["boundary"], s["interior"], s["background"]
        lines.append(
            f"| {stage} | {b['mean']:.5f}+/-{b['std']:.5f} (n={b['n']}) | "
            f"{i['mean']:.5f}+/-{i['std']:.5f} (n={i['n']}) | "
            f"{bg['mean']:.5f}+/-{bg['std']:.5f} (n={bg['n']}) | "
            f"{s['kruskal_p']:.4g} | {'YES' if s['boundary_ne_background'] else 'no'} |"
        )
    lines.append("")
    pooled = pooled_boundary_vs_background(region_stats)
    lines.append(
        f"**Headline statistic**: boundary != background is significant (p < {SIGNIFICANCE_ALPHA}) at "
        f"{'ALL' if pooled['all_stages_significant'] else 'NOT all'} 4 stages; the weakest stage is "
        f"{pooled['conservative_worst_stage']} (Kruskal p={pooled['conservative_kruskal_p']:.4g})."
    )
    lines.append("")

    lines.append("## 3. Per-stage learned alpha")
    lines.append("")
    lines.append(f"Figure: `{figure_paths['alpha_figure']}`")
    lines.append("")
    stage_cols = [c for c in alpha_df.columns if c != "epoch"]
    first_row, last_row = alpha_df.iloc[0], alpha_df.iloc[-1]
    lines.append(f"Logged epochs {int(alpha_df['epoch'].min())}-{int(alpha_df['epoch'].max())} (every `ALPHA_LOG_EVERY` epochs).")
    lines.append("")
    lines.append("| Stage | first logged alpha | final alpha |")
    lines.append("|:--|--:|--:|")
    for stage in stage_cols:
        lines.append(f"| {stage} | {first_row[stage]:.6f} | {last_row[stage]:.6f} |")
    lines.append("")

    lines.append(f"> {DISCLAIMER}")
    lines.append("")
    return "\n".join(lines)


def generate_mechanism_report(paths, hero_config=None, baseline_config=None, n_images: int = 8) -> Dict[str, Path]:
    """Loads the REAL trained hero + baseline checkpoints (Phase-1's fixed
    Ablation-A reference rows by default -- see module docstring), the SAME
    real test images + GT masks, and produces all three section-11
    deliverables. Returns a dict of every artifact path written.
    """
    from lcmunet.data.loaders import build_dataloaders
    from lcmunet.engine import build_model

    if hero_config is None or baseline_config is None:
        default_hero, default_baseline = _default_hero_baseline_configs(paths)
        hero_config = hero_config or default_hero
        baseline_config = baseline_config or default_baseline

    hero_model = build_model(hero_config)
    hero_model.load_state_dict(_load_checkpoint_state_dict(paths, hero_config))
    hero_model.eval()

    baseline_model = build_model(baseline_config)
    baseline_model.load_state_dict(_load_checkpoint_state_dict(paths, baseline_config))
    baseline_model.eval()

    _train_loader, _val_loader, test_loader = build_dataloaders(hero_config, paths, sanity=True, num_workers=0)
    images, masks, ids = next(iter(test_loader))
    images, masks, ids = images[:n_images], masks[:n_images], list(ids[:n_images])

    delta_maps = delta_difference_maps(hero_model, baseline_model, images)
    delta_fig_path = Path(paths.figures) / "delta_difference_map.png"
    save_delta_difference_figure(delta_maps, delta_fig_path)

    region_stats = region_wise_modulation_stats(hero_model, images, masks)
    region_csv_path = Path(paths.results) / "region_wise_modulation_stats.csv"
    _write_region_stats_csv(region_csv_path, region_stats)
    region_fig_path = Path(paths.figures) / "region_wise_modulation.png"
    save_region_wise_figure(region_stats, region_fig_path)

    alpha_df = load_alpha_log(paths, hero_config)
    alpha_fig_path = Path(paths.figures) / "per_stage_alpha.png"
    save_alpha_figure(alpha_df, alpha_fig_path)

    figure_paths = {
        "delta_difference_figure": delta_fig_path,
        "region_wise_figure": region_fig_path,
        "region_wise_csv": region_csv_path,
        "alpha_figure": alpha_fig_path,
    }
    md = render_mechanism_report_md(hero_config, baseline_config, ids, delta_maps, region_stats, alpha_df, figure_paths)
    md_path = Path(paths.results) / "mechanism_report.md"
    md_path.write_text(md, encoding="utf-8")

    return {**figure_paths, "mechanism_report_md": md_path}
