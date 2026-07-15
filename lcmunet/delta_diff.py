"""Delta-difference sanity check (methodology section 11.1; this prompt's
Gate-2 report, section 13: "Delta-difference maps are non-constant").

Delta(ours) - Delta(baseline) is computed on a REAL trained LC-SS2D model by
toggling each LC-VSS stage's learned alpha to exactly 0.0 for one extra
forward pass on the SAME real input, then restoring it immediately after.
At alpha=0, `Delta_k = softplus(s_Delta(u_k) + alpha*tanh(W_Delta*m_k))`
(methodology section 3.1) collapses to exactly `softplus(s_Delta(u_k))` --
the literal "as if the injection did not exist" baseline, using the SAME
trained s_Delta/W_Delta/core weights. This means no separate baseline
model, no mamba_ssm, and no GPU are required -- this runs on CPU using the
real trained checkpoint, hooking lcmunet.lc_vss.selective_scan the same way
lcmunet/audit.py does (Step-0 audit items 1/2).

This is a QUICK sanity check only (methodology section 13's own framing --
"quick Delta(ours)-Delta(baseline) via hook_dts"), not the full
boundary/interior/background regional statistics analysis of section 11.2
(a separate, later prompt/mechanism-analysis deliverable).
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

import lcmunet.lc_vss as lc_vss_module

# std of the captured dts difference across all elements; near-zero means a
# flat/constant (or entirely absent) modulation, not a genuine per-token,
# neighbourhood-conditioned change.
NON_CONSTANT_STD_THRESHOLD = 1e-6


def _stage_modules(model: nn.Module) -> Dict[str, nn.Module]:
    """LC-VSS stages that are actually injecting (alpha is not None) on
    `model`. Works for any model exposing the same E4/E5/D5/D4
    encoder4/encoder5/decoder1/decoder2 Sequential slots as
    lcmunet.lcm_unet.LCMUNet (the hero LC-SS2D architecture)."""
    candidates = {
        "E4": model.encoder4[0],
        "E5": model.encoder5[0],
        "D5": model.decoder1[0],
        "D4": model.decoder2[0],
    }
    return {name: mod for name, mod in candidates.items() if getattr(mod, "alpha", None) is not None}


def _capture_stage_inputs(model: nn.Module, x: torch.Tensor, stage_modules: Dict[str, nn.Module]) -> Dict[str, torch.Tensor]:
    """One real forward pass of the FULL model, capturing each LC-VSS
    stage's actual input activation via a forward-pre-hook -- so the
    per-stage replay below sees exactly what that stage saw in situ."""
    captured: Dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, inputs):
            captured[name] = inputs[0].detach().clone()

        return hook

    for name, mod in stage_modules.items():
        handles.append(mod.register_forward_pre_hook(make_hook(name)))
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()
    return captured


def _capture_dts(stage_module: nn.Module, stage_input: torch.Tensor) -> List[torch.Tensor]:
    """Replays just one stage on `stage_input`, hooking
    lcmunet.lc_vss.selective_scan to record the exact `dts` tensor passed to
    every scan call (one per PVM group -- see lcmunet/audit.py's identical
    spy pattern)."""
    real_scan = lc_vss_module.selective_scan
    calls: List[torch.Tensor] = []

    def spy(u, dts, *args, **kwargs):
        calls.append(dts.detach().clone())
        return real_scan(u, dts, *args, **kwargs)

    lc_vss_module.selective_scan = spy
    try:
        with torch.no_grad():
            stage_module(stage_input)
    finally:
        lc_vss_module.selective_scan = real_scan
    return calls


def delta_difference_report(model: nn.Module, x: torch.Tensor) -> Dict[str, Dict[str, Any]]:
    """Runs `model` once on `x` (real trained weights, real images) to
    capture each LC-VSS stage's real input, then for each stage with an
    active alpha, replays JUST that stage twice on the SAME captured input
    -- once normally ("ours"), once with alpha temporarily zeroed
    ("baseline") -- diffs the two `dts` sets, and reports per-stage stats.

    Raises if no stage on `model` has an active alpha (nothing to diff --
    e.g. this was called on a baseline/GLGF model by mistake instead of the
    trained LC-SS2D hero model).
    """
    model.eval()
    stage_modules = _stage_modules(model)
    if not stage_modules:
        raise ValueError(
            "delta_difference_report: no LC-VSS stage on this model has an "
            "active alpha (nothing to diff) -- pass the trained LC-SS2D hero "
            "model, not a baseline/GLGF model."
        )

    stage_inputs = _capture_stage_inputs(model, x, stage_modules)

    report: Dict[str, Dict[str, Any]] = {}
    for name, mod in stage_modules.items():
        xin = stage_inputs[name]
        dts_ours = _capture_dts(mod, xin)

        original_alpha = mod.alpha.data.clone()
        mod.alpha.data.zero_()
        try:
            dts_baseline = _capture_dts(mod, xin)
        finally:
            mod.alpha.data.copy_(original_alpha)

        if len(dts_ours) != len(dts_baseline):
            raise RuntimeError(f"{name}: scan call count mismatch ({len(dts_ours)} vs {len(dts_baseline)}) -- cannot diff")

        diffs = torch.cat([(o - b).flatten() for o, b in zip(dts_ours, dts_baseline)])
        report[name] = {
            "mean_abs_diff": float(diffs.abs().mean()),
            "std_diff": float(diffs.std()),
            "max_abs_diff": float(diffs.abs().max()),
            "non_constant": bool(diffs.std() > NON_CONSTANT_STD_THRESHOLD),
        }
    return report
