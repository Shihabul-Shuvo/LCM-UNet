"""Efficiency measurement (methodology sections 5.5, 5.6, 8, 9).

Params (M), GFLOPs (thop, 256x256), FPS (batch=1 and batch=8, with a stated
warm-up), and peak training GPU memory (batch=8) -- measured, never quoted
or estimated (GLOBAL RULES). FPS and peak GPU memory are fundamentally
GPU-only measurements (this module still runs on CPU for local testing of
the harness itself, but always labels a CPU FPS number as NOT the required
Colab measurement, and returns None -- not 0.0 -- for peak memory without a
GPU, since "no memory used" and "undefined without a GPU" are different
facts).

Params and GFLOPs are structural counts, not timing measurements -- they do
not depend on which device or selective-scan implementation ran them, so
(unlike FPS/memory) they are valid regardless of where this module runs.

THOP'S BLIND SPOT (important, not a minor footnote): thop counts FLOPs by
hooking `nn.Module.forward()` calls. The selective-scan recurrence
(lcmunet/scan.py's `_selective_scan_ref`, or mamba_ssm's CUDA kernel) is
invoked as a plain function call from inside LCVSSMambaCore.forward() /
mamba_ssm.Mamba.forward() -- NOT as a child nn.Module -- so thop's reported
GFLOPs for every Mamba-family model here (glgf, lc_ss2d, ultralight_baseline)
SILENTLY EXCLUDES the scan itself, which is the dominant cost of a Mamba
layer. This is a known, real gap common to Mamba-family FLOP reporting in
general, not specific to this codebase.

measure_scan_gflops() below closes that gap with a SEPARATE, clearly-labeled
supplementary measurement: it hooks the scan entry point(s) (same
monkeypatch technique as lcmunet/audit.py and lcmunet/delta_diff.py) to
capture the REAL tensor shapes seen during one forward pass, then applies a
FLOP formula derived directly from lcmunet/scan.py's `_selective_scan_ref`
(the exact arithmetic every selective_scan implementation must perform,
CUDA or reference):

    deltaA   = exp(einsum("bdl,dn->bdln", delta, A))        -> B*D*L*N mult + B*D*L*N exp   = 2*B*D*L*N
    deltaB_u = einsum("bdl,bnl,bdl->bdln", delta, B, u)      -> 2*B*D*L*N   (2 multiplies per output element, no reduction)
    for t in L: state = deltaA[t]*state + deltaB_u[t]        -> 2*B*D*N per step  -> 2*B*D*N*L total
                y[t]  = einsum("bdn,bn->bd", state, C[t])    -> ~2*B*D*N per step (N MACs)  -> 2*B*D*N*L total
    total: 2+2+2+2 = 8 * B*D*L*N   (elementary FLOPs: 1 multiply = 1 FLOP, 1 add = 1 FLOP)

This is reported as a SEPARATE column (gflops_scan_supplementary), never
silently folded into "the" GFLOPs number, because it is this project's own
derivation rather than a third-party tool's standard output -- fully
auditable against the formula above, but not independently cross-checked
against a second implementation.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

import lcmunet.lc_vss as lc_vss_module

SCAN_FLOP_MULTIPLIER = 8  # see module docstring derivation


def measure_params_millions(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def measure_module_level_gflops(
    model: torch.nn.Module, input_size: int, batch: int, device: torch.device, input_channels: int = 3
) -> Dict[str, float]:
    """thop module-level GFLOPs (2x MACs convention -- see module docstring
    for what this does and does not capture)."""
    from thop import profile

    model = model.to(device)
    model.eval()
    x = torch.randn(batch, input_channels, input_size, input_size, device=device)
    with torch.no_grad():
        macs, _params = profile(model, inputs=(x,), verbose=False)
    return {"macs": float(macs), "gflops_module_thop": float(2.0 * macs / 1e9)}


def _import_mamba_simple():
    """Indirection point so tests can substitute a fake mamba_ssm-shaped
    module tree without needing the real mamba_ssm installed -- see
    tests/test_efficiency.py's regression test for the bug this guards."""
    import mamba_ssm.modules.mamba_simple as mamba_simple_module  # type: ignore

    return mamba_simple_module


def _capture_scan_shapes(model: torch.nn.Module, x: torch.Tensor) -> Tuple[List[Tuple[Tuple[int, int, int], int]], str]:
    """Hooks BOTH lcmunet.lc_vss.selective_scan (glgf/lc_ss2d) and, if
    importable, mamba_ssm's selective_scan_fn (ultralight_baseline) in a
    SINGLE forward pass, so either code path is captured without running
    the model twice.

    The mamba_ssm branch patches mamba_ssm.modules.mamba_simple.selective_scan_fn
    -- NOT mamba_ssm.ops.selective_scan_interface.selective_scan_fn, even
    though that is where the function is DEFINED. mamba_simple.py imports it
    via `from mamba_ssm.ops.selective_scan_interface import selective_scan_fn`
    (confirmed against the real mamba_ssm source), which binds a SEPARATE
    name into mamba_simple's own module namespace; Mamba.forward()'s call
    resolves against THAT namespace via a plain global lookup, so patching
    the origin module's attribute never intercepts anything (a real bug in
    an earlier version of this function, found and fixed while building
    lcmunet/mechanism_analysis.py, which needs the identical hook for a
    methodology section 11 ESSENTIAL claim and so was checked against the
    real source rather than assumed).

    This also forces every submodule with a `use_fast_path` attribute
    (mamba_ssm.Mamba instances) to False for the duration of this one
    forward pass, restored after: with causal_conv1d installed (the
    intended Colab target after notebooks/01_env.ipynb), Mamba.forward()'s
    default fused path calls mamba_inner_fn instead of selective_scan_fn at
    all, hiding dts inside the kernel (see lcmunet/lc_vss.py's module
    docstring) -- so without this, the hook below would silently observe
    zero calls for ultralight_baseline even once patched at the correct
    module.

    Still best-effort and untestable end-to-end on a machine without
    mamba_ssm installed (this project's local CPU dev box) -- if importing
    mamba_ssm fails for any reason, this branch is silently skipped rather
    than crashing the whole measurement (module-level GFLOPs is still
    valid); the hook-target/use_fast_path LOGIC itself is unit-tested with a
    fake mamba_ssm-shaped module tree (tests/test_efficiency.py) so this
    specific bug class cannot silently regress.
    """
    real_lc_vss_scan = lc_vss_module.selective_scan
    calls: List[Tuple[Tuple[int, int, int], int]] = []
    sources = ["lcmunet.lc_vss.selective_scan"]

    def spy(u, dts, A, *args, **kwargs):
        calls.append((tuple(u.shape), int(A.shape[1])))
        return real_lc_vss_scan(u, dts, A, *args, **kwargs)

    lc_vss_module.selective_scan = spy

    mamba_simple = None
    real_mamba_scan = None
    forced_fast_path: List[Tuple[Any, bool]] = []
    try:
        mamba_simple = _import_mamba_simple()

        real_mamba_scan = mamba_simple.selective_scan_fn

        def mamba_spy(u, dts, A, *args, **kwargs):
            calls.append((tuple(u.shape), int(A.shape[1])))
            return real_mamba_scan(u, dts, A, *args, **kwargs)

        mamba_simple.selective_scan_fn = mamba_spy
        sources.append("mamba_ssm.modules.mamba_simple.selective_scan_fn")

        for m in model.modules():
            if hasattr(m, "use_fast_path"):
                forced_fast_path.append((m, m.use_fast_path))
                m.use_fast_path = False
    except Exception:
        mamba_simple = None

    try:
        model.eval()
        with torch.no_grad():
            model(x)
    finally:
        lc_vss_module.selective_scan = real_lc_vss_scan
        if mamba_simple is not None:
            mamba_simple.selective_scan_fn = real_mamba_scan
        for m, original in forced_fast_path:
            m.use_fast_path = original

    return calls, " + ".join(sources) if calls else "none (no scan calls observed)"


def measure_scan_gflops(
    model: torch.nn.Module, input_size: int, batch: int, device: torch.device, input_channels: int = 3
) -> Dict[str, Any]:
    """SEPARATE, supplementary GFLOPs from the selective-scan recurrence
    itself -- see module docstring. 0.0 for models with no scan (unet,
    malunet, egeunet) -- that is the CORRECT value, not a measurement gap,
    for those (thop's count is already complete for pure-CNN models).
    """
    model = model.to(device)
    x = torch.randn(batch, input_channels, input_size, input_size, device=device)
    calls, source = _capture_scan_shapes(model, x)

    total_flops = sum(SCAN_FLOP_MULTIPLIER * b * d * l * n for (b, d, l), n in calls)
    return {
        "gflops_scan_supplementary": total_flops / 1e9,
        "n_scan_calls": len(calls),
        "scan_hook_source": source,
    }


def measure_fps(
    model: torch.nn.Module,
    input_size: int,
    batch: int,
    device: torch.device,
    n_warmup: int = 20,
    n_measure: int = 100,
    input_channels: int = 3,
) -> Dict[str, Any]:
    """Wall-clock forward-pass throughput. On CUDA, torch.cuda.synchronize()
    brackets both warm-up and the timed loop so async kernel launches never
    leak timing across iterations or into the "warm" state. Runs on CPU too
    (useful for locally sanity-testing this harness), but `is_gpu_measurement`
    is False in that case -- a CPU FPS number must never be reported as this
    prompt's required Colab GPU measurement.
    """
    model = model.to(device)
    model.eval()
    x = torch.randn(batch, input_channels, input_size, input_size, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_measure):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    fps = (n_measure * batch) / elapsed
    return {
        "fps": fps,
        "n_warmup": n_warmup,
        "n_measure": n_measure,
        "elapsed_sec": elapsed,
        "device": str(device),
        "is_gpu_measurement": device.type == "cuda",
    }


def measure_peak_training_memory_mb(
    model: torch.nn.Module, input_size: int, batch: int, device: torch.device, input_channels: int = 3
) -> Optional[float]:
    """Peak GPU memory during one training step (forward + backward),
    batch=8 per this prompt. Returns None -- NOT 0.0 -- when CUDA is
    unavailable: "no memory used" and "undefined without a GPU" are
    different facts, and this project never fabricates the former for the
    latter.
    """
    if device.type != "cuda":
        return None

    model = model.to(device)
    model.train()
    x = torch.randn(batch, input_channels, input_size, input_size, device=device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    y = model(x)
    loss = y.float().sum()
    loss.backward()
    peak_bytes = torch.cuda.max_memory_allocated(device)
    return peak_bytes / (1024.0**2)


def measure_model_efficiency(
    model_name: str,
    model: torch.nn.Module,
    device: torch.device,
    scan_impl: str,
    gpu_name: Optional[str],
    torch_version: str,
    input_size: int = 256,
    fps_batch_sizes: Tuple[int, ...] = (1, 8),
    memory_batch_size: int = 8,
    n_warmup: int = 20,
    n_measure: int = 100,
) -> Dict[str, Any]:
    """Assembles one efficiency.csv row for `model_name`. Params/GFLOPs are
    measured on `device` (structurally device-independent, but some models
    -- ultralight_baseline -- can only run on CUDA at all, so this never
    hardcodes CPU). FPS/peak memory follow this prompt's spec exactly:
    batch=1 and batch=8 FPS, peak training memory at batch=8.
    """
    row: Dict[str, Any] = {"model_name": model_name}

    row["params_M"] = measure_params_millions(model)

    module_gflops = measure_module_level_gflops(model, input_size, batch=1, device=device)
    scan_gflops = measure_scan_gflops(model, input_size, batch=1, device=device)
    row["gflops_module_thop"] = module_gflops["gflops_module_thop"]
    row["gflops_scan_supplementary"] = scan_gflops["gflops_scan_supplementary"]
    row["gflops_total"] = module_gflops["gflops_module_thop"] + scan_gflops["gflops_scan_supplementary"]
    row["n_scan_calls"] = scan_gflops["n_scan_calls"]
    row["scan_hook_source"] = scan_gflops["scan_hook_source"]

    for b in fps_batch_sizes:
        fps_result = measure_fps(model, input_size, b, device, n_warmup=n_warmup, n_measure=n_measure)
        row[f"fps_b{b}"] = fps_result["fps"]
        row[f"fps_b{b}_is_gpu_measurement"] = fps_result["is_gpu_measurement"]

    row["peak_mem_MB_b8"] = measure_peak_training_memory_mb(model, input_size, memory_batch_size, device)

    row["gpu_name"] = gpu_name
    row["torch_version"] = torch_version
    row["scan_impl"] = scan_impl
    row["input_size"] = input_size
    row["n_warmup"] = n_warmup
    row["n_measure"] = n_measure

    return row
