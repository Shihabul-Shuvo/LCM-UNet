"""Efficiency report orchestration (methodology sections 5.5, 5.6, 8, 9).

Measures Params/GFLOPs/FPS/peak-memory (lcmunet/efficiency.py) for EVERY
compared model (baseline, GLGF, LC-SS2D, and each reproduced comparator)
under the SAME scan_impl (recorded from results/env.json via
lcmunet.experiment_matrix.resolve_scan_impl -- section 5.5 fairness rule:
"mixed-implementation efficiency comparisons are invalid", so this module
records and states the single scan_impl used for every row rather than
letting it vary per model), and writes results/efficiency.csv +
results/efficiency_report.md.

Models that cannot be built in the current environment (ultralight_baseline
requires mamba-ssm's CUDA build -- GLOBAL RULES rule 5) are skipped, NOT
silently dropped: their absence and the reason are recorded in the report
(same "orchestration resilience" pattern as lcmunet.data.splits.build_all_splits)
so the emitted report never claims completeness it doesn't have.

VERIFIES (does not assume) methodology section 5.6's claim that LC-SS2D
adds fewer params than GLGF -- reports the measured delta plainly, in
either direction, once both rows are available.
"""

from __future__ import annotations

import csv as _csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from lcmunet import efficiency as eff
from lcmunet.config import DEFAULT_MODEL_CFG, RunConfig
from lcmunet.experiment_matrix import resolve_scan_impl

MODEL_NAMES: Tuple[str, ...] = ("ultralight_baseline", "glgf", "lc_ss2d", "unet", "malunet", "egeunet")

EFFICIENCY_COLUMNS: Tuple[str, ...] = (
    "model_name",
    "params_M",
    "gflops_module_thop",
    "gflops_scan_supplementary",
    "gflops_total",
    "n_scan_calls",
    "scan_hook_source",
    "fps_b1",
    "fps_b1_is_gpu_measurement",
    "fps_b8",
    "fps_b8_is_gpu_measurement",
    "peak_mem_MB_b8",
    "gpu_name",
    "torch_version",
    "scan_impl",
    "input_size",
    "n_warmup",
    "n_measure",
)


def _build_config(model_name: str, scan_impl: str) -> RunConfig:
    """A minimal RunConfig just to reuse lcmunet.engine.build_model's
    existing, tested model-construction logic -- efficiency measurement is
    architecture-level, not tied to any particular dataset/seed."""
    return RunConfig(
        run_name=f"efficiency_{model_name}",
        model_name=model_name,
        dataset="kvasir_seg",
        seed=0,
        split_file="splits/kvasir_seg.json",
        scan_impl=scan_impl,
        model_cfg=dict(DEFAULT_MODEL_CFG),
    )


def build_all_models(scan_impl: str) -> Tuple[Dict[str, torch.nn.Module], Dict[str, str]]:
    """Returns ({model_name: model}, {model_name: error}) -- every
    MODEL_NAMES entry ends up in exactly one of the two dicts. A model that
    fails to build (e.g. ultralight_baseline without mamba-ssm's CUDA
    build) does not block measuring the others.
    """
    from lcmunet.engine import build_model

    models: Dict[str, torch.nn.Module] = {}
    errors: Dict[str, str] = {}
    for model_name in MODEL_NAMES:
        try:
            models[model_name] = build_model(_build_config(model_name, scan_impl))
        except Exception as exc:  # noqa: BLE001 -- orchestration boundary, see module docstring
            errors[model_name] = repr(exc)
    return models, errors


def verify_lc_ss2d_lighter_than_glgf(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Methodology section 5.6: "less than GLGF (no gate conv, no fused
    branch)" -- VERIFIED here from measured params_M, never assumed.
    Returns None if either row is missing (nothing to verify yet).
    """
    by_name = {r["model_name"]: r for r in rows}
    if "lc_ss2d" not in by_name or "glgf" not in by_name:
        return None
    hero_params = by_name["lc_ss2d"]["params_M"]
    glgf_params = by_name["glgf"]["params_M"]
    delta = hero_params - glgf_params  # negative -> lc_ss2d IS lighter (confirms the claim)
    return {
        "lc_ss2d_params_M": hero_params,
        "glgf_params_M": glgf_params,
        "delta_M": delta,
        "claim_confirmed": delta < 0,
    }


def _write_efficiency_csv(paths, rows: List[Dict[str, Any]]) -> Path:
    path = Path(paths.results) / "efficiency.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=EFFICIENCY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in EFFICIENCY_COLUMNS})
    return path


def _fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return "N/A"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, float)):
        return f"{x:.{digits}f}"
    return str(x)


def render_efficiency_report_md(
    rows: List[Dict[str, Any]],
    errors: Dict[str, str],
    scan_impl: str,
    scan_impl_source: str,
    device: torch.device,
    gpu_name: Optional[str],
    torch_version: str,
) -> str:
    lines = ["# Efficiency report (methodology sections 5.5, 5.6, 8, 9)", ""]
    lines.append(f"**scan_impl = `{scan_impl}`** for EVERY model below (source: {scan_impl_source}).")
    lines.append(
        "Mixed-implementation efficiency comparisons are invalid (section 5.5) -- every row in "
        "this report used the same selective-scan implementation."
    )
    lines.append("")
    lines.append(f"Device: `{device}` | GPU: `{gpu_name or 'N/A (no CUDA device)'}` | torch: `{torch_version}`")
    lines.append("")

    is_gpu = device.type == "cuda"
    if not is_gpu:
        lines.append(
            "**WARNING: this run has NO CUDA device.** FPS and peak training GPU memory below "
            "are either N/A or CPU-measured (`*_is_gpu_measurement=False`) -- CPU numbers are "
            "for locally sanity-testing this harness ONLY and must NEVER be reported as this "
            "prompt's required Colab GPU measurements. Params and GFLOPs are structural counts "
            "and ARE valid regardless of device."
        )
        lines.append("")

    lines.append("## Measured rows")
    lines.append("")
    lines.append("| Model | Params (M) | GFLOPs (module, thop) | GFLOPs (scan, supplementary) | GFLOPs (total) | FPS b=1 | FPS b=8 | Peak train mem (MB, b=8) |")
    lines.append("|:--|--:|--:|--:|--:|--:|--:|--:|")
    for row in rows:
        lines.append(
            f"| {row['model_name']} | {_fmt(row['params_M'])} | {_fmt(row['gflops_module_thop'])} | "
            f"{_fmt(row['gflops_scan_supplementary'])} | {_fmt(row['gflops_total'])} | "
            f"{_fmt(row.get('fps_b1'), 1)} | {_fmt(row.get('fps_b8'), 1)} | {_fmt(row.get('peak_mem_MB_b8'), 1)} |"
        )
    lines.append("")

    if errors:
        lines.append("## Models NOT measured (this report is INCOMPLETE until these are fixed)")
        lines.append("")
        for model_name, error in errors.items():
            lines.append(f"- **{model_name}**: {error}")
        lines.append("")

    claim = verify_lc_ss2d_lighter_than_glgf(rows)
    lines.append("## Claim check: does LC-SS2D add fewer params than GLGF? (section 5.6)")
    lines.append("")
    if claim is None:
        lines.append("*Not verifiable yet -- both `lc_ss2d` and `glgf` rows are required (see 'Models NOT measured' above).*")
    else:
        verdict = "CONFIRMED" if claim["claim_confirmed"] else "FALSE -- reversed from what the methodology claims"
        lines.append(f"**{verdict}.**")
        lines.append("")
        lines.append(f"- LC-SS2D: {claim['lc_ss2d_params_M']:.4f} M params")
        lines.append(f"- GLGF: {claim['glgf_params_M']:.4f} M params")
        lines.append(f"- Delta (LC-SS2D − GLGF): {claim['delta_M']:+.4f} M")
    lines.append("")

    lines.append("## Measurement notes")
    lines.append("")
    lines.append(
        "- **GFLOPs (module, thop)**: hook-based, counts every nn.Module layer (conv/linear/"
        "groupnorm/etc.) at 256x256, batch=1. 2x-MACs convention."
    )
    lines.append(
        "- **GFLOPs (scan, supplementary)**: a SEPARATE, this-project-derived measurement of the "
        "selective-scan recurrence itself (thop cannot see it -- it is a plain function call, not "
        "a child nn.Module). See lcmunet/efficiency.py's module docstring for the exact formula "
        "and its derivation. 0.0 is the CORRECT value for pure-CNN comparators (unet/malunet/"
        "egeunet), not a measurement gap."
    )
    lines.append(
        "- **FPS**: mean throughput over `n_measure` timed passes after `n_warmup` untimed passes "
        "(see the row's own n_warmup/n_measure in efficiency.csv), torch.cuda.synchronize()-"
        "bracketed on GPU."
    )
    lines.append("- **Peak train mem**: torch.cuda.max_memory_allocated() after one forward+backward step at batch=8. `N/A` (not `0`) without CUDA.")
    lines.append("")

    return "\n".join(lines)


def generate_efficiency_report(
    paths,
    input_size: int = 256,
    fps_batch_sizes: Tuple[int, ...] = (1, 8),
    memory_batch_size: int = 8,
    n_warmup: int = 20,
    n_measure: int = 100,
) -> Dict[str, Any]:
    scan_impl, scan_impl_source = resolve_scan_impl(paths)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    torch_version = torch.__version__

    models, errors = build_all_models(scan_impl)

    rows: List[Dict[str, Any]] = []
    for model_name, model in models.items():
        row = eff.measure_model_efficiency(
            model_name, model, device=device, scan_impl=scan_impl, gpu_name=gpu_name, torch_version=torch_version,
            input_size=input_size, fps_batch_sizes=fps_batch_sizes, memory_batch_size=memory_batch_size,
            n_warmup=n_warmup, n_measure=n_measure,
        )
        rows.append(row)

    csv_path = _write_efficiency_csv(paths, rows)
    md = render_efficiency_report_md(rows, errors, scan_impl, scan_impl_source, device, gpu_name, torch_version)
    md_path = Path(paths.results) / "efficiency_report.md"
    md_path.write_text(md, encoding="utf-8")

    return {"efficiency_csv": csv_path, "efficiency_report_md": md_path, "rows": rows, "errors": errors}
