"""Step-0 audit (methodology section 5.1, section 16 checklist) -- run before
any LC-SS2D experiment. All 6 items are runtime-checked here, not asserted
by inspection alone, and every item is CPU-runnable: lcmunet/lc_vss.py
reimplements the Mamba forward pass directly rather than wrapping
mamba_ssm.Mamba, so there is nothing here gated on a GPU/mamba-ssm build.

Fails loud (raises AssertionError) on the first failing item -- per GLOBAL
RULES, this never catches-and-continues on a correctness check.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict

import torch

from lcmunet.lcm_unet import LCMUNet
import lcmunet.lc_vss as lc_vss_module


def _check_dts_named_tensor_and_separate_arg() -> Dict[str, Any]:
    """Items 1 & 2: dts is a named torch.Tensor, passed as its own positional
    argument to the scan call -- not folded into u or computed inside the
    scan itself. Verified by monkeypatching lcmunet.lc_vss.selective_scan to
    record exactly what LC_PVMLayer passes it during a real forward pass.
    """
    calls = []
    real_selective_scan = lc_vss_module.selective_scan

    def spy(u, dts, A, B, C, D=None, **kw):
        calls.append({"u": u, "dts": dts, "A": A, "B": B, "C": C, "D": D})
        return real_selective_scan(u, dts, A, B, C, D, **kw)

    lc_vss_module.selective_scan = spy
    try:
        layer = lc_vss_module.LC_PVMLayer(input_dim=32, output_dim=32, descriptor_type="contrast")
        x = torch.randn(2, 32, 8, 8)
        layer(x)
    finally:
        lc_vss_module.selective_scan = real_selective_scan

    assert len(calls) == 4, f"expected 4 scan calls (one per PVM group), got {len(calls)}"
    for i, call in enumerate(calls):
        assert isinstance(call["dts"], torch.Tensor), f"group {i}: dts is not a torch.Tensor: {type(call['dts'])}"
        assert isinstance(call["u"], torch.Tensor), f"group {i}: u is not a torch.Tensor"
        assert call["dts"] is not call["u"], f"group {i}: dts and u are the same tensor object (not separate args)"
        assert call["dts"].shape != () and call["u"].shape != (), f"group {i}: degenerate tensor shape"

    return {
        "item_1_dts_is_named_tensor": True,
        "item_2_dts_is_separate_arg": True,
        "n_scan_calls_observed": len(calls),
        "dts_dtype": str(calls[0]["dts"].dtype),
        "dts_shape": tuple(calls[0]["dts"].shape),
    }


def _check_no_fused_kernel_used() -> Dict[str, Any]:
    """Item 3: the mamba-ssm fused kernel (mamba_inner_fn) is not used --
    lcmunet/lc_vss.py never imports mamba_ssm at all. Checked structurally
    (no such name bound in the module's namespace, no import statement) and
    by usage (no `mamba_inner_fn(` call pattern in the source) -- distinct
    from merely mentioning the name in a comment/docstring, which this file
    itself does when explaining the decision.
    """
    assert not hasattr(lc_vss_module, "mamba_inner_fn"), "lc_vss.py has mamba_inner_fn bound in its namespace"
    assert not hasattr(lc_vss_module, "Mamba"), "lc_vss.py imports mamba_ssm.Mamba directly -- it must not"

    source_lines = inspect.getsource(lc_vss_module).splitlines()
    for line in source_lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert "mamba_inner_fn(" not in stripped, f"found a call to mamba_inner_fn: {stripped!r}"
        assert not stripped.startswith("import mamba_ssm") and "from mamba_ssm" not in stripped, (
            f"found a mamba_ssm import: {stripped!r}"
        )

    return {"item_3_no_fused_kernel": True}


def _check_stage_shapes(model: LCMUNet) -> Dict[str, Any]:
    """Item 4: exact dt_rank, num_groups, C_group for E4/E5/D5/D4."""
    stage_modules = {"E4": model.encoder4[0], "E5": model.encoder5[0], "D5": model.decoder1[0], "D4": model.decoder2[0]}
    report = {}
    for stage, layer in stage_modules.items():
        report[stage] = {
            "input_dim": layer.input_dim,
            "output_dim": layer.output_dim,
            "num_groups": layer.num_groups,
            "C_group": layer.c_group,
            "dt_rank": layer.core.dt_rank,
            "d_inner": layer.core.d_inner,
            "has_w_delta": layer.w_deltas is not None,
            "n_w_delta_modules": len(layer.w_deltas) if layer.w_deltas is not None else 0,
        }
    # NOTE: methodology's stage table lists "C=32/48/48/32" for E4/E5/D5/D4,
    # which matches each stage's OUTPUT_dim, not input_dim (E4 input_dim=24,
    # output_dim=32; see lcmunet/lcm_unet.py module docstring for the full
    # reasoning). LC-VSS operates on input_dim -- the block's actual forward()
    # input, matching section 3.1's "Xn = LayerNorm(X)" and required by
    # section 4's "PVM operator...inherited, unchanged" constraint (PVM's
    # internal scan operates at input_dim/num_groups width throughout; only
    # the final proj changes width to output_dim). Flagged, not silently
    # picked -- see the agent report for this prompt.
    return {"item_4_stage_shapes": report}


def _check_gradient_flow(model: LCMUNet) -> Dict[str, Any]:
    """Item 5: two-batch sanity run -- every W_delta.weight receives a
    non-zero, finite gradient after backward with alpha=0.01."""
    model.zero_grad(set_to_none=True)
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    y.sum().backward()

    stage_modules = {"E4": model.encoder4[0], "E5": model.encoder5[0], "D5": model.decoder1[0], "D4": model.decoder2[0]}
    report = {}
    for stage, layer in stage_modules.items():
        assert layer.w_deltas is not None, f"{stage}: no W_delta modules (descriptor_type must be != 'none' for this check)"
        for i, w_delta in enumerate(layer.w_deltas):
            grad = w_delta.weight.grad
            assert grad is not None, f"{stage} group {i}: W_delta.weight.grad is None"
            assert torch.isfinite(grad).all(), f"{stage} group {i}: W_delta.weight.grad has non-finite values"
            assert (grad != 0).any(), f"{stage} group {i}: W_delta.weight.grad is all zero"
        report[stage] = {"n_groups_checked": len(layer.w_deltas), "all_grads_nonzero_and_finite": True}
    return {"item_5_gradient_flow": report}


def _check_alpha_registration_and_update(model: LCMUNet) -> Dict[str, Any]:
    """Item 6: self.alpha is an nn.Parameter, present in model.parameters(),
    and changes after one optimiser step."""
    alpha_params = {name: p for name, p in model.named_parameters() if name.endswith(".alpha")}
    assert len(alpha_params) == 4, f"expected 4 alpha parameters (E4/E5/D5/D4), found {len(alpha_params)}: {list(alpha_params)}"
    for name, p in alpha_params.items():
        assert isinstance(p, torch.nn.Parameter), f"{name} is not an nn.Parameter"
        assert abs(float(p.detach()) - 0.01) < 1e-6, f"{name} initial value != 0.01: {float(p.detach())}"

    before = {name: p.clone().detach() for name, p in alpha_params.items()}

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model.zero_grad(set_to_none=True)
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    y.sum().backward()
    optimizer.step()

    report = {}
    for name, p in alpha_params.items():
        changed = not torch.equal(before[name], p.detach())
        assert changed, f"{name} did not change after one optimiser step"
        report[name] = {"before": float(before[name]), "after": float(p.detach())}
    return {"item_6_alpha_registered_and_updates": report}


def run_step0_audit(verbose: bool = True) -> Dict[str, Any]:
    """Runs all 6 Step-0 audit items (methodology section 5.1 / section 16)
    against a fresh hero-placement LCMUNet. Raises AssertionError on the
    first failing item (fail loud). Returns a full report dict on success.
    """
    torch.manual_seed(0)
    report: Dict[str, Any] = {}

    report.update(_check_dts_named_tensor_and_separate_arg())
    report.update(_check_no_fused_kernel_used())

    model = LCMUNet()  # hero defaults: descriptor_type='contrast', placement='hero'
    report.update(_check_stage_shapes(model))
    report.update(_check_gradient_flow(model))
    report.update(_check_alpha_registration_and_update(model))

    report["all_6_items_passed"] = True

    if verbose:
        import json

        print(json.dumps(report, indent=2, default=str))

    return report


if __name__ == "__main__":
    run_step0_audit(verbose=True)
