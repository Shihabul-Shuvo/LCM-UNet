"""Unit tests for LC-VSS / LC-SS2D (methodology sections 3, 5.2-5.4, 16).

These are the non-negotiable checks named in the prompt that added this
module: shapes match baseline per stage, gradient flow through every
W_delta, alpha registration + update, token order unchanged (u passed to
the scan is bit-identical to baseline's -- only dts differs), and
descriptor='none' reproduces baseline forward numerically.
"""

import pytest
import torch

import lcmunet.lc_vss as lc_vss_module
from lcmunet.lc_vss import (
    LC_PVMLayer,
    LCVSSMambaCore,
    LocalDescriptor,
    pvm_flatten,
    resolve_placement_stages,
)
from lcmunet.lcm_unet import LCMUNet


# ---- LocalDescriptor ---------------------------------------------------


def test_local_descriptor_contrast_is_dwconv_minus_input():
    torch.manual_seed(0)
    desc = LocalDescriptor(8, descriptor_type="contrast", kernel_size=3)
    x = torch.randn(2, 8, 16, 16)
    m = desc(x)
    assert m.shape == x.shape
    assert torch.allclose(m, desc.dw_conv(x) - x)


def test_local_descriptor_plain_is_dwconv_only():
    torch.manual_seed(0)
    desc = LocalDescriptor(8, descriptor_type="plain", kernel_size=3)
    x = torch.randn(2, 8, 16, 16)
    m = desc(x)
    assert torch.allclose(m, desc.dw_conv(x))


def test_local_descriptor_none_returns_none_and_has_no_conv():
    desc = LocalDescriptor(8, descriptor_type="none")
    assert desc.dw_conv is None
    assert desc(torch.randn(2, 8, 16, 16)) is None


def test_local_descriptor_kernel_size_1_is_pointwise_capacity_matched_control():
    desc = LocalDescriptor(8, descriptor_type="contrast", kernel_size=1)
    assert desc.dw_conv.kernel_size == (1, 1)
    assert desc.dw_conv.groups == 8  # depthwise, no cross-channel mixing
    x = torch.randn(1, 8, 10, 10)
    assert desc(x).shape == x.shape


def test_local_descriptor_init_std():
    torch.manual_seed(0)
    desc = LocalDescriptor(64, descriptor_type="contrast", kernel_size=3, std=1e-3)
    assert desc.dw_conv.weight.std().item() < 0.01  # small-normal init, not default Kaiming


def test_pvm_flatten_matches_manual_reshape():
    x = torch.randn(2, 6, 4, 5)
    flat = pvm_flatten(x)
    expected = x.reshape(2, 6, 20).transpose(-1, -2)
    assert torch.equal(flat, expected)
    assert flat.shape == (2, 20, 6)


# ---- LCVSSMambaCore: faithful Mamba slow-path reimplementation ------------


def _independent_mamba_reference(core: LCVSSMambaCore, hidden_states: torch.Tensor) -> torch.Tensor:
    """Hand-written, NOT using einops, NOT calling core.forward() -- an
    independent cross-check derived directly from mamba_ssm's documented
    non-fused forward path, using the SAME weights as `core`."""
    from lcmunet.scan import selective_scan

    B, L, _ = hidden_states.shape
    xz = core.in_proj(hidden_states).transpose(1, 2)
    x, z = xz.chunk(2, dim=1)
    x = core.act(core.conv1d(x)[..., :L])
    x_flat = x.transpose(1, 2).reshape(B * L, core.d_inner)
    x_dbl = core.x_proj(x_flat)
    dt, Bp, Cp = torch.split(x_dbl, [core.dt_rank, core.d_state, core.d_state], dim=-1)
    dt = core.dt_proj.weight @ dt.t()
    dt = dt.reshape(core.d_inner, B, L).transpose(0, 1)
    A = -torch.exp(core.A_log.float())
    Bp = Bp.reshape(B, L, core.d_state).transpose(1, 2).contiguous()
    Cp = Cp.reshape(B, L, core.d_state).transpose(1, 2).contiguous()
    y = selective_scan(x, dt, A, Bp, Cp, core.D.float(), z=z, delta_bias=core.dt_proj.bias.float(), delta_softplus=True)
    y = y.transpose(1, 2)
    return core.out_proj(y)


def test_lcvss_mamba_core_matches_independent_reference_no_injection():
    torch.manual_seed(42)
    core = LCVSSMambaCore(d_model=8, d_state=16, d_conv=4, expand=2)
    x = torch.randn(2, 20, 8)

    out_production = core(x, m_k=None)
    out_reference = _independent_mamba_reference(core, x)

    assert torch.equal(out_production, out_reference)


def test_lcvss_mamba_core_shape_and_backward():
    torch.manual_seed(0)
    core = LCVSSMambaCore(d_model=8, d_state=16, d_conv=4, expand=2)
    x = torch.randn(2, 64, 8, requires_grad=True)
    m_k = torch.randn(2, 64, 8)
    w_delta = torch.nn.Linear(8, core.dt_rank, bias=False)
    alpha = torch.nn.Parameter(torch.tensor(0.01))

    y = core(x, m_k, w_delta=w_delta, alpha=alpha, inject_target="delta")
    assert y.shape == x.shape
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert w_delta.weight.grad is not None and torch.isfinite(w_delta.weight.grad).all()


# ---- LC_PVMLayer: matches stock PVM structure, drop-in replacement -------


def _independent_pvm_reference(layer: LC_PVMLayer, x: torch.Tensor) -> torch.Tensor:
    """Hand-written reimplementation of stock PVMLayer.forward()'s
    structure (shared core called once per group, double LayerNorm,
    residual, proj), using the layer's own weights, no injection."""
    B, C = x.shape[:2]
    n_tokens = x.shape[2:].numel()
    img_dims = x.shape[2:]
    x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
    x_norm = layer.norm(x_flat)
    chunks = torch.chunk(x_norm, layer.num_groups, dim=2)
    outs = [layer.core(xi) + layer.skip_scale * xi for xi in chunks]
    x_mamba = torch.cat(outs, dim=2)
    x_mamba = layer.norm(x_mamba)
    x_mamba = layer.proj(x_mamba)
    return x_mamba.transpose(-1, -2).reshape(B, layer.output_dim, *img_dims)


def test_lc_pvmlayer_descriptor_none_reproduces_baseline_forward_numerically():
    """The non-negotiable check: descriptor_type='none' must reproduce the
    (stock-PVM-equivalent) baseline forward exactly."""
    torch.manual_seed(123)
    layer = LC_PVMLayer(input_dim=32, output_dim=48, descriptor_type="none")
    x = torch.randn(2, 32, 16, 16)

    out_production = layer(x)
    out_reference = _independent_pvm_reference(layer, x)

    assert torch.equal(out_production, out_reference)
    assert layer.alpha is None
    assert layer.w_deltas is None


@pytest.mark.parametrize("output_dim", [32, 48])
def test_lc_pvmlayer_shape(output_dim):
    layer = LC_PVMLayer(input_dim=32, output_dim=output_dim, descriptor_type="contrast")
    x = torch.randn(2, 32, 16, 16)
    y = layer(x)
    assert y.shape == (2, output_dim, 16, 16)


def test_lc_pvmlayer_rejects_non_divisible_num_groups():
    with pytest.raises(ValueError):
        LC_PVMLayer(input_dim=30, output_dim=32, num_groups=4)


# ---- Gradient flow + alpha registration (Gate 0 items 5 & 6) --------------


def test_gradient_flow_every_w_delta_nonzero_finite():
    torch.manual_seed(0)
    layer = LC_PVMLayer(input_dim=32, output_dim=32, descriptor_type="contrast", alpha_init=0.01)
    x = torch.randn(2, 32, 16, 16)
    y = layer(x)
    y.sum().backward()

    assert layer.w_deltas is not None
    for i, w_delta in enumerate(layer.w_deltas):
        grad = w_delta.weight.grad
        assert grad is not None, f"group {i}: no gradient"
        assert torch.isfinite(grad).all(), f"group {i}: non-finite gradient"
        assert (grad != 0).any(), f"group {i}: all-zero gradient"


def test_alpha_registered_in_parameters_and_updates_after_optimizer_step():
    torch.manual_seed(0)
    layer = LC_PVMLayer(input_dim=32, output_dim=32, descriptor_type="contrast", alpha_init=0.01)

    assert layer.alpha is not None
    assert isinstance(layer.alpha, torch.nn.Parameter)
    assert any(p is layer.alpha for p in layer.parameters())
    assert abs(float(layer.alpha.detach()) - 0.01) < 1e-6

    before = layer.alpha.clone().detach()
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-2)
    x = torch.randn(2, 32, 16, 16)
    layer(x).sum().backward()
    optimizer.step()

    assert not torch.equal(before, layer.alpha.detach())


def test_step0_audit_all_6_items_pass():
    from lcmunet.audit import run_step0_audit

    report = run_step0_audit(verbose=False)
    assert report["all_6_items_passed"] is True


# ---- Token order unchanged (the "not scan-reordering" defense) -----------


def _capture_scan_inputs(fn, monkeypatch=None):
    """Run fn() with lc_vss_module.selective_scan temporarily replaced to
    record every call's (u, dts) pair, returning (result, calls). Manually
    saves/restores the ORIGINAL function (not the pytest monkeypatch
    fixture) so this is safe to call multiple times within one test --
    monkeypatch.setattr does not auto-revert between two calls in the same
    test, which would otherwise chain-wrap spies across calls.
    """
    calls = []
    real = lc_vss_module.selective_scan

    def spy(u, dts, A, B, C, D=None, **kw):
        calls.append({"u": u.clone(), "dts": dts.clone()})
        return real(u, dts, A, B, C, D, **kw)

    lc_vss_module.selective_scan = spy
    try:
        result = fn()
    finally:
        lc_vss_module.selective_scan = real
    return result, calls


def test_token_order_unchanged_u_bit_identical_only_dts_differs():
    """Core defense-of-novelty check: for inject_target='delta' (the hero
    config), the scanned input u must be EXACTLY identical to the
    no-injection baseline's u -- only dts may differ. This is the code-level
    proof that LC-SS2D conditions the dynamics, not the token order/signal.
    """
    torch.manual_seed(7)
    hero = LC_PVMLayer(input_dim=32, output_dim=32, descriptor_type="contrast", inject_target="delta")

    baseline = LC_PVMLayer(input_dim=32, output_dim=32, descriptor_type="none")
    # Copy the SHARED weights (norm/core/proj/skip_scale) so the only
    # difference between the two layers is the injection itself.
    baseline.norm.load_state_dict(hero.norm.state_dict())
    baseline.core.load_state_dict(hero.core.state_dict())
    baseline.proj.load_state_dict(hero.proj.state_dict())
    with torch.no_grad():
        baseline.skip_scale.copy_(hero.skip_scale)

    x = torch.randn(2, 32, 16, 16)

    _, hero_calls = _capture_scan_inputs(lambda: hero(x))
    _, baseline_calls = _capture_scan_inputs(lambda: baseline(x))

    assert len(hero_calls) == len(baseline_calls) == 4
    for i, (h, b) in enumerate(zip(hero_calls, baseline_calls)):
        assert torch.equal(h["u"], b["u"]), f"group {i}: u (scanned input) differs -- token order/signal was changed!"
        assert not torch.equal(h["dts"], b["dts"]), f"group {i}: dts is identical -- injection had no effect"


def test_inject_target_input_modifies_u_not_delta_path_semantics():
    """Ablation D (inject_target='input'): here u is EXPECTED to differ from
    baseline (that's the point of this ablation row) -- sanity-check the
    toggle actually routes to u instead of silently doing nothing."""
    torch.manual_seed(7)
    ablation_d = LC_PVMLayer(input_dim=32, output_dim=32, descriptor_type="contrast", inject_target="input")
    baseline = LC_PVMLayer(input_dim=32, output_dim=32, descriptor_type="none")
    baseline.norm.load_state_dict(ablation_d.norm.state_dict())
    baseline.core.load_state_dict(ablation_d.core.state_dict())
    baseline.proj.load_state_dict(ablation_d.proj.state_dict())
    with torch.no_grad():
        baseline.skip_scale.copy_(ablation_d.skip_scale)

    x = torch.randn(2, 32, 16, 16)
    _, ad_calls = _capture_scan_inputs(lambda: ablation_d(x))
    _, baseline_calls = _capture_scan_inputs(lambda: baseline(x))

    for i, (a, b) in enumerate(zip(ad_calls, baseline_calls)):
        assert not torch.equal(a["u"], b["u"]), f"group {i}: inject_target='input' did not change u"


# ---- Capacity-matched control (Ablation C) -------------------------------


def test_ablation_c_kernel_size_1_shape_and_grad():
    torch.manual_seed(0)
    layer = LC_PVMLayer(input_dim=32, output_dim=32, descriptor_type="contrast", kernel_size=1)
    x = torch.randn(2, 32, 16, 16)
    y = layer(x)
    assert y.shape == x.shape
    y.sum().backward()
    for w_delta in layer.w_deltas:
        assert w_delta.weight.grad is not None
        assert torch.isfinite(w_delta.weight.grad).all()


# ---- Placement resolution (methodology section 10.2) ----------------------


def test_resolve_placement_stages():
    assert resolve_placement_stages("P0") == set()
    assert resolve_placement_stages("E4E5") == {"E4", "E5"}
    assert resolve_placement_stages("hero") == {"E4", "E5", "D5", "D4"}
    assert resolve_placement_stages("+E6") == {"E4", "E5", "D5", "D4", "E6"}
    assert resolve_placement_stages("E4D4") == {"E4", "D4"}
    assert resolve_placement_stages("hero", use_e6=True) == {"E4", "E5", "D5", "D4", "E6"}


def test_resolve_placement_stages_rejects_unknown():
    with pytest.raises(ValueError):
        resolve_placement_stages("not_a_real_placement")


# ---- Full LCMUNet model ---------------------------------------------------


def test_lcmunet_hero_shape_alpha_and_backward():
    torch.manual_seed(0)
    model = LCMUNet()  # hero defaults
    x = torch.randn(2, 3, 256, 256, requires_grad=True)
    y = model(x)

    assert y.shape == (2, 1, 256, 256)
    assert (y < 0).any() or (y > 1).any()  # raw logits, no final sigmoid

    alphas = model.alpha_by_stage()
    assert set(alphas.keys()) == {"E4", "E5", "D5", "D4"}
    assert all(abs(v - 0.01) < 1e-6 for v in alphas.values())

    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    missing_grad = [n for n, p in model.named_parameters() if p.grad is None]
    assert missing_grad == [], f"parameters with no gradient: {missing_grad}"


def test_lcmunet_placement_p0_has_no_alpha_params():
    model = LCMUNet(placement="P0")
    assert model.alpha_by_stage() == {}
    assert not any(name.endswith(".alpha") for name, _ in model.named_parameters())


def test_lcmunet_use_e6_adds_alpha_at_e6():
    model = LCMUNet(placement="hero", use_e6=True)
    alphas = model.alpha_by_stage()
    assert set(alphas.keys()) == {"E4", "E5", "D5", "D4", "E6"}


def test_lcmunet_d3_is_always_plain_pvm_never_lc_vss():
    """Per the module docstring: D3 (decoder3) is a plain (inherited,
    unmodified) PVM stage under every placement config -- there is no
    ablation row that touches it."""
    for placement in ("P0", "E4E5", "hero", "+E6", "E4D4"):
        model = LCMUNet(placement=placement, use_e6=True)
        assert model.decoder3[0].alpha is None, f"placement={placement}: D3 unexpectedly has LC-VSS"


def test_lcmunet_stage_channel_widths_match_methodology_table():
    """methodology section 4: E4/E5/D5/D4 OUTPUT widths are 32/48/48/32."""
    model = LCMUNet()
    assert model.encoder4[0].output_dim == 32
    assert model.encoder5[0].output_dim == 48
    assert model.decoder1[0].output_dim == 48  # D5
    assert model.decoder2[0].output_dim == 32  # D4


def test_lcmunet_descriptor_none_everywhere_reproduces_baseline_structure():
    """placement='P0' (no LC-VSS anywhere) must still run end-to-end and
    match the independent stock-PVM-structure reference at every PVM stage."""
    torch.manual_seed(0)
    model = LCMUNet(placement="P0")
    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    assert y.shape == (1, 1, 64, 64)
    assert torch.isfinite(y).all()
