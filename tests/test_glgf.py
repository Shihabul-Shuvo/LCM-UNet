"""GLGF late-fusion baseline/ablation variant (CONTEXT.md; methodology
sections 1, 3.5, 10.1 Ablation A, 12). CPU-runnable, no mamba-ssm dependency
-- same guarantee as lc_ss2d (lcmunet/lc_vss.py's shared-core design).
"""

import torch

from lcmunet.config import RunConfig
from lcmunet.engine import build_model
from lcmunet.glgf import GLGFLayer, GLGFUNet
from lcmunet.lc_vss import LC_PVMLayer


def test_glgf_layer_shape_matches_lc_pvm_layer_shape():
    for input_dim, output_dim in [(24, 32), (32, 48), (64, 48), (48, 32)]:
        layer = GLGFLayer(input_dim, output_dim)
        x = torch.randn(2, input_dim, 16, 16)
        y = layer(x)
        assert y.shape == (2, output_dim, 16, 16)


def test_glgf_layer_mamba_branch_has_no_lc_ss2d_injection():
    """Ablation A requires GLGF's Mamba branch to be completely vanilla --
    no Delta modulation, matching baseline PVM exactly (see lc_vss.py's
    descriptor_type='none' numerical-equivalence guarantee)."""
    layer = GLGFLayer(24, 32)
    assert isinstance(layer.mamba_branch, LC_PVMLayer)
    assert layer.mamba_branch.descriptor_type == "none"
    assert layer.mamba_branch.alpha is None
    assert layer.mamba_branch.w_deltas is None


def test_glgf_layer_gradient_flow_finite_and_nonzero():
    layer = GLGFLayer(24, 32)
    x = torch.randn(2, 24, 16, 16, requires_grad=True)
    y = layer(x)
    y.sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, p in layer.named_parameters():
        assert p.grad is not None, f"{name}: no gradient"
        assert torch.isfinite(p.grad).all(), f"{name}: non-finite gradient"

    # the two branches and the gate must all actually receive signal, not
    # just "some" params -- otherwise this could silently degrade to a
    # single-branch model and no longer be a valid late-fusion ablation row.
    assert (layer.gate_conv.weight.grad != 0).any()
    assert (layer.conv_branch[0].weight.grad != 0).any()
    assert (layer.mamba_branch.core.out_proj.weight.grad != 0).any()


def test_glgf_layer_gate_actually_mixes_both_branches():
    """A degenerate gate stuck at 0 or 1 would make this either a pure conv
    block or a pure PVM block -- not a fusion gate. Sanity-check the gate
    output is not saturated at init for a random input."""
    torch.manual_seed(0)
    layer = GLGFLayer(24, 32)
    x = torch.randn(2, 24, 16, 16)

    f_mamba = layer.mamba_branch(x)
    f_conv = layer.conv_branch(x)
    gate = torch.sigmoid(layer.gate_conv(torch.cat([f_mamba, f_conv], dim=1)))

    assert gate.min() > 1e-4 and gate.max() < 1 - 1e-4


def test_glgf_unet_forward_shape():
    model = GLGFUNet()
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 64, 64)
    assert (y < 0).any() or (y > 1).any()  # raw logits, no final sigmoid


def test_glgf_unet_gradient_flow_all_params_finite():
    model = GLGFUNet()
    x = torch.randn(2, 3, 64, 64, requires_grad=True)
    y = model(x)
    y.sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    n_checked = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
            n_checked += 1
    assert n_checked > 0


def test_glgf_unet_e4_e5_d5_d4_are_glgf_layers_e6_d3_are_plain_pvm():
    model = GLGFUNet()
    assert isinstance(model.encoder4[0], GLGFLayer)
    assert isinstance(model.encoder5[0], GLGFLayer)
    assert isinstance(model.decoder1[0], GLGFLayer)  # D5
    assert isinstance(model.decoder2[0], GLGFLayer)  # D4

    assert isinstance(model.encoder6[0], LC_PVMLayer)
    assert model.encoder6[0].descriptor_type == "none"
    assert isinstance(model.decoder3[0], LC_PVMLayer)  # D3
    assert model.decoder3[0].descriptor_type == "none"


def test_glgf_unet_channel_widths_match_stage_table():
    model = GLGFUNet()
    assert (model.encoder4[0].input_dim, model.encoder4[0].output_dim) == (24, 32)  # E4
    assert (model.encoder5[0].input_dim, model.encoder5[0].output_dim) == (32, 48)  # E5
    assert (model.decoder1[0].input_dim, model.decoder1[0].output_dim) == (64, 48)  # D5
    assert (model.decoder2[0].input_dim, model.decoder2[0].output_dim) == (48, 32)  # D4


def test_engine_build_model_glgf_returns_a_working_glgf_unet():
    """No longer NotImplementedError -- build_model("glgf") must now return
    a real, trainable model (this used to raise NotImplementedError; see
    tests/test_engine.py, which dropped its "glgf still unimplemented" case
    in this same prompt)."""
    config = RunConfig(
        run_name="glgf_engine_check",
        model_name="glgf",
        dataset="kvasir_seg",
        seed=0,
        split_file="splits/kvasir_seg.json",
    )
    model = build_model(config)
    assert isinstance(model, GLGFUNet)

    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    assert y.shape == (1, 1, 64, 64)


def test_glgf_sanity_training_run(make_kvasir_raw):
    """Definition of done: 'glgf' trains a few sanity epochs, valid output,
    finite grads -- exercised end-to-end through the real resumable engine,
    same pattern as other CPU-runnable models in tests/test_engine.py."""
    from lcmunet.data.splits import build_kvasir_split
    from lcmunet.engine import run_one

    paths = make_kvasir_raw(n=20)
    build_kvasir_split(paths)

    config = RunConfig(
        run_name="glgf_sanity",
        model_name="glgf",
        dataset="kvasir_seg",
        seed=0,
        split_file="splits/kvasir_seg.json",
        epochs=2,
        batch_size=4,
        input_size=64,
    )
    result = run_one(config, paths=paths, num_workers=0)

    assert result["completed"] is True
    assert result["reached_epoch"] == 1
    assert all(v == v for v in result["test_metrics"].values() if isinstance(v, float))  # no NaNs
