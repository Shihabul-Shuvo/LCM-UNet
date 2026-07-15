"""Forward/backward smoke tests for every registered comparator builder
(methodology section 9). ultralight_baseline needs mamba-ssm's CUDA build
(GLOBAL RULES rule 5) and cannot run on this CPU-only machine -- its tests
here check the fail-loud behavior instead (clear errors, never a silent
partial result).
"""

import pytest
import torch

from lcmunet.adapters import LogitsAdapter
from lcmunet.comparators import load_egeunet, load_malunet
from lcmunet.unet import UNet


def _forward_backward(model, size=64, batch=2):
    x = torch.randn(batch, 3, size, size, requires_grad=True)
    y = model(x)
    assert y.shape == (batch, 1, size, size)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
    return y


def test_unet_forward_backward():
    model = UNet()
    y = _forward_backward(model, size=256, batch=1)  # standard U-Net needs a large enough input for 4 downsamples
    assert (y < 0).any() or (y > 1).any()  # raw logits, unbounded -- no final sigmoid

    n_params = sum(p.numel() for p in model.parameters())
    assert 25e6 < n_params < 35e6  # standard U-Net is ~31M params


def test_malunet_forward_backward():
    model = LogitsAdapter(load_malunet())
    _forward_backward(model, size=64, batch=2)

    n_params = sum(p.numel() for p in model.parameters())
    assert 0.1e6 < n_params < 0.3e6  # paper: ~0.177M params


def test_egeunet_forward_backward():
    model = LogitsAdapter(load_egeunet())
    _forward_backward(model, size=64, batch=2)

    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 0.1e6  # paper: ~53K params


def test_malunet_and_egeunet_output_is_logits_not_probabilities():
    """The vendored models' raw forward() returns sigmoid-bounded [0,1]
    output; LogitsAdapter must unwrap that back to unbounded logits."""
    torch.manual_seed(0)
    x = torch.randn(1, 3, 64, 64)

    raw_malunet = load_malunet()
    raw_out = raw_malunet(x)
    assert (raw_out >= 0).all() and (raw_out <= 1).all()  # confirms the model itself IS sigmoid-bounded

    wrapped_malunet = LogitsAdapter(raw_malunet)
    logits = wrapped_malunet(x)
    assert torch.allclose(torch.sigmoid(logits), raw_out.clamp(1e-6, 1 - 1e-6), atol=1e-3)


def test_egeunet_gt_ds_false_is_a_known_upstream_bug():
    """Documented in third_party/EGE-UNet/VENDORED.md: gt_ds=False omits a
    required argument to group_aggregation_bridge.forward(). This test pins
    that the bug is still present at the vendored commit (so if upstream
    ever fixes it, we notice) and that our default (gt_ds=True) avoids it."""
    from lcmunet.comparators import EGEUNET_DIR, _ensure_on_path

    _ensure_on_path(EGEUNET_DIR)
    from models.egeunet import EGEUNet  # type: ignore

    broken = EGEUNet(gt_ds=False)
    with pytest.raises(TypeError, match="mask"):
        broken(torch.randn(1, 3, 64, 64))

    working = load_egeunet()  # gt_ds=True by default
    out = working(torch.randn(1, 3, 64, 64))
    assert isinstance(out, tuple) and len(out) == 2


def test_ultralight_baseline_load_fails_clearly_without_mamba_ssm():
    from lcmunet.backbone import load_ultralight_vmunet

    with pytest.raises(ImportError, match="mamba-ssm"):
        load_ultralight_vmunet()


def test_build_model_ultralight_baseline_fails_clearly_when_scan_impl_is_ref():
    from lcmunet.config import RunConfig
    from lcmunet.engine import build_model
    from lcmunet.scan import SCAN_IMPL

    config = RunConfig(
        run_name="gate1_check",
        model_name="ultralight_baseline",
        dataset="kvasir_seg",
        seed=0,
        split_file="splits/kvasir_seg.json",
    )
    if SCAN_IMPL == "cuda":
        pytest.skip("SCAN_IMPL is 'cuda' on this machine; the ref-gate branch isn't exercised here.")
    with pytest.raises(RuntimeError, match="SCAN_IMPL"):
        build_model(config)
