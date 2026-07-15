import torch
import pytest

from lcmunet import scan as scan_module
from lcmunet.scan import SCAN_IMPL, selective_scan


def test_scan_impl_locks_to_ref_without_cuda_mamba():
    # On a machine with no CUDA device, the lock must resolve to "ref" --
    # never silently claim "cuda" without a working GPU (GLOBAL RULES rule 5).
    if not torch.cuda.is_available():
        assert SCAN_IMPL == "ref"


def test_ref_scan_shape_and_dtype():
    torch.manual_seed(0)
    batch, dim, dstate, length = 2, 6, 4, 10
    u = torch.randn(batch, dim, length)
    dts = torch.rand(batch, dim, length)
    A = -torch.rand(dim, dstate)  # negative-definite, per methodology §3.6
    B = torch.randn(batch, dstate, length)
    C = torch.randn(batch, dstate, length)
    D = torch.randn(dim)

    y = selective_scan(u, dts, A, B, C, D)

    assert y.shape == (batch, dim, length)
    assert y.dtype == u.dtype


def test_ref_scan_gradients_are_finite():
    torch.manual_seed(0)
    batch, dim, dstate, length = 2, 4, 4, 8
    u = torch.randn(batch, dim, length, requires_grad=True)
    dts = torch.rand(batch, dim, length, requires_grad=True)
    A = (-torch.rand(dim, dstate)).requires_grad_()
    B = torch.randn(batch, dstate, length, requires_grad=True)
    C = torch.randn(batch, dstate, length, requires_grad=True)
    D = torch.randn(dim, requires_grad=True)

    y = selective_scan(u, dts, A, B, C, D)
    y.sum().backward()

    for name, t in [("u", u), ("dts", dts), ("A", A), ("B", B), ("C", C), ("D", D)]:
        assert t.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(t.grad).all(), f"{name} gradient has non-finite values"


def test_ref_scan_matches_zero_decay_cumsum_closed_form():
    """With A=0 (no decay), the recurrence collapses to a cumulative sum --
    an independent closed form (not the scan code itself) to check against."""
    torch.manual_seed(0)
    batch, dim, dstate, length = 1, 3, 2, 12
    u = torch.randn(batch, dim, length)
    dts = torch.rand(batch, dim, length) + 0.1
    A = torch.zeros(dim, dstate)
    B = torch.randn(batch, dstate, length)
    C = torch.randn(batch, dstate, length)

    y = selective_scan(u, dts, A, B, C, D=None)

    deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", dts, B, u)
    state_cumsum = torch.cumsum(deltaB_u, dim=2)  # (batch, dim, length, dstate)
    y_expected = torch.einsum("bdln,bnl->bdl", state_cumsum, C)

    assert torch.allclose(y, y_expected, atol=1e-5)


def test_ref_scan_large_delta_forgets_and_small_delta_remembers():
    """Sanity check on §3.6's state-space-level meaning: with A very negative,
    a large Δ token should suppress carried-over state much more than a
    small Δ token does."""
    torch.manual_seed(0)
    batch, dim, dstate, length = 1, 1, 1, 2
    u = torch.tensor([[[5.0, 0.0]]])  # strong signal at t=0, nothing at t=1
    A = torch.tensor([[-5.0]])
    B = torch.ones(batch, dstate, length)
    C = torch.ones(batch, dstate, length)

    dts_small = torch.tensor([[[0.5, 0.01]]])  # small Δ at t=1 -> long memory
    dts_large = torch.tensor([[[0.5, 5.0]]])  # large Δ at t=1 -> fast forgetting

    y_small = selective_scan(u, dts_small, A, B, C, D=None)
    y_large = selective_scan(u, dts_large, A, B, C, D=None)

    # both start from the same t=0 state; a larger Δ at t=1 must forget more
    assert y_small[0, 0, 1].item() > y_large[0, 0, 1].item()


@pytest.mark.skipif(
    scan_module._cuda_selective_scan_fn is None or not torch.cuda.is_available(),
    reason=(
        "mamba-ssm CUDA scan not available in this environment. The CUDA "
        "build is a GPU gate the user runs and reports from Colab "
        "(GLOBAL RULES rule 5) -- cannot be verified on this machine."
    ),
)
def test_cuda_ref_numerical_equivalence():
    torch.manual_seed(0)
    batch, dim, dstate, length = 2, 8, 4, 16
    device = "cuda"
    u = torch.randn(batch, dim, length, device=device)
    dts = torch.rand(batch, dim, length, device=device)
    A = -torch.rand(dim, dstate, device=device)
    B = torch.randn(batch, dstate, length, device=device)
    C = torch.randn(batch, dstate, length, device=device)
    D = torch.randn(dim, device=device)

    y_cuda = scan_module._cuda_selective_scan_fn(u, dts, A, B, C, D)
    y_ref = scan_module._selective_scan_ref(u, dts, A, B, C, D)

    assert torch.allclose(y_cuda, y_ref, atol=1e-3, rtol=1e-3)
