"""Selective-scan primitive: the ONE implementation used by baseline, GLGF, and LC-SS2D.

Per methodology §5.5, every efficiency number in the paper must come from
runs using the SAME selective_scan implementation across every compared
model. This module resolves that choice once, at import time, and exposes
it via SCAN_IMPL so every run's config/results row can record which one ran.

Decision rule:
  - primary:  mamba-ssm's NON-FUSED `selective_scan_fn` (a standalone CUDA
              op where `dts` is a separate, explicit tensor argument) —
              NOT `mamba_inner_fn`, which fuses conv1d + in_proj + dt-compute
              + scan + gate + out_proj into a single kernel call and hides
              dts inside it. This distinction is Step-0 audit item 3 (§5.1):
              LC-SS2D must modify dts BEFORE the scan is invoked, which is
              only possible against the non-fused entry point.
  - fallback: a pure-PyTorch reference scan (`_selective_scan_ref` below),
              used whenever mamba-ssm's CUDA extension isn't importable
              (e.g. a flaky/absent Colab build) or no CUDA device exists.

The choice is made once at import time — not per call — so a single run
cannot silently mix implementations partway through.

GLOBAL RULES rule 5: the mamba-ssm CUDA build is a GPU gate the user runs
and reports from Colab; this module cannot verify "cuda" actually works on
a machine without a CUDA device (it can only confirm import success).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

_cuda_selective_scan_fn = None
_CUDA_IMPORT_ERROR: Optional[Exception] = None

try:
    from mamba_ssm.ops.selective_scan_interface import (  # type: ignore
        selective_scan_fn as _cuda_selective_scan_fn,
    )
except Exception as exc:  # ImportError, or any error while importing mamba_ssm
    _CUDA_IMPORT_ERROR = exc

SCAN_IMPL = "cuda" if (_cuda_selective_scan_fn is not None and torch.cuda.is_available()) else "ref"


def _selective_scan_ref(
    u: torch.Tensor,
    dts: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
):
    """Pure-PyTorch selective scan (real-valued only; SS2D never needs complex A/B/C).

    Shapes:
      u, dts: (Batch, D, L)     A: (D, N)
      B, C:   (Batch, N, L)     D: (D,) or None
      z:      (Batch, D, L) or None      delta_bias: (D,) or None

    Zero-order-hold discretisation (§3.6): Ā = exp(Δ·A), with A
    negative-definite so large Δ -> fast forgetting, small Δ -> long memory.

    Returns y: (Batch, D, L)  [, last_state: (Batch, D, N) if return_last_state]
    """
    dtype_in = u.dtype
    u = u.float()
    delta = dts.float()
    if delta_bias is not None:
        delta = delta + delta_bias.float().view(1, -1, 1)
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, length = u.shape
    dstate = A.shape[1]
    Bf = B.float()
    Cf = C.float()

    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, Bf, u)

    state = A.new_zeros(batch, dim, dstate)
    ys = []
    for t in range(length):
        state = deltaA[:, :, t] * state + deltaB_u[:, :, t]
        ys.append(torch.einsum("bdn,bn->bd", state, Cf[:, :, t]))
    y = torch.stack(ys, dim=2)  # (Batch, D, L)

    if D is not None:
        y = y + u * D.float().view(1, -1, 1)
    if z is not None:
        y = y * F.silu(z.float())

    y = y.to(dtype_in)
    if return_last_state:
        return y, state
    return y


def selective_scan(u, dts, A, B, C, D=None, **kw):
    """The one selective-scan entry point every model (baseline/GLGF/LC-SS2D) must call.

    Dispatches to whichever implementation SCAN_IMPL locked in at import
    time. Same signature either way, so swapping SCAN_IMPL never requires
    touching call sites.
    """
    if SCAN_IMPL == "cuda":
        return _cuda_selective_scan_fn(u, dts, A, B, C, D, **kw)
    return _selective_scan_ref(u, dts, A, B, C, D, **kw)
