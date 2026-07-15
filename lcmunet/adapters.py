"""Adapters that bridge vendored, unmodified model implementations to this
project's logits-based loss/metrics interface (lcmunet/losses.py's
BCEWithLogitsLoss + DiceLoss, lcmunet/metrics.py's evaluate() both expect
raw logits and apply sigmoid themselves).

Every vendored baseline inspected so far (third_party/UltraLight-VM-UNet,
and MALUNet/EGE-UNet vendored alongside it) hard-codes a final
`torch.sigmoid(...)` at the end of forward() and returns probabilities, not
logits. Feeding that straight into our pipeline would apply sigmoid twice
-- silently wrong, not an error: sigmoid(x) for x in [0,1] lands in
[0.5, 0.731], so thresholding at 0.5 would classify every pixel as
foreground. LogitsAdapter recovers logits via the exact inverse (logit =
sigmoid^-1), so the rest of the pipeline is identical for every model
regardless of whether the vendored code returns probabilities or logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LogitsAdapter(nn.Module):
    """Wraps a model whose forward() already applies a final sigmoid,
    converting its probability output back to logit space via torch.logit
    (clamped for numerical stability near 0/1)."""

    def __init__(self, model: nn.Module, eps: float = 1e-6):
        super().__init__()
        self.model = model
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, tuple):
            # some vendored models return (deep_supervision_outputs, main_output)
            out = out[-1]
        probs = out.clamp(self.eps, 1.0 - self.eps)
        return torch.logit(probs)
