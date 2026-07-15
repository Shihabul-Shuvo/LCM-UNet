"""Test-only toy model, used SOLELY to exercise lcmunet/engine.py's mechanics
(checkpoint/resume/AMP/scheduler/alpha-logging) before any real segmentation
model exists in this repo.

This is NOT a candidate architecture and must never back a real experiment
row. As of this module's creation, no baseline (reproduced UltraLight PVM),
GLGF, or LC-SS2D/LCM-UNet model exists in lcmunet/: the real
UltraLight_VM_UNet backbone (lcmunet/backbone.py) requires mamba-ssm (a
GPU-only build -- GLOBAL RULES rule 5), and LC-VSS implementation is pending
an architecture decision flagged in a prior session (VMamba-style
4-directional SS2D cross-scan vs. the vendored PVM's channel-group scan --
see third_party/UltraLight-VM-UNet/VENDORED.md). engine.py is deliberately
model-agnostic; this module lets that be proven by a real end-to-end test
instead of asserted.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _DummyAlphaBlock(nn.Module):
    """Mimics an LC-VSS block only insofar as it exposes a learnable scalar
    named `alpha` (methodology section 3.3), so engine.py's alpha-logging /
    mechanism-collapse-warning code path can be exercised end to end. Not an
    SSM, not LC-SS2D -- just enough surface area for that one code path.
    """

    def __init__(self, channels: int, alpha_init: float = 0.01):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * torch.tanh(self.conv(x))


class EngineSanityNet(nn.Module):
    """Tiny conv net, (B,3,H,W) -> (B,1,H,W) logits. CPU-cheap on purpose."""

    def __init__(self, channels: int = 8, with_alpha_block: bool = True):
        super().__init__()
        self.stem = nn.Conv2d(3, channels, 3, padding=1)
        self.alpha_block: nn.Module = _DummyAlphaBlock(channels) if with_alpha_block else nn.Identity()
        self.head = nn.Conv2d(channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.stem(x))
        h = self.alpha_block(h)
        return self.head(h)
