"""Segmentation loss (methodology section 6). Exactly BCE + Dice, equal
weight. No other terms -- no focal, no boundary loss, no class weighting.
"""

from __future__ import annotations

import torch
import torch.nn as nn

DICE_EPS = 1e-6


class DiceLoss(nn.Module):
    """DiceLoss = 1 - (2*sum(p*y) + eps) / (sum(p) + sum(y) + eps), per-image then averaged over the batch.

    Operates on logits (applies sigmoid internally), matching CombinedLoss's
    BCEWithLogitsLoss so both terms share one forward pass over raw model
    output.
    """

    def __init__(self, eps: float = DICE_EPS):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).flatten(1)
        targets = targets.flatten(1).float()
        intersection = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """BCE + DiceLoss, equal weight (methodology section 6): loss = BCE + Dice.

    BCE is implemented as BCEWithLogitsLoss (numerically fused sigmoid+BCE)
    rather than sigmoid() followed by BCELoss -- mathematically identical to
    the paper's stated `BCE = -(1/N)*sum[y*log(p) + (1-y)*log(1-p)]` with
    p = sigmoid(logits), just numerically stable for large |logits|.
    """

    def __init__(self, eps: float = DICE_EPS):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(eps=eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        return self.bce(logits, targets) + self.dice(logits, targets)
