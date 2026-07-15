"""Standard U-Net (Ronneberger et al., 2015, "U-Net: Convolutional Networks
for Biomedical Image Segmentation," MICCAI). A clean-room reimplementation
of the well-established textbook architecture -- not vendored from any
specific repo, since there is no single canonical public implementation and
the architecture itself (encoder/decoder with skip connections, doubling
channels per downsample) is standard, well-documented, and low-risk to
reimplement directly. Used as the "heavy" comparator in methodology
section 9's table, contrasting against the lightweight model family.

Outputs raw logits directly (no final sigmoid) -- unlike every vendored
comparator in third_party/, this one needs no LogitsAdapter.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = _DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Guard against off-by-one spatial mismatches from odd input sizes
        # (standard U-Net concern; center-crop/pad the skip to match).
        if x.shape[-2:] != skip.shape[-2:]:
            diff_h = skip.shape[-2] - x.shape[-2]
            diff_w = skip.shape[-1] - x.shape[-1]
            x = nn.functional.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """Standard 4-downsample U-Net. base_channels=64 matches the original
    paper (this is deliberately the "heavy" comparator, not a lightweight
    variant -- see module docstring)."""

    def __init__(self, num_classes: int = 1, input_channels: int = 3, base_channels: int = 64):
        super().__init__()
        c = [base_channels * (2**i) for i in range(5)]  # e.g. [64,128,256,512,1024]

        self.inc = _DoubleConv(input_channels, c[0])
        self.down1 = _Down(c[0], c[1])
        self.down2 = _Down(c[1], c[2])
        self.down3 = _Down(c[2], c[3])
        self.down4 = _Down(c[3], c[4])

        self.up1 = _Up(c[4], c[3], c[3])
        self.up2 = _Up(c[3], c[2], c[2])
        self.up3 = _Up(c[2], c[1], c[1])
        self.up4 = _Up(c[1], c[0], c[0])

        self.outc = nn.Conv2d(c[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)  # raw logits, (B, num_classes, H, W)
