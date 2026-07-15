"""LCM-UNet (methodology section 4): UltraLight VM-UNet's backbone structure
with LC-VSS injected at E4/E5/D5/D4, everything else inherited unchanged.

Reimplements the vendored UltraLight_VM_UNet's non-Mamba modules
(Channel_Att_Bridge, Spatial_Att_Bridge, SC_Att_Bridge, conv stages,
GroupNorm/GELU/interpolate structure) directly rather than importing them,
because third_party/UltraLight-VM-UNet/models/UltraLight_VM_UNet.py has
`from mamba_ssm import Mamba` at module level -- importing ANYTHING from
that file (even the mamba-independent classes) fails on any machine without
mamba-ssm installed. This reimplementation is verified byte-for-byte
faithful to the vendored source by direct comparison, not written from
memory (see the agent report for this prompt). Every "PVM-family" slot
(encoder4/5/6, decoder1/2/3) uses lcmunet.lc_vss.LC_PVMLayer, which is
mathematically identical to stock PVMLayer when descriptor_type='none'
(verified in tests/test_lc_vss.py) -- so this entire model runs on CPU,
without mamba-ssm or a GPU, for every stage, not just the LC-VSS ones.

STAGE-TABLE DISCREPANCY (flagged, not silently resolved). Methodology
section 4's stage table lists "D3-D1: Conv blocks, inherited, unchanged."
But the actual vendored UltraLight_VM_UNet.py has `decoder3 = PVMLayer(
input_dim=c_list[3], output_dim=c_list[2])` -- D3 (the decoder stage that
pairs with encoder3's skip connection, by resolution) IS a PVMLayer in the
real backbone, not a conv block; only decoder4 (D2) and decoder5 (D1) are
conv. This module follows the REAL vendored structure (D3 = plain PVM,
inherited/unchanged, analogous to E6's status) rather than the table, since
methodology section 4 itself says the backbone structure is "inherited...
unchanged" -- matching actual inherited code takes priority over a
transcription simplification in the summary table. The LC-VSS PLACEMENT
itself (which 4 stages are NEW) is unambiguous in the text and unaffected:
E4, E5, D5, D4 only. D3 never receives LC-VSS under any placement config
(unlike E6, which can via placement='+E6' or use_e6=True) -- there is no
ablation row that touches D3.
"""

from __future__ import annotations

import math
from typing import Dict, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from lcmunet.lc_vss import LC_PVMLayer, resolve_placement_stages

DEFAULT_C_LIST = [8, 16, 24, 32, 48, 64]  # methodology section 4


class ChannelAttBridge(nn.Module):
    """Reimplementation of the vendored Channel_Att_Bridge (see module docstring)."""

    def __init__(self, c_list, split_att: str = "fc"):
        super().__init__()
        c_list_sum = sum(c_list) - c_list[-1]
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        make = (lambda c: nn.Linear(c_list_sum, c)) if split_att == "fc" else (lambda c: nn.Conv1d(c_list_sum, c, 1))
        self.att1 = make(c_list[0])
        self.att2 = make(c_list[1])
        self.att3 = make(c_list[2])
        self.att4 = make(c_list[3])
        self.att5 = make(c_list[4])
        self.sigmoid = nn.Sigmoid()

    def forward(self, t1, t2, t3, t4, t5):
        att = torch.cat((self.avgpool(t1), self.avgpool(t2), self.avgpool(t3), self.avgpool(t4), self.avgpool(t5)), dim=1)
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        if self.split_att != "fc":
            att = att.transpose(-1, -2)
        att1 = self.sigmoid(self.att1(att))
        att2 = self.sigmoid(self.att2(att))
        att3 = self.sigmoid(self.att3(att))
        att4 = self.sigmoid(self.att4(att))
        att5 = self.sigmoid(self.att5(att))
        if self.split_att == "fc":
            att1 = att1.transpose(-1, -2).unsqueeze(-1).expand_as(t1)
            att2 = att2.transpose(-1, -2).unsqueeze(-1).expand_as(t2)
            att3 = att3.transpose(-1, -2).unsqueeze(-1).expand_as(t3)
            att4 = att4.transpose(-1, -2).unsqueeze(-1).expand_as(t4)
            att5 = att5.transpose(-1, -2).unsqueeze(-1).expand_as(t5)
        else:
            att1 = att1.unsqueeze(-1).expand_as(t1)
            att2 = att2.unsqueeze(-1).expand_as(t2)
            att3 = att3.unsqueeze(-1).expand_as(t3)
            att4 = att4.unsqueeze(-1).expand_as(t4)
            att5 = att5.unsqueeze(-1).expand_as(t5)
        return att1, att2, att3, att4, att5


class SpatialAttBridge(nn.Module):
    """Reimplementation of the vendored Spatial_Att_Bridge (see module docstring)."""

    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3), nn.Sigmoid())

    def forward(self, t1, t2, t3, t4, t5):
        att_list = []
        for t in (t1, t2, t3, t4, t5):
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att_list.append(self.shared_conv2d(torch.cat([avg_out, max_out], dim=1)))
        return tuple(att_list)


class SCAttBridge(nn.Module):
    """Reimplementation of the vendored SC_Att_Bridge (see module docstring)."""

    def __init__(self, c_list, split_att: str = "fc"):
        super().__init__()
        self.catt = ChannelAttBridge(c_list, split_att=split_att)
        self.satt = SpatialAttBridge()

    def forward(self, t1, t2, t3, t4, t5):
        r1, r2, r3, r4, r5 = t1, t2, t3, t4, t5
        satt1, satt2, satt3, satt4, satt5 = self.satt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = satt1 * t1, satt2 * t2, satt3 * t3, satt4 * t4, satt5 * t5

        r1_, r2_, r3_, r4_, r5_ = t1, t2, t3, t4, t5
        t1, t2, t3, t4, t5 = t1 + r1, t2 + r2, t3 + r3, t4 + r4, t5 + r5

        catt1, catt2, catt3, catt4, catt5 = self.catt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = catt1 * t1, catt2 * t2, catt3 * t3, catt4 * t4, catt5 * t5
        return t1 + r1_, t2 + r2_, t3 + r3_, t4 + r4_, t5 + r5_


def _init_weights(m: nn.Module) -> None:
    """Identical to the vendored UltraLight_VM_UNet._init_weights."""
    from timm.models.layers import trunc_normal_

    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv1d):
        n = m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()


class LCMUNet(nn.Module):
    """methodology section 4. c_list=[8,16,24,32,48,64]. LC-VSS (methodology
    sections 3, 5) is injected at the stages named by `placement`
    (default 'hero' = {E4, E5, D5, D4}); every other PVM-family stage
    (E6, D3) is plain (descriptor_type='none', mathematically identical to
    stock PVMLayer -- see lcmunet/lc_vss.py).

    All ablation toggles are config, not new model classes (descriptor_type,
    kernel_size, inject_target, placement, use_e6, alpha_init, wdelta_std --
    see lcmunet.config.DEFAULT_MODEL_CFG).
    """

    def __init__(
        self,
        num_classes: int = 1,
        input_channels: int = 3,
        c_list=None,
        split_att: str = "fc",
        bridge: bool = True,
        descriptor_type: str = "contrast",
        kernel_size: int = 3,
        inject_target: str = "delta",
        placement: str = "hero",
        use_e6: bool = False,
        alpha_init: float = 0.01,
        wdelta_std: float = 1e-3,
    ):
        super().__init__()
        c_list = list(c_list) if c_list is not None else list(DEFAULT_C_LIST)
        self.bridge = bridge
        self.lc_vss_stages: Set[str] = resolve_placement_stages(placement, use_e6=use_e6)

        def pvm_slot(stage: str, input_dim: int, output_dim: int) -> LC_PVMLayer:
            is_lc = stage in self.lc_vss_stages
            return LC_PVMLayer(
                input_dim=input_dim,
                output_dim=output_dim,
                descriptor_type=(descriptor_type if is_lc else "none"),
                kernel_size=kernel_size,
                inject_target=inject_target,
                alpha_init=alpha_init,
                wdelta_std=wdelta_std,
            )

        self.encoder1 = nn.Sequential(nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1))
        self.encoder2 = nn.Sequential(nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1))
        self.encoder3 = nn.Sequential(nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1))
        self.encoder4 = nn.Sequential(pvm_slot("E4", c_list[2], c_list[3]))
        self.encoder5 = nn.Sequential(pvm_slot("E5", c_list[3], c_list[4]))
        self.encoder6 = nn.Sequential(pvm_slot("E6", c_list[4], c_list[5]))

        if bridge:
            self.scab = SCAttBridge(c_list, split_att)

        self.decoder1 = nn.Sequential(pvm_slot("D5", c_list[5], c_list[4]))
        self.decoder2 = nn.Sequential(pvm_slot("D4", c_list[4], c_list[3]))
        self.decoder3 = nn.Sequential(pvm_slot("D3", c_list[3], c_list[2]))  # plain PVM always -- see module docstring
        self.decoder4 = nn.Sequential(nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1))
        self.decoder5 = nn.Sequential(nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1))

        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out

        if self.bridge:
            t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)

        out = F.gelu(self.encoder6(out))

        out5 = F.gelu(self.dbn1(self.decoder1(out)))
        out5 = torch.add(out5, t5)

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode="bilinear", align_corners=True))
        out4 = torch.add(out4, t4)

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode="bilinear", align_corners=True))
        out3 = torch.add(out3, t3)

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode="bilinear", align_corners=True))
        out2 = torch.add(out2, t2)

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode="bilinear", align_corners=True))
        out1 = torch.add(out1, t1)

        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode="bilinear", align_corners=True)
        return out0  # raw logits -- unlike the vendored model, no final sigmoid (see lcmunet/adapters.py)

    def alpha_by_stage(self) -> Dict[str, float]:
        """Per-stage alpha values (methodology section 11.3 / engine alpha logging)."""
        stage_modules = {"E4": self.encoder4[0], "E5": self.encoder5[0], "E6": self.encoder6[0],
                          "D5": self.decoder1[0], "D4": self.decoder2[0], "D3": self.decoder3[0]}
        return {name: float(mod.alpha.detach().cpu().item()) for name, mod in stage_modules.items() if mod.alpha is not None}
