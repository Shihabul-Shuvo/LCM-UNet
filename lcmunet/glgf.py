"""GLGF (Gated Late-fusion) baseline/ablation variant (CONTEXT.md; methodology
sections 1, 3.5, 10.1 Ablation A, 12).

GLGF (formerly "GS-VMUNet") is this project's PRIOR design iteration,
retained ONLY as a baseline/ablation row for Ablation A ("Baseline PVM
(reproduced) -> GLGF late-fusion -> LC-SS2D (ours)", section 10.1) -- it
proves LC-SS2D's gain comes from conditioning the scan's DYNAMICS (Delta),
not from ordinary feature-level fusion. It is explicitly NOT part of the
paper's proposed model name (section 1: "The prior late-fusion design
(GLGF) is retained only as a baseline/ablation row, not in the model name").

Mechanism distinction from LC-SS2D (section 3.5, "Why not feature fusion,
hybridisation, or scan-reordering" -- "Not late feature fusion. No parallel
feature branch; the conv output enters only Delta... (Ablation A vs GLGF)"):
GLGF runs a completely VANILLA Mamba/PVM branch (no Delta modulation, no
local descriptor inside the scan -- u_k, Delta_k, B_k, C_k all stock,
descriptor_type='none') IN PARALLEL with a separate conv feature branch,
then fuses the two branches' OUTPUT features (i.e. AFTER the scan has
already run -- feature-level, not dynamics-level) with a learned sigmoid
gate. LC-SS2D has no parallel branch at all; its conv output only ever
touches Delta, inside the scan, before selective_scan runs.

Architecture: the SAME UltraLight-shaped backbone as lcmunet.lcm_unet.LCMUNet
(same channel widths, same SCAttBridge/conv-stage/GroupNorm/GELU structure
-- ChannelAttBridge/SpatialAttBridge/SCAttBridge/_init_weights/DEFAULT_C_LIST
are imported directly from lcmunet.lcm_unet, not duplicated), with GLGFLayer
replacing LC_PVMLayer at E4/E5/D5/D4 only -- the same 4 stages LC-VSS
targets (methodology section 4), for an apples-to-apples Ablation A
comparison at identical channel widths and hardware. E6 and D3 remain plain
PVM (lcmunet.lc_vss.LC_PVMLayer, descriptor_type='none'), matching the same
"inherited, unchanged" stance documented in lcmunet/lcm_unet.py for those
two stages. GLGF is not itself ablated across placement/descriptor variants
(only LC-SS2D is), so there are no config toggles here.

Uses the SAME locked lcmunet.scan.selective_scan implementation as the
baseline and LC-SS2D (methodology section 5.5 fairness rule), via
lcmunet.lc_vss.LC_PVMLayer/LCVSSMambaCore -- CPU-runnable, no mamba-ssm
dependency, exactly like lc_ss2d.

Param count: NOT assumed here. Methodology section 5.6 claims LC-SS2D adds
fewer params than GLGF ("...less than GLGF -- no gate conv, no fused
branch"); this is a measured comparison for a later prompt (fvcore/thop),
never asserted by this module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lcmunet.lc_vss import PVM_NUM_GROUPS, LC_PVMLayer
from lcmunet.lcm_unet import DEFAULT_C_LIST, SCAttBridge, _init_weights


class GLGFLayer(nn.Module):
    """Late-fusion GATE: a vanilla Mamba/PVM branch (no LC-SS2D injection,
    stock scan) fused with a parallel conv feature branch AFTER the scan,
    via a learned sigmoid gate -- exactly the construction methodology
    section 3.5's "Not late feature fusion" sentence contrasts LC-SS2D
    against.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_groups: int = PVM_NUM_GROUPS,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Mamba branch: bit-identical to stock/baseline PVM -- descriptor_type
        #='none' means no LC-SS2D modulation anywhere (see lcmunet/lc_vss.py's
        # own numerical-equivalence guarantee for descriptor_type='none').
        self.mamba_branch = LC_PVMLayer(
            input_dim=input_dim,
            output_dim=output_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_groups=num_groups,
            descriptor_type="none",
        )

        # Conv (local) feature branch, parallel to the Mamba branch -- the
        # "fused branch" methodology section 5.6 contrasts LC-SS2D against.
        self.conv_branch = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, output_dim),
            nn.GELU(),
        )

        # Gate conv: fuses the two branches' OUTPUT features (late/feature-
        # level fusion, after the scan has already produced f_mamba) --
        # the "gate conv" methodology section 5.6 contrasts LC-SS2D against.
        self.gate_conv = nn.Conv2d(2 * output_dim, output_dim, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_mamba = self.mamba_branch(x)  # (B, output_dim, H, W) -- AFTER the (unmodified) scan
        f_conv = self.conv_branch(x)  # (B, output_dim, H, W) -- parallel conv branch, no scan involved
        gate = torch.sigmoid(self.gate_conv(torch.cat([f_mamba, f_conv], dim=1)))
        return gate * f_mamba + (1.0 - gate) * f_conv


class GLGFUNet(nn.Module):
    """UltraLight-shaped backbone (same as lcmunet.lcm_unet.LCMUNet) with
    GLGFLayer at E4/E5/D5/D4 and plain PVM elsewhere. No config toggles --
    GLGF is a single fixed baseline/ablation row (methodology section 10.1
    Ablation A), not itself ablated.
    """

    def __init__(
        self,
        num_classes: int = 1,
        input_channels: int = 3,
        c_list=None,
        split_att: str = "fc",
        bridge: bool = True,
    ):
        super().__init__()
        c_list = list(c_list) if c_list is not None else list(DEFAULT_C_LIST)
        self.bridge = bridge

        def plain_pvm(input_dim: int, output_dim: int) -> LC_PVMLayer:
            return LC_PVMLayer(input_dim=input_dim, output_dim=output_dim, descriptor_type="none")

        self.encoder1 = nn.Sequential(nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1))
        self.encoder2 = nn.Sequential(nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1))
        self.encoder3 = nn.Sequential(nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1))
        self.encoder4 = nn.Sequential(GLGFLayer(c_list[2], c_list[3]))
        self.encoder5 = nn.Sequential(GLGFLayer(c_list[3], c_list[4]))
        self.encoder6 = nn.Sequential(plain_pvm(c_list[4], c_list[5]))

        if bridge:
            self.scab = SCAttBridge(c_list, split_att)

        self.decoder1 = nn.Sequential(GLGFLayer(c_list[5], c_list[4]))
        self.decoder2 = nn.Sequential(GLGFLayer(c_list[4], c_list[3]))
        self.decoder3 = nn.Sequential(plain_pvm(c_list[3], c_list[2]))  # plain PVM always -- see lcmunet/lcm_unet.py D3 note
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
        return out0  # raw logits, no final sigmoid -- same convention as lcmunet.lcm_unet.LCMUNet
