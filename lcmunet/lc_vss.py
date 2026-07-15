"""LC-VSS / LC-SS2D: the core contribution (methodology sections 3, 5.2-5.4).

Injects a bounded, learnable modulation into the Mamba selective step-size
Delta, conditioned on a local 2D-neighbourhood descriptor. This module
reimplements mamba_ssm.Mamba's own documented non-fused ("slow path")
forward computation directly (mirrors mamba_ssm/modules/mamba_simple.py's
Mamba.forward() line-by-line for every part that must stay unchanged) --
it never imports mamba_ssm itself, so it runs on CPU without a GPU or the
mamba-ssm package installed, using lcmunet.scan.selective_scan (the ONE
locked selective-scan primitive, methodology section 5.5 fairness rule) --
the same primitive every other model in this project must use.

Token order, B, C, D, and out_proj are UNCHANGED from stock Mamba. The only
new computation is the modulation added to dt (or, for inject_target=
'input', to the scanned signal x) before the scan call -- never the fused
mamba_inner_fn path (Step-0 audit item 3: that path hides dts inside the
kernel, which is exactly what must stay visible for this injection).

SHARED-CORE NOTE. The vendored PVMLayer calls ONE shared `self.mamba`
instance on all 4 channel-group chunks (same weights every time) --
"PVM operator" is explicitly listed as inherited/unchanged in methodology
section 4. So LCVSSMambaCore (the in_proj/conv1d/x_proj/dt_proj/A_log/D/
out_proj weights) is likewise ONE shared module per LC-VSS block; only
W_delta is instantiated once per PVM group (methodology section 5.3: "Each
PVM channel group has its own W_delta -- not shared across groups"), and
alpha is one learnable scalar per LC-VSS block (section 3.3, Gate 0 item 6:
"self.alpha" singular). This is also what makes "descriptor_type='none'
reproduces baseline forward numerically" possible at all: with independent
per-group Mamba weights instead of one shared set, the two could never
match even with the injection disabled.

ARCHITECTURE NOTE -- read before touching this file. The vendored,
unmodified UltraLight_VM_UNet's PVMLayer (third_party/UltraLight-VM-UNet/
models/UltraLight_VM_UNet.py) does NOT implement VMamba-style 4-directional
(up/down/left/right) SS2D cross-scanning. It implements 4-way CHANNEL-GROUP
parallelism: one shared 1D Mamba module applied independently to 4 slices
of C/4 channels each, all using the SAME single flatten order (a plain
reshape+transpose of the 2D grid). There is no cross_scan utility and no
per-direction dt_projs anywhere in the vendored code (confirmed by grep in
a prior session).

Methodology section 5.3 already describes this operationally: "UltraLight
PVM splits C into num_groups parallel...groups... slice M into num_groups
parts along the channel dimension; feed each slice to that group's
W_delta... flatten_k applied to each group's M slice uses the same
permutation as its token group." That is, section 5.3's own operational
language maps "k" onto the channel-GROUP index for this backbone, not a
spatial direction. Section 3.1's abstract "for each scan direction k in
{->, <-, v, ^}" is the mechanism's general/conceptual framing (written to
cover genuine SS2D backbones in general); with only one real flatten order
in PVM, "shared across the four scan directions" (section 3.4) is trivially
satisfied (a set of size one), while "each group has its own W_delta, not
shared across groups" (section 5.3) is the active, meaningful constraint --
exactly what this module implements.

This resolves what would otherwise be a direct contradiction between
section 3.4 and section 5.3 if "direction" and "group" were the same axis.
Flagged explicitly, not silently assumed -- see the agent report for the
prompt that added this file for the full reasoning and the alternative
reading, in case this call needs correcting.
"""

from __future__ import annotations

import math
from typing import List, Optional, Set

import torch
import torch.nn as nn
from einops import rearrange

from lcmunet.scan import selective_scan

# Fixed by the vendored PVMLayer's own torch.chunk(x_norm, 4, dim=2).
PVM_NUM_GROUPS = 4

# methodology section 4.
HERO_PLACEMENT_STAGES = ("E4", "E5", "D5", "D4")
STAGE_CHANNELS = {"E1": 8, "E2": 16, "E3": 24, "E4": 32, "E5": 48, "E6": 64, "D5": 48, "D4": 32, "D3": 24, "D2": 16, "D1": 8}

VALID_DESCRIPTOR_TYPES = ("contrast", "plain", "none")
VALID_INJECT_TARGETS = ("delta", "input")


def resolve_placement_stages(placement: str, use_e6: bool = False) -> Set[str]:
    """methodology section 10.2 Place ablation: P0 / E4E5 / hero / +E6 / E4D4."""
    mapping = {
        "P0": (),
        "E4E5": ("E4", "E5"),
        "hero": HERO_PLACEMENT_STAGES,
        "+E6": HERO_PLACEMENT_STAGES + ("E6",),
        "E4D4": ("E4", "D4"),
    }
    if placement not in mapping:
        raise ValueError(f"unknown placement: {placement!r} (expected one of {sorted(mapping)})")
    stages = set(mapping[placement])
    if use_e6:
        stages.add("E6")
    return stages


class LocalDescriptor(nn.Module):
    """M = DWConv_k(Xn) [- Xn if contrast] (methodology section 3.2).

    descriptor_type: 'contrast' (default, local-contrast/high-pass) |
    'plain' (DWConv3x3(Xn) only, Desc ablation) | 'none' (baseline, no
    modulation -- forward() returns None and the conv is never constructed).

    kernel_size: 3 (default, proposed) | 1 (Ablation C capacity-matched
    control -- point-wise, no neighbourhood access, same W_delta downstream).
    """

    def __init__(self, channels: int, descriptor_type: str = "contrast", kernel_size: int = 3, std: float = 1e-3):
        super().__init__()
        if descriptor_type not in VALID_DESCRIPTOR_TYPES:
            raise ValueError(f"descriptor_type must be one of {VALID_DESCRIPTOR_TYPES}, got {descriptor_type!r}")
        self.descriptor_type = descriptor_type
        self.channels = channels
        self.kernel_size = kernel_size

        if descriptor_type == "none":
            self.dw_conv = None
            return

        padding = kernel_size // 2
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels, bias=False)
        nn.init.normal_(self.dw_conv.weight, std=std)

    def forward(self, Xn: torch.Tensor) -> Optional[torch.Tensor]:
        """Xn: (B, C, H, W) -> M: (B, C, H, W), or None if descriptor_type=='none'."""
        if self.dw_conv is None:
            return None
        conv_out = self.dw_conv(Xn)
        if self.descriptor_type == "contrast":
            return conv_out - Xn
        return conv_out  # 'plain'


def pvm_flatten(x: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, L, C): the SAME flatten PVMLayer.forward() does
    (x.reshape(B, C, n_tokens).transpose(-1, -2)) -- reused verbatim, not a
    new cross_scan utility, so the descriptor is guaranteed to align to
    tokens with the identical permutation (methodology section 5.2)."""
    b, c = x.shape[:2]
    n_tokens = x.shape[2:].numel()
    return x.reshape(b, c, n_tokens).transpose(-1, -2)


class LCVSSMambaCore(nn.Module):
    """The Mamba-equivalent weights SHARED across all PVM groups in one
    LC-VSS block (in_proj/conv1d/x_proj/dt_proj/A_log/D/out_proj) -- mirrors
    the vendored PVMLayer's single shared `self.mamba` instance.

    A faithful drop-in for `mamba_ssm.Mamba(d_model, d_state, d_conv,
    expand)`: identical submodules, identical initialisation (mirrors
    mamba_ssm.modules.mamba_simple.Mamba.__init__ exactly), identical
    forward math when called with no modulation. w_delta/alpha are NOT
    owned here -- they are per-group (w_delta) / per-block (alpha) and
    passed into forward() by the caller (LC_PVMLayer), since this core's
    weights are shared across groups but the modulation is not.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: "str | int" = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # ---- identical to mamba_ssm.modules.mamba_simple.Mamba.__init__ ----
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1, bias=conv_bias
        )
        self.activation = "silu"
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(dt_init)

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        # ---- end identical-to-stock-Mamba section ----

    def forward(
        self,
        hidden_states: torch.Tensor,
        m_k: Optional[torch.Tensor] = None,
        w_delta: Optional[nn.Linear] = None,
        alpha: Optional[torch.Tensor] = None,
        inject_target: str = "delta",
    ) -> torch.Tensor:
        """hidden_states: (B, L, d_model) -- SAME flatten/order as stock PVM.
        m_k: (B, L, descriptor_channels) or None -- local descriptor,
        flattened with the SAME permutation as hidden_states (caller's
        responsibility -- see pvm_flatten above; methodology section 5.2).
        w_delta/alpha: this group's W_delta and this block's alpha (None ->
        no modulation, mathematically identical to stock Mamba).
        Returns: (B, L, d_model).
        """
        batch, seqlen, _ = hidden_states.shape

        xz = rearrange(self.in_proj(hidden_states), "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)  # each (B, d_inner, L) -- UNCHANGED split

        x = self.act(self.conv1d(x)[..., :seqlen])  # (B, d_inner, L) -- UNCHANGED scanned input path

        inject = w_delta is not None and alpha is not None and m_k is not None
        if inject and inject_target == "input":
            mod = alpha * torch.tanh(w_delta(m_k))  # (B, L, d_inner)
            x = x + rearrange(mod, "b l d -> b d l")  # Ablation D: modulate u, not dt

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        if inject and inject_target == "delta":
            m_flat = rearrange(m_k, "b l d -> (b l) d")
            dt = dt + alpha * torch.tanh(w_delta(m_flat))  # THE contribution (hero, section 3.1)

        dt = self.dt_proj.weight @ dt.t()  # (d_inner, B*L) -- dt_proj weight UNCHANGED
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)

        A = -torch.exp(self.A_log.float())  # UNCHANGED
        B_ssm = rearrange(B_ssm, "(b l) dstate -> b dstate l", l=seqlen).contiguous()  # UNCHANGED
        C_ssm = rearrange(C_ssm, "(b l) dstate -> b dstate l", l=seqlen).contiguous()  # UNCHANGED

        y = selective_scan(
            x, dt, A, B_ssm, C_ssm, self.D.float(),
            z=z, delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
        )
        y = rearrange(y, "b d l -> b l d")
        return self.out_proj(y)  # UNCHANGED


class LC_PVMLayer(nn.Module):
    """Drop-in replacement for PVMLayer (third_party/UltraLight-VM-UNet/
    models/UltraLight_VM_UNet.py), reusing its exact norm/chunk/flatten/proj
    structure (methodology section 4: "PVM operator" is inherited), adding
    LC-SS2D Delta-conditioning (methodology sections 3, 5.2-5.4).

    When descriptor_type='none', this reduces to stock PVMLayer exactly
    (same shared Mamba-equivalent core, same double-LayerNorm, same
    residual/proj structure) -- see tests/test_lc_vss.py for the numerical
    equivalence check.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_groups: int = PVM_NUM_GROUPS,
        descriptor_type: str = "contrast",
        kernel_size: int = 3,
        inject_target: str = "delta",
        alpha_init: float = 0.01,
        wdelta_std: float = 1e-3,
    ):
        super().__init__()
        if input_dim % num_groups != 0:
            raise ValueError(f"input_dim={input_dim} must be divisible by num_groups={num_groups}")
        if descriptor_type not in VALID_DESCRIPTOR_TYPES:
            raise ValueError(f"descriptor_type must be one of {VALID_DESCRIPTOR_TYPES}, got {descriptor_type!r}")
        if inject_target not in VALID_INJECT_TARGETS:
            raise ValueError(f"inject_target must be one of {VALID_INJECT_TARGETS}, got {inject_target!r}")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_groups = num_groups
        self.c_group = input_dim // num_groups
        self.descriptor_type = descriptor_type
        self.inject_target = inject_target

        self.norm = nn.LayerNorm(input_dim)
        self.core = LCVSSMambaCore(d_model=self.c_group, d_state=d_state, d_conv=d_conv, expand=expand)
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

        self.descriptor = LocalDescriptor(input_dim, descriptor_type=descriptor_type, kernel_size=kernel_size, std=wdelta_std)

        self.alpha: Optional[nn.Parameter] = None
        self.w_deltas: Optional[nn.ModuleList] = None
        if descriptor_type != "none":
            out_features = self.core.d_inner if inject_target == "input" else self.core.dt_rank
            w_deltas: List[nn.Linear] = []
            for _ in range(num_groups):
                w_delta = nn.Linear(self.c_group, out_features, bias=False)
                nn.init.normal_(w_delta.weight, std=wdelta_std)
                w_deltas.append(w_delta)
            self.w_deltas = nn.ModuleList(w_deltas)
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        batch, channels = x.shape[:2]
        assert channels == self.input_dim
        img_dims = x.shape[2:]

        x_flat = pvm_flatten(x)  # (B, L, C) -- identical to stock PVMLayer
        x_norm = self.norm(x_flat)

        m_chunks: List[Optional[torch.Tensor]]
        if self.descriptor.dw_conv is not None:
            xn_2d = x_norm.transpose(-1, -2).reshape(batch, channels, *img_dims)
            m_2d = self.descriptor(xn_2d)
            m_flat = pvm_flatten(m_2d)
            m_chunks = list(torch.chunk(m_flat, self.num_groups, dim=2))
        else:
            m_chunks = [None] * self.num_groups

        x_chunks = torch.chunk(x_norm, self.num_groups, dim=2)

        outputs = []
        for i in range(self.num_groups):
            xi = x_chunks[i]
            w_delta_i = self.w_deltas[i] if self.w_deltas is not None else None
            y_i = self.core(xi, m_chunks[i], w_delta=w_delta_i, alpha=self.alpha, inject_target=self.inject_target)
            outputs.append(y_i + self.skip_scale * xi)

        x_mamba = torch.cat(outputs, dim=2)
        x_mamba = self.norm(x_mamba)  # SAME norm module, applied twice -- matches stock PVMLayer
        x_mamba = self.proj(x_mamba)
        return x_mamba.transpose(-1, -2).reshape(batch, self.output_dim, *img_dims)
