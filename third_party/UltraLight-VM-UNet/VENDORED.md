# Vendoring notice

This directory is an **unmodified snapshot** of:

- **Source:** https://github.com/wurenkai/UltraLight-VM-UNet
- **Pinned commit:** `27e44181b3cd5b0b2ab3ad1e3ddb8cad67368fbd` (`main`, fetched 2026-07-09)
- **License:** MIT (c) 2025 Renkai Wu — see `LICENSE` in this directory, kept intact.
- **Paper:** Wu et al., "UltraLight VM-UNet: Parallel Vision Mamba Significantly
  Reduces Parameters for Skin Lesion Segmentation," *Patterns* (Cell), 2025.
  https://doi.org/10.1016/j.patter.2025.101298

## Why vendored (not a git submodule, not cloned at Colab runtime)

Pinning an exact commit as committed source — rather than tracking upstream
`main` — means the baseline code cannot silently drift mid-experiment. This
matters directly for the Fairness rule (same split/seed/preprocessing/
**hardware/code** for every compared model): if `wurenkai/UltraLight-VM-UNet`
changed upstream between our baseline run and our GLGF/LC-SS2D runs, the
"reproduced baseline" comparison would be invalid. The whole tree is also
already covered by `git pull` of *this* repo in Colab, so no separate
third-party clone step is needed there.

## Rule: do not edit in place

Nothing under this directory is ever edited directly. `lcmunet/backbone.py`
imports it as-is and any LC-SS2D injection wraps or subclasses its modules
(e.g. `PVMLayer`, `UltraLight_VM_UNet`) from `lcmunet/`, per
`docs/LCM-UNet_FINAL_methodology_v4.1.md` §5 ("tensor-level, no kernel
surgery... wrap, don't edit").

## Architecture reality check (see agent report for full detail)

`models/UltraLight_VM_UNet.py` builds `PVMLayer` from `mamba_ssm.Mamba`
(a 1D causal scan), splitting each stage's C channels into 4 **channel
groups** (`d_model=input_dim//4`) that share ONE `Mamba` instance, scanning
the SAME single flatten order for every group. There is no 4-directional
(→/←/↓/↑) 2D cross-scan anywhere in this file — that is a VMamba/SS2D
concept, not present in this backbone. This affects how
`docs/LCM-UNet_FINAL_methodology_v4.1.md` §3.1's "for each scan direction
k ∈ {→,←,↓,↑}" language maps onto the real code and needs a decision before
LC-VSS is implemented (flagged to the user/supervisor; not resolved here per
GLOBAL RULES Rule 1).
