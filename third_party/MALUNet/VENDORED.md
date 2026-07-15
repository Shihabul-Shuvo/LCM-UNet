# Vendoring notice

This directory is an **unmodified snapshot** of:

- **Source:** https://github.com/JCruan519/MALUNet
- **Pinned commit:** `e184b47de99b3fda6b7880fcf3ff99b806308d2b` (`main`, fetched 2026-07-10)
- **License: NONE FOUND.** No `LICENSE` file and no license statement in
  `README.md` at the pinned commit. This is a real, unresolved gap, not an
  oversight -- flagging it explicitly rather than assuming permission.
- **Paper:** Ruan et al., "MALUNet: A Multi-Attention and Light-weight UNet
  for Skin Lesion Segmentation," IEEE BIBM 2022. https://arxiv.org/abs/2211.01784

## Why vendored anyway, given no license

Used here strictly for **local, non-distributed research reproduction** --
running the authors' own published code to compare against, on our own
machine, under our own splits -- which is standard, low-risk academic
practice (comparable to how virtually every paper that benchmarks against
a baseline runs that baseline's own public code). This is materially
different from redistributing the code.

**Before any public release of this project's code** (methodology's own
reproducibility checklist calls for "Code, configs, and trained weights
prepared for release on acceptance"), this directory needs one of:
contacting the authors for explicit permission, replacing it with a
clean-room reimplementation from the paper description only, or excluding
it from the public release while still reporting the (locally reproduced)
MALUNet numbers with a note on how they were obtained. Do not ship this
directory publicly as-is.

## Rule: do not edit in place

Wrap/subclass from `lcmunet/`, per the same policy as
`third_party/UltraLight-VM-UNet/VENDORED.md`.

## Integration note

`models/malunet.py`'s `MALUNet.forward()` ends in `return torch.sigmoid(out0)`
-- it returns probabilities, not logits. `lcmunet/adapters.py`'s
`LogitsAdapter` recovers logits (via `torch.logit`) so this is compatible
with `lcmunet/losses.py`/`lcmunet/metrics.py`, which both expect raw logits
and apply sigmoid themselves.
