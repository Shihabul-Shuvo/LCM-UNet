# Vendoring notice

This directory is an **unmodified snapshot** of:

- **Source:** https://github.com/JCruan519/EGE-UNet
- **Pinned commit:** `f52ba30c6bf7d0ca479c2c9d4a3cbda999f49d3a` (`main`, fetched 2026-07-10)
- **License:** Apache License 2.0 -- see `LICENSE` in this directory, kept intact.
- **Paper:** Ruan et al., "EGE-UNet: an Efficient Group Enhanced UNet for
  skin lesion segmentation," MICCAI 2023. https://arxiv.org/abs/2307.08473

## Rule: do not edit in place

Wrap/subclass from `lcmunet/`, per the same policy as
`third_party/UltraLight-VM-UNet/VENDORED.md`.

## Integration notes

- `models/egeunet.py`'s `EGEUNet.forward()` ends in `return torch.sigmoid(out0)`
  (or, when `gt_ds=True`, a tuple of five deep-supervision sigmoid outputs
  plus the main one) -- probabilities, not logits. `lcmunet/adapters.py`'s
  `LogitsAdapter` recovers logits (and takes the last element if a tuple is
  returned), so this is compatible with `lcmunet/losses.py`/
  `lcmunet/metrics.py`.
- This project builds `EGEUNet(..., gt_ds=True)` -- NOT False. `gt_ds=False`
  is a genuine bug in the upstream repo at this pinned commit:
  `group_aggregation_bridge.forward(self, xh, xl, mask)` has no default for
  `mask`, but `EGEUNet.forward()`'s `gt_ds=False` branch calls it as
  `self.GAB5(t6, t5)` (2 args) -- a guaranteed `TypeError`, confirmed by
  running it. `gt_ds=True` (the repo's own `configs/config_setting.py`
  default) is not just "deep supervision on" -- it's the only branch that
  actually supplies the `mask` argument the GAB blocks need to run at all,
  so it is required for the model to function, not a stylistic choice.
  `LogitsAdapter` takes the tuple's last element (the main output) and
  discards the 5 auxiliary deep-supervision predictions, so training still
  uses methodology section 6's uniform BCE+Dice-on-final-output loss only
  -- the auxiliary heads still run (and shape the encoder via the GAB
  blocks, since their predictions feed back in as `mask`) but never
  contribute their own loss term.
- Needs `einops` (see requirements.txt) in addition to the dependencies
  UltraLight-VM-UNet already requires (`timm`).
