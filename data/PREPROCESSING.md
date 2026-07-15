# Preprocessing and augmentation (methodology section 7)

This file is the single source of truth for LCM-UNet's data preprocessing
and augmentation. It is written to be dropped directly into the paper's
Datasets/Implementation section. Implementation: `lcmunet/data/preprocess.py`
and `lcmunet/data/augment.py`.

**Nothing in this directory (`data/`) is ever real data.** Raw files,
processed caches, and split id-lists live under `DRIVE_ROOT/data_raw/`,
`DRIVE_ROOT/data/`, and `DRIVE_ROOT/splits/` (`lcmunet/paths.py`), never in
the git repository. This `data/` directory in the repo holds only this
documentation file.

## Preprocessing (fixed, deterministic, identical for every model and dataset)

1. **Resize.** Images and masks are resized to 256x256.
   - Images: bilinear interpolation.
   - Masks: **nearest-neighbour** interpolation, not bilinear. Bilinear
     resize blurs a hard mask boundary into intermediate grey values, which
     erodes small lesions/polyps disproportionately; nearest-neighbour keeps
     the boundary crisp.
2. **Intensity normalisation.** Images are cast to float32 and scaled to
   `[0, 1]` by dividing by 255 (per-image min-max over the fixed `[0, 255]`
   uint8 range). This is a fixed, corpus-independent transform: it is never
   computed from dataset-wide or split-wide statistics, so there is no
   channel for val/test information to leak into training through
   normalisation statistics.
3. **Mask binarisation.** Masks are converted to single-channel grayscale,
   resized (nearest-neighbour, above), then binarised to `{0, 1}` with
   threshold 127 on the 0-255 range (`pixel > 127 -> 1`). Verified against
   real Kvasir-SEG masks: although Kvasir stores masks as JPEG (lossy),
   sampled mask pixel values cluster entirely at the two extremes (0 and
   255) with zero pixels observed in the [20, 235] band across a 15-mask,
   ~7M-pixel sample -- so a fixed threshold binarises cleanly with no
   partial-coverage ambiguity.
4. This exact pipeline is applied uniformly to the baseline (reproduced
   UltraLight VM-UNet), GLGF, and LC-SS2D, and to all four datasets
   (Fairness rule, methodology section 7).

**Deviation from the UltraLight VM-UNet reference implementation.** The
vendored baseline's own preprocessing (`third_party/UltraLight-VM-UNet/loader.py`,
`dataset_normalized`) instead z-score normalises using the mean/std of the
*entire loaded image array at once* and then rescales each image to `[0,
255]` by its own post-z-score min/max -- a corpus-level statistic computed
without a documented train/val/test boundary. We deliberately do not
replicate that scheme: per-image `/255` is simpler, fully deterministic, and
cannot leak split statistics. If Gate 1 (baseline reproduction within ~0.5%
Dice) fails by a wide margin, revisiting this choice is one lever to pull,
but it is not assumed necessary.

## Augmentation (train split only)

Applied only to the training split; validation and test are never
augmented. Per methodology section 7: flips, rotation, mild scale, and mild
photometric perturbation -- **no augmentation that can erase a small
lesion** (no aggressive cropping, no cutout/erasing, no heavy elastic
deformation).

| Augmentation | Range | Notes |
|:--|:--|:--|
| Horizontal flip | p=0.5 | |
| Vertical flip | p=0.5 | |
| Rotation | uniform in [-30, 30] degrees, p=0.5 | mask uses nearest-neighbour resample to stay binary |
| Scale (zoom) | uniform in [0.9, 1.1], p=0.5 | resize-then-center-crop/pad back to 256x256; mild, not a random-crop that can cut a lesion out of frame |
| Photometric | brightness/contrast jitter, factor in [0.85, 1.15], p=0.5 | image only, never applied to the mask |

Explicitly excluded, and why: random/aggressive cropping and cutout/erasing
can remove a small polyp or lesion from the image entirely while leaving a
label that says it should be present; heavy elastic deformation can distort
lesion boundaries beyond what a clinical image plausibly looks like. Both
are excluded per methodology section 7.
