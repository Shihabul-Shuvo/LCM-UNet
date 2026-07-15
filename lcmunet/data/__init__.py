"""Data pipeline: download/detect raw archives, preprocess, split, load.

Raw files live in DRIVE_ROOT/data_raw/ (methodology section 7). Processed
tensors and split id-lists are cached under DRIVE_ROOT/data/ and
DRIVE_ROOT/splits/ so the work happens once across Colab sessions, not once
per run.
"""
