"""Global determinism. Every headline run must be reproducible from a fixed seed."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Fix python/numpy/torch/cuda RNGs and (optionally) force deterministic kernels.

    Call once at the start of every run, before model construction. The seed
    used must be recorded in the run's config (see config.py) and results row
    (see results_store.py) — never leave it implicit.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # older torch versions do not accept warn_only
            torch.use_deterministic_algorithms(True)
