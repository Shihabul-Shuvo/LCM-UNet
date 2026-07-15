"""Thin loader for the vendored UltraLight VM-UNet backbone.

third_party/UltraLight-VM-UNet is an unmodified vendored snapshot (MIT
licensed; see third_party/UltraLight-VM-UNet/VENDORED.md for the pinned
commit). We never edit it in place — this module only puts it on sys.path
and imports its public class. Any LC-SS2D injection wraps or subclasses
these modules from lcmunet/, per methodology §5 ("tensor-level, no kernel
surgery").

NOTE: models/UltraLight_VM_UNet.py does `from mamba_ssm import Mamba` at
module level, so `load_ultralight_vmunet()` raises ImportError on any
machine without a working mamba-ssm install — including this local CPU
dev machine. That's expected: the mamba-ssm CUDA build is a GPU gate the
user runs and reports from Colab (GLOBAL RULES rule 5), not something this
module can verify.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
THIRD_PARTY_DIR = _THIS_DIR.parent / "third_party" / "UltraLight-VM-UNet"

# Methodology §4: channel widths [8,16,24,32,48,64] (matches the vendored
# repo's own default configs/config_setting.py c_list).
DEFAULT_C_LIST = [8, 16, 24, 32, 48, 64]


def load_ultralight_vmunet(**kwargs: Any):
    """Import and instantiate the vendored, unmodified UltraLight_VM_UNet."""
    if not THIRD_PARTY_DIR.is_dir():
        raise FileNotFoundError(
            f"Vendored backbone not found at {THIRD_PARTY_DIR}. "
            "Did third_party/UltraLight-VM-UNet get committed/pulled?"
        )
    if str(THIRD_PARTY_DIR) not in sys.path:
        sys.path.insert(0, str(THIRD_PARTY_DIR))

    try:
        from models.UltraLight_VM_UNet import UltraLight_VM_UNet  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Could not import the vendored UltraLight_VM_UNet from "
            f"{THIRD_PARTY_DIR} (it requires mamba-ssm, a GPU-only build). "
            "This is expected on a CPU-only machine; run/verify in Colab "
            f"instead. Original error: {exc!r}"
        ) from exc

    kwargs.setdefault("c_list", list(DEFAULT_C_LIST))
    return UltraLight_VM_UNet(**kwargs)
