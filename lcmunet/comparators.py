"""Thin loaders for the vendored comparator models (methodology section 9):
MALUNet and EGE-UNet. Same policy as lcmunet/backbone.py -- never edit
third_party/ in place, only import + wrap from here.

Both vendored models hard-code a final sigmoid in forward() (see each
model's third_party/*/VENDORED.md "Integration notes"); callers must wrap
the returned module in lcmunet.adapters.LogitsAdapter before using it with
this project's logits-based loss/metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_THIRD_PARTY = Path(__file__).resolve().parent.parent / "third_party"
MALUNET_DIR = _THIRD_PARTY / "MALUNet"
EGEUNET_DIR = _THIRD_PARTY / "EGE-UNet"

# Methodology §4-style channel widths; MALUNet/EGE-UNet's own default is the
# same [8,16,24,32,48,64] family (shared convention across this model lineage).
DEFAULT_C_LIST = [8, 16, 24, 32, 48, 64]


def _ensure_on_path(vendored_dir: Path) -> None:
    if not vendored_dir.is_dir():
        raise FileNotFoundError(f"Vendored model not found at {vendored_dir}. Did third_party/ get committed/pulled?")
    if str(vendored_dir) not in sys.path:
        sys.path.insert(0, str(vendored_dir))


def load_malunet(**kwargs: Any):
    """Import and instantiate the vendored, unmodified MALUNet. CPU-runnable
    (no mamba-ssm dependency)."""
    _ensure_on_path(MALUNET_DIR)
    from models.malunet import MALUNet  # type: ignore

    kwargs.setdefault("c_list", list(DEFAULT_C_LIST))
    return MALUNet(**kwargs)


def load_egeunet(**kwargs: Any):
    """Import and instantiate the vendored, unmodified EGEUNet. CPU-runnable
    (no mamba-ssm dependency).

    gt_ds defaults to True -- NOT an arbitrary choice: gt_ds=False is a
    genuine bug in the upstream repo at the pinned commit (the
    group_aggregation_bridge call omits a required `mask` argument in that
    branch). See third_party/EGE-UNet/VENDORED.md. forward() returns a
    (aux_outputs_tuple, main_output) pair when gt_ds=True;
    lcmunet.adapters.LogitsAdapter already handles that by taking the last
    element.
    """
    _ensure_on_path(EGEUNET_DIR)
    from models.egeunet import EGEUNet  # type: ignore

    kwargs.setdefault("c_list", list(DEFAULT_C_LIST))
    kwargs.setdefault("gt_ds", True)
    return EGEUNet(**kwargs)
