"""Collects environment facts (torch/CUDA/GPU/scan-impl) into results/env.json.

Every downstream efficiency number depends on which selective-scan
implementation ran, so env.json is the single recorded source of truth for
that decision (SCAN_IMPL, from lcmunet/scan.py) alongside the hardware it
ran on. GPU facts are only ever true facts when this runs on the real
target machine — running locally on a CPU-only dev box still produces a
valid, honest env.json (cuda_available=False, scan_impl="ref"); it does not
fabricate a GPU gate result (GLOBAL RULES rule 5).
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _git_commit(repo_root: str | Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def collect_env_info(repo_root: str | Path = ".") -> Dict[str, Any]:
    from lcmunet import scan as scan_module

    cuda_available = torch.cuda.is_available()
    info: Dict[str, Any] = {
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "gpu_free_vram_gb": None,
        "gpu_total_vram_gb": None,
        "scan_impl": scan_module.SCAN_IMPL,
        "mamba_ssm_importable": scan_module._cuda_selective_scan_fn is not None,
        "mamba_ssm_import_error": (
            repr(scan_module._CUDA_IMPORT_ERROR) if scan_module._CUDA_IMPORT_ERROR else None
        ),
        "repo_commit": _git_commit(repo_root),
    }
    if cuda_available:
        free_b, total_b = torch.cuda.mem_get_info(0)
        info["gpu_free_vram_gb"] = round(free_b / 1024**3, 2)
        info["gpu_total_vram_gb"] = round(total_b / 1024**3, 2)
    return info


def write_env_json(results_dir: str | Path, repo_root: str | Path = ".") -> Path:
    """Refreshes every collect_env_info() key (SCAN_IMPL, GPU, etc.) with
    current values, but -- via update_env_json -- preserves any extra key
    written by something else (e.g. "isic2017_source", written by
    lcmunet.data.download.ensure_isic2017) regardless of which one ran
    first in a given session.
    """
    info = collect_env_info(repo_root)
    return update_env_json(results_dir, info)


def update_env_json(results_dir: str | Path, updates: Dict[str, Any]) -> Path:
    """Merges `updates` into the existing results/env.json (creating it if
    absent) without clobbering fields written by write_env_json (e.g.
    SCAN_IMPL) -- for provenance facts recorded outside the 01_env.ipynb
    GPU-gate flow (e.g. which ISIC2017 source -- S3 or Kaggle -- succeeded).
    """
    path = Path(results_dir) / "env.json"
    data: Dict[str, Any] = {}
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    data.update(updates)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return path
