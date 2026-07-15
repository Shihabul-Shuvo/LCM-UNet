"""Project path resolution and layout.

Google Drive (DRIVE_ROOT) is the persistence layer. Nothing important is
written only to ephemeral Colab /content/. Tree creation is idempotent:
calling get_paths() repeatedly never duplicates or clobbers existing data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_DRIVE_ROOT_COLAB = "/content/drive/MyDrive/LCM-UNet/"

# Used when neither an explicit root nor DRIVE_ROOT env var is given and we
# are not running inside Colab (e.g. local CPU development/tests).
LOCAL_FALLBACK_DIRNAME = ".lcmunet_local_root"

SUBDIRS = (
    "data_raw",
    "data",
    "splits",
    "checkpoints",
    "results",
    "logs",
    "figures",
    "secrets",
    "configs",
)


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_raw: Path
    data: Path
    splits: Path
    checkpoints: Path
    results: Path
    logs: Path
    figures: Path
    secrets: Path
    configs: Path


def resolve_drive_root(override: Optional[str | Path] = None) -> Path:
    """Resolve DRIVE_ROOT with priority: explicit override > DRIVE_ROOT env > Colab default > local fallback."""
    if override is not None:
        return Path(override)
    env = os.environ.get("DRIVE_ROOT")
    if env:
        return Path(env)
    if Path("/content").is_dir():
        return Path(DEFAULT_DRIVE_ROOT_COLAB)
    return Path.cwd() / LOCAL_FALLBACK_DIRNAME


def get_paths(root: Optional[str | Path] = None, create: bool = True) -> ProjectPaths:
    """Resolve DRIVE_ROOT and (optionally) create the project tree. Idempotent."""
    root_path = Path(root) if root is not None else resolve_drive_root()
    sub = {name: root_path / name for name in SUBDIRS}

    if create:
        root_path.mkdir(parents=True, exist_ok=True)
        for p in sub.values():
            p.mkdir(parents=True, exist_ok=True)

    return ProjectPaths(root=root_path, **sub)
