"""Zero-touch Kaggle API auth, every Colab session.

CVC-ClinicDB, ISIC2018, and the ISIC2017 Kaggle FALLBACK all need a Kaggle
API token to download. Kvasir-SEG (direct wget) and the ISIC2017 S3 PRIMARY
path do not, and must never be gated on this -- so this module is called
lazily, per-dataset, from lcmunet/data/download.py's Kaggle-backed
ensure_* functions, never eagerly for the whole pipeline.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

MISSING_MESSAGE = (
    "Missing DRIVE_ROOT/secrets/kaggle.json — create a Kaggle API token "
    "(kaggle.com -> Settings -> Create New Token) and place it there, then "
    "re-run this cell."
)


class KaggleAuthMissingError(RuntimeError):
    def __init__(self, message: str = MISSING_MESSAGE):
        super().__init__(message)


def _kaggle_installed() -> bool:
    try:
        import kaggle  # noqa: F401

        return True
    except Exception:
        return False


def ensure_kaggle_auth(paths) -> Path:
    """Copies DRIVE_ROOT/secrets/kaggle.json to ~/.kaggle/kaggle.json (chmod
    600) and pip-installs the `kaggle` package if it isn't already present.

    Raises KaggleAuthMissingError with the exact required message if the
    token file isn't there -- never proceeds on a missing/guessed token.
    Idempotent: safe to call every session/every dataset.
    """
    src = Path(paths.secrets) / "kaggle.json"
    if not src.is_file():
        raise KaggleAuthMissingError()

    if not _kaggle_installed():
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)

    dst_dir = Path.home() / ".kaggle"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "kaggle.json"
    shutil.copy(src, dst)
    dst.chmod(0o600)
    return dst


def try_ensure_kaggle_auth(paths) -> Optional[Path]:
    """Non-raising wrapper for an early, informational notebook cell: attempts
    ensure_kaggle_auth(), prints and swallows KaggleAuthMissingError instead
    of crashing the whole notebook (Kvasir/ISIC2017-S3 don't need Kaggle auth
    and must be able to proceed regardless). The DOWNLOAD functions that
    actually need Kaggle (CVC, ISIC2018, ISIC2017 fallback) call the
    raising ensure_kaggle_auth() directly, so a genuinely missing token still
    fails loudly for exactly those datasets.
    """
    try:
        dst = ensure_kaggle_auth(paths)
        print(f"Kaggle auth ready: {dst}")
        return dst
    except KaggleAuthMissingError as exc:
        print(f"Kaggle auth not set up yet ({exc}). Kvasir-SEG and the ISIC2017 S3 primary path do not need it and will proceed regardless.")
        return None
