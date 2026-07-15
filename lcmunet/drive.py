"""Google Drive mount (Colab) with a local fallback for CPU/dev machines."""

from __future__ import annotations

from pathlib import Path

from lcmunet.paths import LOCAL_FALLBACK_DIRNAME

DRIVE_MOUNT_POINT = "/content/drive"
PROJECT_SUBPATH = "MyDrive/LCM-UNet"


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def mount_drive(
    mount_point: str = DRIVE_MOUNT_POINT, project_subpath: str = PROJECT_SUBPATH
) -> Path:
    """Mount Google Drive (in Colab) and return the LCM-UNet project root.

    On a non-Colab machine, returns a local fallback directory instead of
    raising, so the same code path works for local CPU-only development.
    """
    if is_colab():
        from google.colab import drive as _colab_drive  # type: ignore[import]

        _colab_drive.mount(mount_point)
        root = Path(mount_point) / project_subpath
    else:
        root = Path.cwd() / LOCAL_FALLBACK_DIRNAME

    root.mkdir(parents=True, exist_ok=True)
    return root
