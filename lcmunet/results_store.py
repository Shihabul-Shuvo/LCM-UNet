"""Append-only results table with idempotent upsert on (config_id, seed).

Re-running the same (config_id, seed) — e.g. after a Colab disconnect and
resume — replaces that row in place. It never produces duplicate rows.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Dict

import pandas as pd

COLUMNS = [
    "config_id",
    "model_name",
    "dataset",
    "seed",
    "split_file",
    "dsc",
    "miou",
    "sensitivity",
    "specificity",
    "accuracy",
    "hd95",
    "assd",
    "params_M",
    "gflops",
    "fps_b1",
    "fps_b8",
    "peak_mem_MB",
    "gpu_name",
    "scan_impl",
    "timestamp",
    "notes",
]

_REQUIRED = ("config_id", "seed")


def _results_path(results_dir: str | Path) -> Path:
    return Path(results_dir) / "results.csv"


def load_results(results_dir: str | Path) -> pd.DataFrame:
    """Load results/results.csv as a DataFrame. Returns an empty, correctly-columned frame if absent."""
    path = _results_path(results_dir)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[COLUMNS]


def upsert_result(results_dir: str | Path, row: Dict[str, Any]) -> pd.DataFrame:
    """Insert or replace the row for (row['config_id'], row['seed']). Never duplicates. Never estimates a metric — the caller must have actually measured it."""
    missing = [k for k in _REQUIRED if k not in row or row[k] is None]
    if missing:
        raise ValueError(f"result row missing required keys: {missing}")

    full_row = {col: row.get(col) for col in COLUMNS}
    if not full_row.get("timestamp"):
        full_row["timestamp"] = _dt.datetime.now(_dt.timezone.utc).isoformat()

    df = load_results(results_dir)
    if len(df):
        keep_mask = ~(
            (df["config_id"] == full_row["config_id"])
            & (df["seed"] == full_row["seed"])
        )
        df = df[keep_mask]

    new_row_df = pd.DataFrame([full_row], columns=COLUMNS)
    out = pd.concat([df, new_row_df], ignore_index=True)[COLUMNS]

    path = _results_path(results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".csv.tmp")
    out.to_csv(tmp_path, index=False)
    tmp_path.replace(path)

    return out
