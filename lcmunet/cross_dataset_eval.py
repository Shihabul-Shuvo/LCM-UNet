"""Cross-dataset generalisation evaluation (methodology section 7,
"required"; section 14 week 5): train on Kvasir-SEG / test on CVC-ClinicDB,
and the reverse -- hero + baseline only (this prompt's Phase-2 spec).

This is EVALUATION over an ALREADY-TRAINED in-domain checkpoint (the
Phase-2 headline run for (model_name, train_dataset, seed)), NOT a new
training config -- lcmunet.engine.run_one only ever trains+tests on the
SAME dataset, so there is no RunConfig field for "train on A, test on B"
(flagged as a gap when Phase-2 configs were first authored -- see
lcmunet/experiment_matrix.py's original module docstring). This module
resolves that gap by reusing lcmunet.data.loaders.build_cross_dataset_loader
(already existed, unused until now) for the test-set loader.

Results go to results/cross_dataset_results.csv -- a SEPARATE table from
results/results.csv, since these rows are not keyed by a trained config_id
(there is no "cross-dataset RunConfig", only an in-domain checkpoint being
re-evaluated on a different test set).
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

from lcmunet.data.loaders import build_cross_dataset_loader
from lcmunet.data.splits import CROSS_DATASET_PAIRS
from lcmunet.metrics import evaluate

CROSS_DATASET_COLUMNS = [
    "model_name",
    "descriptor_type",
    "train_dataset",
    "test_dataset",
    "seed",
    "dsc",
    "miou",
    "sensitivity",
    "specificity",
    "accuracy",
    "hd95",
    "assd",
    "timestamp",
]

_KEY_COLUMNS = ("model_name", "train_dataset", "test_dataset", "seed")


def _results_path(results_dir: str | Path) -> Path:
    return Path(results_dir) / "cross_dataset_results.csv"


def load_cross_dataset_results(results_dir: str | Path) -> pd.DataFrame:
    path = _results_path(results_dir)
    if not path.exists():
        return pd.DataFrame(columns=CROSS_DATASET_COLUMNS)
    df = pd.read_csv(path)
    for col in CROSS_DATASET_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[CROSS_DATASET_COLUMNS]


def _upsert(results_dir: str | Path, row: Dict[str, Any]) -> None:
    """Idempotent upsert keyed by (model_name, train_dataset, test_dataset,
    seed) -- same never-duplicates guarantee as results_store.upsert_result,
    so re-running a cross-dataset evaluation after a Colab disconnect just
    replaces the row rather than appending a duplicate."""
    df = load_cross_dataset_results(results_dir)
    if len(df):
        keep_mask = ~(
            (df["model_name"] == row["model_name"])
            & (df["train_dataset"] == row["train_dataset"])
            & (df["test_dataset"] == row["test_dataset"])
            & (df["seed"] == row["seed"])
        )
        df = df[keep_mask]

    full_row = {col: row.get(col) for col in CROSS_DATASET_COLUMNS}
    if not full_row.get("timestamp"):
        full_row["timestamp"] = _dt.datetime.now(_dt.timezone.utc).isoformat()

    out = pd.concat([df, pd.DataFrame([full_row], columns=CROSS_DATASET_COLUMNS)], ignore_index=True)[CROSS_DATASET_COLUMNS]

    path = _results_path(results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".csv.tmp")
    out.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def _load_checkpoint_state_dict(paths, config) -> dict:
    from lcmunet.engine import checkpoint_dir

    ckpt_path = checkpoint_dir(paths, config) / "best.pt"
    if not ckpt_path.is_file():
        raise RuntimeError(
            f"No trained checkpoint at {ckpt_path} for config_id={config.config_id} "
            f"({config.model_name} on {config.dataset}, seed={config.seed}). The "
            "in-domain Phase-2 headline run must complete before cross-dataset "
            "evaluation can reuse its checkpoint."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt["model"]


def evaluate_cross_dataset_row(paths, in_domain_config, test_dataset: str) -> Dict[str, Any]:
    """Loads in_domain_config's ALREADY-TRAINED checkpoint (in_domain_config.dataset
    is the train_dataset), evaluates it on test_dataset's TEST split (via
    build_cross_dataset_loader), and returns a result row -- does NOT write
    it (see run_cross_dataset_suite, which upserts one row at a time so a
    partial suite is still resumable).
    """
    from lcmunet.engine import build_model

    model = build_model(in_domain_config)
    model.load_state_dict(_load_checkpoint_state_dict(paths, in_domain_config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    _train_loader, test_loader = build_cross_dataset_loader(
        in_domain_config.dataset,
        test_dataset,
        paths,
        batch_size=in_domain_config.batch_size,
        input_size=in_domain_config.input_size,
        seed=in_domain_config.seed,
    )
    metrics = evaluate(model, test_loader, device, boundary=True)

    return {
        "model_name": in_domain_config.model_name,
        "descriptor_type": in_domain_config.model_cfg.get("descriptor_type"),
        "train_dataset": in_domain_config.dataset,
        "test_dataset": test_dataset,
        "seed": in_domain_config.seed,
        "dsc": metrics["dsc"],
        "miou": metrics["miou"],
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "accuracy": metrics["accuracy"],
        "hd95": metrics["hd95"],
        "assd": metrics["assd"],
    }


def run_cross_dataset_suite(paths, hero_descriptor_type: str, seeds: Optional[List[int]] = None) -> pd.DataFrame:
    """hero + baseline, both directions (Kvasir<->CVC, lcmunet.data.splits.
    CROSS_DATASET_PAIRS), for every seed in `seeds` (defaults to methodology
    section 6's 5 headline seeds -- the same seeds Phase-2's headline
    in-domain runs used, so a matching checkpoint is expected to exist for
    each). Upserts every row into results/cross_dataset_results.csv
    one at a time (resumable: a session that dies partway through just
    re-evaluates the remaining pairs on the next call, never re-writing
    already-recorded rows incorrectly) and returns the full table.
    """
    from lcmunet import experiment_matrix as em

    seeds = seeds if seeds is not None else list(em.HEADLINE_SEEDS)
    scan_impl, _source = em.resolve_scan_impl(paths)
    headline_rows = dict(em.build_phase2_headline(scan_impl, hero_descriptor_type=hero_descriptor_type))

    for train_dataset, test_dataset in CROSS_DATASET_PAIRS:
        for model_role in ("baseline_pvm", "lc_ss2d_hero"):
            for seed in seeds:
                role = f"phase2_headline_{train_dataset}_{model_role}_seed{seed}"
                config = headline_rows[role]
                row = evaluate_cross_dataset_row(paths, config, test_dataset)
                _upsert(paths.results, row)

    return load_cross_dataset_results(paths.results)
