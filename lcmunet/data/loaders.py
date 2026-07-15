"""Build (train, val, test) DataLoaders from a RunConfig (methodology
section 7). Config-driven: which ids go in which split comes from
config.split_file, resolved against DRIVE_ROOT -- never a hardcoded path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from lcmunet.config import RunConfig
from lcmunet.data.datasets import SegmentationDataset
from lcmunet.data.splits import CROSS_DATASET_PAIRS, load_split, load_split_file

SANITY_N = 8
NUM_WORKERS = 2


def _resolve_split_path(config: RunConfig, paths) -> Path:
    p = Path(config.split_file)
    return p if p.is_absolute() else Path(paths.root) / p


def build_dataloaders(
    config: RunConfig, paths, sanity: bool = False, num_workers: int = NUM_WORKERS
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Returns (train_loader, val_loader, test_loader) for config.dataset,
    using the split named in config.split_file. sanity=True truncates each
    split to SANITY_N ids and forces num_workers=0 for a fast smoke test.
    """
    split_payload = load_split_file(_resolve_split_path(config, paths))

    loaders = []
    for split_name in ("train", "val", "test"):
        ids = split_payload[split_name]
        if sanity:
            ids = ids[:SANITY_N]
        ds = SegmentationDataset(
            config.dataset,
            ids,
            paths,
            augment=(split_name == "train"),
            input_size=config.input_size,
            base_seed=config.seed,
        )
        loader = DataLoader(
            ds,
            batch_size=config.batch_size,
            shuffle=(split_name == "train"),
            num_workers=0 if sanity else num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split_name == "train"),
        )
        loaders.append(loader)
    return tuple(loaders)  # type: ignore[return-value]


def build_cross_dataset_loader(
    train_dataset: str,
    test_dataset: str,
    paths,
    batch_size: int = 8,
    input_size: int = 256,
    sanity: bool = False,
    num_workers: int = NUM_WORKERS,
    seed: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Cross-dataset generalisation (methodology section 7, required): train
    on train_dataset's TRAIN split, evaluate on test_dataset's TEST split.
    """
    if (train_dataset, test_dataset) not in CROSS_DATASET_PAIRS:
        raise ValueError(
            f"({train_dataset!r}, {test_dataset!r}) is not one of the required "
            f"cross-dataset pairs: {CROSS_DATASET_PAIRS}"
        )

    train_ids = load_split(paths.splits, train_dataset)["train"]
    test_ids = load_split(paths.splits, test_dataset)["test"]
    if sanity:
        train_ids, test_ids = train_ids[:SANITY_N], test_ids[:SANITY_N]

    train_ds = SegmentationDataset(train_dataset, train_ids, paths, augment=True, input_size=input_size, base_seed=seed)
    test_ds = SegmentationDataset(test_dataset, test_ids, paths, augment=False, input_size=input_size, base_seed=seed)

    workers = 0 if sanity else num_workers
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin)
    return train_loader, test_loader


if __name__ == "__main__":
    import argparse

    from lcmunet.paths import get_paths

    parser = argparse.ArgumentParser(description="Smoke-test DataLoaders for one dataset.")
    parser.add_argument("dataset", choices=["kvasir_seg", "cvc_clinicdb", "isic2017", "isic2018"])
    parser.add_argument("--sanity", action="store_true", help="tiny subsets, num_workers=0")
    args = parser.parse_args()

    paths = get_paths()
    config = RunConfig(
        run_name="loader_check",
        model_name="ultralight_baseline",
        dataset=args.dataset,
        seed=0,
        split_file=f"splits/{args.dataset}.json",
    )
    train_loader, val_loader, test_loader = build_dataloaders(config, paths, sanity=args.sanity)
    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        images, masks, ids = next(iter(loader))
        print(
            f"{name}: {len(loader.dataset)} samples, "
            f"images {tuple(images.shape)}, masks {tuple(masks.shape)}, "
            f"e.g. ids={list(ids[:3])}"
        )
