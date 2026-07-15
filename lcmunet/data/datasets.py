"""torch Dataset: wraps raw_layout (file discovery) + preprocess (resize/
normalise/binarise) + augment (train-only) for one dataset given an explicit
list of ids. Which ids belong to which split is loaders.py's job (driven by
RunConfig.split_file) -- this class just turns ids into tensors.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from lcmunet.data import raw_layout as rl
from lcmunet.data.augment import augment_pair
from lcmunet.data.preprocess import INPUT_SIZE, load_and_preprocess_pair


class SegmentationDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        ids: List[str],
        paths,
        augment: bool,
        input_size: int = INPUT_SIZE,
        base_seed: int = 0,
    ):
        pair_by_id = {p.id: p for p in rl.list_pairs(dataset, paths.data_raw)}
        missing = [i for i in ids if i not in pair_by_id]
        if missing:
            raise KeyError(
                f"{len(missing)} requested ids for {dataset} have no matching raw "
                f"file (e.g. {missing[:5]}). Raw data or split file is stale -- "
                "re-run lcmunet.data.splits.build_all_splits()."
            )
        self.dataset = dataset
        self.pairs = [pair_by_id[i] for i in ids]
        self.augment = augment
        self.input_size = input_size
        # Augmentation RNG is a pure function of (base_seed, epoch, item
        # index) -- computed fresh per __getitem__ call, never cached on
        # self. This is deliberately stateless (no persistent Generator
        # object) so it is correct across process restarts: engine.py's
        # resume must reproduce exactly what an uninterrupted run would have
        # done at the same epoch, but a resumed run always reconstructs a
        # fresh Dataset object -- any cached/lazily-seeded Generator (e.g.
        # seeded once from torch.initial_seed() and never touched again)
        # would silently restart the augmentation stream from scratch
        # instead of continuing it. Being a pure function of epoch sidesteps
        # that entirely, and also makes multi-worker DataLoaders correct
        # (each item gets its own deterministic stream regardless of which
        # worker happens to process it).
        self.base_seed = base_seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Call once per epoch (both cold-start and resumed epochs) before
        iterating the DataLoader, so augmentation varies epoch-to-epoch but
        is exactly reproducible for a given (base_seed, epoch)."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        image, mask = load_and_preprocess_pair(pair.image_path, pair.mask_path, size=self.input_size)
        if self.augment:
            rng = np.random.default_rng([self.base_seed & 0xFFFFFFFF, self.epoch, idx])
            image, mask = augment_pair(image, mask, rng)
        image_t = torch.from_numpy(image).permute(2, 0, 1).contiguous()  # (3, H, W) float32 [0,1]
        mask_t = torch.from_numpy(mask).unsqueeze(0).contiguous()  # (1, H, W) float32 {0,1}
        return image_t, mask_t, pair.id
