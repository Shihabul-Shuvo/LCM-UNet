import pytest
import torch

from lcmunet.config import RunConfig
from lcmunet.data.datasets import SegmentationDataset
from lcmunet.data.loaders import build_cross_dataset_loader, build_dataloaders
from lcmunet.data.splits import build_cvc_split, build_kvasir_split


def _kvasir_config(paths, **overrides):
    defaults = dict(
        run_name="test",
        model_name="ultralight_baseline",
        dataset="kvasir_seg",
        seed=0,
        split_file="splits/kvasir_seg.json",
        batch_size=4,
        input_size=64,
    )
    defaults.update(overrides)
    return RunConfig(**defaults)


def test_segmentation_dataset_shapes_and_types(make_kvasir_raw):
    paths = make_kvasir_raw(n=10)
    ids = [f"img{i:04d}" for i in range(10)]
    ds = SegmentationDataset("kvasir_seg", ids, paths, augment=False, input_size=64)
    assert len(ds) == 10
    image, mask, item_id = ds[0]
    assert isinstance(image, torch.Tensor) and image.shape == (3, 64, 64)
    assert isinstance(mask, torch.Tensor) and mask.shape == (1, 64, 64)
    assert image.dtype == torch.float32 and mask.dtype == torch.float32
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}
    assert item_id in ids


def test_segmentation_dataset_missing_id_raises(make_kvasir_raw):
    paths = make_kvasir_raw(n=5)
    with pytest.raises(KeyError):
        SegmentationDataset("kvasir_seg", ["not_a_real_id"], paths, augment=False)


def test_segmentation_dataset_no_augment_is_deterministic(make_kvasir_raw):
    paths = make_kvasir_raw(n=5)
    ids = ["img0000"]
    ds = SegmentationDataset("kvasir_seg", ids, paths, augment=False, input_size=64)
    img_a, mask_a, _ = ds[0]
    img_b, mask_b, _ = ds[0]
    assert torch.equal(img_a, img_b)
    assert torch.equal(mask_a, mask_b)


def test_build_dataloaders_full(make_kvasir_raw):
    paths = make_kvasir_raw(n=100)
    build_kvasir_split(paths)
    config = _kvasir_config(paths)

    train_loader, val_loader, test_loader = build_dataloaders(config, paths, num_workers=0)
    assert len(train_loader.dataset) == 80
    assert len(val_loader.dataset) == 10
    assert len(test_loader.dataset) == 10

    images, masks, ids = next(iter(train_loader))
    assert images.shape == (4, 3, 64, 64)
    assert masks.shape == (4, 1, 64, 64)


def test_build_dataloaders_sanity_mode(make_kvasir_raw):
    paths = make_kvasir_raw(n=100)
    build_kvasir_split(paths)
    config = _kvasir_config(paths)

    train_loader, val_loader, test_loader = build_dataloaders(config, paths, sanity=True)
    assert len(train_loader.dataset) == 8
    assert len(val_loader.dataset) == 8
    assert len(test_loader.dataset) == 8
    assert train_loader.num_workers == 0


def test_build_dataloaders_train_is_augmented_val_is_not(make_kvasir_raw):
    paths = make_kvasir_raw(n=100)
    build_kvasir_split(paths)
    config = _kvasir_config(paths)

    train_loader, val_loader, _ = build_dataloaders(config, paths, num_workers=0)
    assert train_loader.dataset.augment is True
    assert val_loader.dataset.augment is False


def test_cross_dataset_loader_kvasir_to_cvc(make_kvasir_raw, make_cvc_raw):
    paths = make_kvasir_raw(n=100)
    make_cvc_raw(n=612)  # writes into the same paths.data_raw root
    build_kvasir_split(paths)
    build_cvc_split(paths)

    train_loader, test_loader = build_cross_dataset_loader(
        "kvasir_seg", "cvc_clinicdb", paths, batch_size=4, input_size=64, sanity=True
    )
    images, masks, ids = next(iter(train_loader))
    assert images.shape == (4, 3, 64, 64)
    images, masks, ids = next(iter(test_loader))
    assert images.shape == (4, 3, 64, 64)
    assert test_loader.dataset.augment is False


def test_cross_dataset_loader_rejects_non_required_pair(paths):
    with pytest.raises(ValueError):
        build_cross_dataset_loader("kvasir_seg", "isic2017", paths)
