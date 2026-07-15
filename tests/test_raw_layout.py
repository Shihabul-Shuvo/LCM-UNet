import pytest

from lcmunet.data import raw_layout as rl


def test_kvasir_missing_raises_with_instructions(paths):
    with pytest.raises(rl.RawDataMissingError) as exc_info:
        rl.list_kvasir_pairs(paths.data_raw)
    assert "kvasir-seg.zip" in exc_info.value.instructions


def test_kvasir_pairs_found(make_kvasir_raw):
    paths = make_kvasir_raw(n=20)
    pairs = rl.list_kvasir_pairs(paths.data_raw)
    assert len(pairs) == 20
    assert all(p.image_path.exists() and p.mask_path.exists() for p in pairs)


def test_cvc_missing_raises_with_instructions(paths):
    with pytest.raises(rl.RawDataMissingError) as exc_info:
        rl.list_cvc_pairs(paths.data_raw)
    assert "polyp.grand-challenge.org" in exc_info.value.instructions


def test_cvc_pairs_found_with_original_groundtruth_names(make_cvc_raw):
    paths = make_cvc_raw(n=612)
    pairs = rl.list_cvc_pairs(paths.data_raw)
    assert len(pairs) == 612


def test_cvc_pairs_found_case_insensitive_images_masks_names(paths):
    root = rl.cvc_root(paths.data_raw)
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir(parents=True)
    from PIL import Image
    import numpy as np

    for i in range(1, 6):
        Image.fromarray(np.zeros((8, 8, 3), dtype="uint8")).save(root / "images" / f"{i}.png")
        Image.fromarray(np.zeros((8, 8), dtype="uint8")).save(root / "masks" / f"{i}.png")
    pairs = rl.list_cvc_pairs(paths.data_raw)
    assert len(pairs) == 5


def test_cvc_pairs_bridge_zero_padding(paths):
    """Some mirrors zero-pad one side (e.g. '001.tif') but not the other."""
    root = rl.cvc_root(paths.data_raw)
    (root / "Original").mkdir(parents=True)
    (root / "Ground Truth").mkdir(parents=True)
    from PIL import Image
    import numpy as np

    for i in range(1, 6):
        Image.fromarray(np.zeros((8, 8, 3), dtype="uint8")).save(root / "Original" / f"{i:03d}.tif")
        Image.fromarray(np.zeros((8, 8), dtype="uint8")).save(root / "Ground Truth" / f"{i}.tif")
    pairs = rl.list_cvc_pairs(paths.data_raw)
    assert len(pairs) == 5


def test_isic_missing_raises_with_instructions(paths):
    with pytest.raises(rl.RawDataMissingError) as exc_info:
        rl.list_isic_pairs(paths.data_raw, "isic2017")
    assert "challenge.isic-archive.com" in exc_info.value.instructions
    assert "ISIC2017_Task1-2_Training_Input" in exc_info.value.instructions


def test_isic_pairs_found(make_isic_raw):
    paths = make_isic_raw(version="isic2017", n=50)
    pairs = rl.list_isic_pairs(paths.data_raw, "isic2017")
    assert len(pairs) == 50


def test_list_pairs_dispatch(make_kvasir_raw):
    paths = make_kvasir_raw(n=5)
    assert len(rl.list_pairs("kvasir_seg", paths.data_raw)) == 5


def test_list_pairs_unknown_dataset_raises(paths):
    with pytest.raises(ValueError):
        rl.list_pairs("not_a_real_dataset", paths.data_raw)
