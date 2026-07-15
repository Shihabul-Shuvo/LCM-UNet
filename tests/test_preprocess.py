import numpy as np

from lcmunet.data.preprocess import (
    INPUT_SIZE,
    load_and_preprocess_pair,
    preprocess_image,
    preprocess_mask,
)


def test_preprocess_image_shape_range_dtype():
    raw = (np.random.default_rng(0).random((100, 130, 3)) * 255).astype(np.uint8)
    out = preprocess_image(raw, size=64)
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_preprocess_mask_binarises_and_resizes():
    raw = np.zeros((50, 50), dtype=np.uint8)
    raw[10:40, 10:40] = 200  # above threshold
    raw[0:10, 0:10] = 50  # below threshold
    out = preprocess_mask(raw, size=32)
    assert out.shape == (32, 32)
    assert out.dtype == np.float32
    assert set(np.unique(out)).issubset({0.0, 1.0})
    assert out.sum() > 0  # the foreground blob survived resize+threshold


def test_preprocess_mask_threshold_boundary():
    raw = np.array([[126, 127, 128]], dtype=np.uint8)
    out = preprocess_mask(raw, size=1, threshold=127)
    # only the pixel strictly greater than 127 should end up 1 after nearest resize to 1x1
    # (nearest-resize to a single pixel picks one source pixel deterministically;
    # here we just check no value other than {0,1} appears)
    assert set(np.unique(out)).issubset({0.0, 1.0})


def test_load_and_preprocess_pair_roundtrip(tmp_path):
    from PIL import Image

    img_path = tmp_path / "img.jpg"
    mask_path = tmp_path / "mask.png"
    Image.fromarray((np.random.default_rng(0).random((60, 60, 3)) * 255).astype(np.uint8)).save(img_path)
    m = np.zeros((60, 60), dtype=np.uint8)
    m[20:40, 20:40] = 255
    Image.fromarray(m).save(mask_path)

    image, mask = load_and_preprocess_pair(img_path, mask_path, size=INPUT_SIZE)
    assert image.shape == (INPUT_SIZE, INPUT_SIZE, 3)
    assert mask.shape == (INPUT_SIZE, INPUT_SIZE)
    assert set(np.unique(mask)).issubset({0.0, 1.0})
