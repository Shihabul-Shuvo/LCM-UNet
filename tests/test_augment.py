import numpy as np

from lcmunet.data.augment import augment_pair, random_hflip, random_vflip


class _AlwaysApply:
    """Stub rng that always triggers the augmentation branch (rng.random() < 0.5)
    and returns a fixed value for uniform(), for deterministic assertions."""

    def random(self):
        return 0.0

    def uniform(self, low, high):
        return (low + high) / 2.0


def _sample_image_mask(size=32, seed=0):
    rng = np.random.default_rng(seed)
    image = rng.random((size, size, 3)).astype(np.float32)
    mask = (rng.random((size, size)) > 0.5).astype(np.float32)
    return image, mask


def test_hflip_is_a_true_reflection():
    image, mask = _sample_image_mask()
    out_img, out_mask = random_hflip(image.copy(), mask.copy(), _AlwaysApply())
    assert np.array_equal(out_img, image[:, ::-1, :])
    assert np.array_equal(out_mask, mask[:, ::-1])


def test_vflip_is_a_true_reflection():
    image, mask = _sample_image_mask()
    out_img, out_mask = random_vflip(image.copy(), mask.copy(), _AlwaysApply())
    assert np.array_equal(out_img, image[::-1, :, :])
    assert np.array_equal(out_mask, mask[::-1, :])


def test_augment_pair_preserves_shape_range_and_binary_mask():
    image, mask = _sample_image_mask(size=64)
    rng = np.random.default_rng(42)
    for _ in range(30):
        aug_img, aug_mask = augment_pair(image.copy(), mask.copy(), rng)
        assert aug_img.shape == (64, 64, 3)
        assert aug_mask.shape == (64, 64)
        assert aug_img.dtype == np.float32 and aug_mask.dtype == np.float32
        assert aug_img.min() >= 0.0 and aug_img.max() <= 1.0
        assert set(np.unique(aug_mask)).issubset({0.0, 1.0})


def test_augment_pair_is_stochastic_across_seeds():
    image, mask = _sample_image_mask(size=64)
    out_a, _ = augment_pair(image.copy(), mask.copy(), np.random.default_rng(1))
    out_b, _ = augment_pair(image.copy(), mask.copy(), np.random.default_rng(2))
    assert not np.array_equal(out_a, out_b)
