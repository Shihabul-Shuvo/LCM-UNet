import numpy as np
import torch

from lcmunet.metrics import (
    accuracy_score,
    compute_all_metrics,
    dice_score,
    evaluate,
    hd95_and_assd,
    iou_score,
    load_per_image_dice,
    save_per_image_dice,
    sensitivity_score,
    specificity_score,
)


def test_dice_and_iou_perfect_match():
    a = np.ones((8, 8), dtype=bool)
    assert dice_score(a, a) == 1.0
    assert iou_score(a, a) == 1.0


def test_dice_and_iou_no_overlap_near_zero():
    a = np.zeros((8, 8), dtype=bool)
    a[:4, :4] = True
    b = np.zeros((8, 8), dtype=bool)
    b[4:, 4:] = True
    assert dice_score(a, b) < 1e-3
    assert iou_score(a, b) < 1e-3


def test_sensitivity_specificity_accuracy_known_values():
    # pred: top-left quadrant is 1; gt: top-left quadrant is 1 (identical)
    pred = np.zeros((4, 4), dtype=bool)
    pred[:2, :2] = True
    gt = pred.copy()
    assert sensitivity_score(pred, gt) == 1.0
    assert specificity_score(pred, gt) == 1.0
    assert accuracy_score(pred, gt) == 1.0


def test_sensitivity_zero_when_all_positives_missed():
    pred = np.zeros((4, 4), dtype=bool)
    gt = np.zeros((4, 4), dtype=bool)
    gt[:2, :2] = True
    assert sensitivity_score(pred, gt) < 1e-3
    assert specificity_score(pred, gt) == 1.0  # true negatives everywhere else


def test_hd95_assd_both_empty_is_zero():
    empty = np.zeros((10, 10), dtype=bool)
    hd95, assd = hd95_and_assd(empty, empty)
    assert hd95 == 0.0 and assd == 0.0


def test_hd95_assd_one_empty_is_nan():
    empty = np.zeros((10, 10), dtype=bool)
    nonempty = np.zeros((10, 10), dtype=bool)
    nonempty[3:6, 3:6] = True
    hd95, assd = hd95_and_assd(empty, nonempty)
    assert np.isnan(hd95) and np.isnan(assd)


def test_hd95_assd_identical_shapes_is_zero():
    sq = np.zeros((20, 20), dtype=bool)
    sq[5:15, 5:15] = True
    hd95, assd = hd95_and_assd(sq, sq.copy())
    assert hd95 == 0.0 and assd == 0.0


def test_hd95_assd_shifted_shape_is_small_positive():
    sq1 = np.zeros((20, 20), dtype=bool)
    sq1[5:15, 5:15] = True
    sq2 = np.zeros((20, 20), dtype=bool)
    sq2[6:16, 5:15] = True  # shifted by 1 pixel
    hd95, assd = hd95_and_assd(sq1, sq2)
    assert 0 < hd95 < 3
    assert 0 < assd < 3


def test_compute_all_metrics_boundary_flag():
    sq = np.zeros((10, 10), dtype=bool)
    sq[2:5, 2:5] = True
    with_boundary = compute_all_metrics(sq, sq.copy(), boundary=True)
    without_boundary = compute_all_metrics(sq, sq.copy(), boundary=False)
    assert "hd95" in with_boundary and "assd" in with_boundary
    assert "hd95" not in without_boundary and "assd" not in without_boundary


class _ConstantModel(torch.nn.Module):
    """Outputs a fixed logit map regardless of input -- for deterministic metric tests."""

    def __init__(self, value: float, shape):
        super().__init__()
        self.value = value
        self.shape = shape

    def forward(self, x):
        b = x.shape[0]
        return torch.full((b, 1, *self.shape), self.value)


def test_evaluate_perfect_model_gives_dice_one():
    shape = (8, 8)
    model = _ConstantModel(10.0, shape)  # sigmoid(10) ~= 1 everywhere
    images = torch.randn(2, 3, *shape)
    masks = torch.ones(2, 1, *shape)
    loader = [(images, masks, ["a", "b"])]

    result = evaluate(model, loader, torch.device("cpu"), boundary=True)
    assert result["dsc"] > 0.999
    assert result["ids"] == ["a", "b"]
    assert result["per_image"]["dsc"].shape == (2,)


def test_save_and_load_per_image_dice_roundtrip(tmp_path):
    ids = ["img1", "img2", "img3"]
    dsc = np.array([0.9, 0.5, 0.1])
    save_per_image_dice("cfg_abc", 3, ids, dsc, tmp_path)
    loaded = load_per_image_dice("cfg_abc", 3, tmp_path)
    assert list(loaded["ids"]) == ids
    assert np.allclose(loaded["dsc"], dsc)
    assert (tmp_path / "perimage" / "cfg_abc_3.npy").is_file()
