import torch

from lcmunet.losses import CombinedLoss, DiceLoss


def test_dice_loss_perfect_match_is_near_zero():
    logits = torch.full((2, 1, 8, 8), 10.0)  # sigmoid(10) ~= 1
    target = torch.ones((2, 1, 8, 8))
    loss = DiceLoss()(logits, target)
    assert loss.item() < 1e-3


def test_dice_loss_perfect_mismatch_is_near_one():
    logits = torch.full((2, 1, 8, 8), -10.0)  # sigmoid(-10) ~= 0
    target = torch.ones((2, 1, 8, 8))
    loss = DiceLoss()(logits, target)
    assert loss.item() > 0.999


def test_dice_loss_no_overlap_near_one():
    logits = torch.full((1, 1, 4, 4), -10.0)
    target = torch.zeros((1, 1, 4, 4))
    target[:, :, :2, :2] = 1.0
    logits2 = torch.full((1, 1, 4, 4), -10.0)
    logits2[:, :, 2:, 2:] = 10.0  # predicted region disjoint from target region
    loss = DiceLoss()(logits2, target)
    assert loss.item() > 0.99


def test_combined_loss_perfect_match_is_small():
    logits = torch.full((2, 1, 8, 8), 10.0)
    target = torch.ones((2, 1, 8, 8))
    loss = CombinedLoss()(logits, target)
    assert loss.item() < 1e-2


def test_combined_loss_is_bce_plus_dice():
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 8, 8)
    target = (torch.rand(2, 1, 8, 8) > 0.5).float()

    combined = CombinedLoss()(logits, target)
    bce = torch.nn.BCEWithLogitsLoss()(logits, target)
    dice = DiceLoss()(logits, target)

    assert torch.isclose(combined, bce + dice, atol=1e-6)


def test_dice_loss_gradients_are_finite():
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    target = (torch.rand(2, 1, 8, 8) > 0.5).float()
    loss = CombinedLoss()(logits, target)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
