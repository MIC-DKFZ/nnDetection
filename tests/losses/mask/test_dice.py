import math

import pytest
import torch

from nndet.losses.mask.dice import BDiceMaskLoss
from nndet.losses.mask.functional.dice import soft_dice


@pytest.mark.parametrize("batch_dice", [True, False])
def test_bdicemask_loss_example(batch_dice: bool):
    loss_object = BDiceMaskLoss(
        batch_dice=batch_dice,
        smooth_nom=0.0,
        smooth_denom=0.0,
        reduction="mean",
    )

    preds = torch.zeros(2, 2, 8, 8, 8, dtype=torch.float)
    targets = torch.ones_like(preds)

    loss = loss_object(preds, targets)
    if batch_dice:
        v = -2 * 2 * 8 * 8 * 8 * 0.5 / (2 * 8 * 8 * 8 * 0.5 + 2 * 8 * 8 * 8)
        expected_loss = torch.tensor(
            [
                [v, v],
            ],
            dtype=torch.float,
        ).mean()
    else:
        v = -2 * 8 * 8 * 8 * 0.5 / (8 * 8 * 8 * 0.5 + 8 * 8 * 8)
        expected_loss = torch.tensor(
            [
                [v, v],
                [v, v],
            ],
            dtype=torch.float,
        ).mean()
    assert torch.allclose(loss, expected_loss)


@pytest.mark.parametrize("batch_dice", [True, False])
def test_dice_example(batch_dice: bool):
    preds = torch.zeros(2, 2, 8, 8, 8, dtype=torch.float)
    preds[0, 1, :4] = 0.5
    preds[0, 0, 4:] = 0.5
    targets = torch.ones_like(preds)

    loss = soft_dice(
        preds,
        targets,
        batch_dice=batch_dice,
        smooth_nom=0.0,
        smooth_denom=0.0,
    )  # [2, 2]
    if batch_dice:
        v = -2 * 8 * 8 * 4 * 0.5 / (8 * 8 * 4 * 0.5 + 2 * 8 * 8 * 8)
        expected_loss = torch.tensor(
            [
                [v, v],
            ],
            dtype=torch.float,
        )
    else:
        v = -2 * 8 * 8 * 4 * 0.5 / (8 * 8 * 4 * 0.5 + 8 * 8 * 8)
        expected_loss = torch.tensor(
            [
                [v, v],
                [0, 0],
            ],
            dtype=torch.float,
        )
    assert torch.allclose(loss, expected_loss)


@pytest.mark.parametrize("batch_dice", [True, False])
def test_dice_worst_prediction(batch_dice: bool):
    preds = torch.zeros(4, 3, 8, 8, 8, dtype=torch.float)
    targets = torch.ones_like(preds)
    loss = soft_dice(
        preds,
        targets,
        batch_dice=batch_dice,
        smooth_nom=0.0,
        smooth_denom=0.0,
    )

    if batch_dice:
        assert tuple(loss.shape) == (3,)
    else:
        assert tuple(loss.shape) == (4, 3)
    assert math.isclose(loss.mean().item(), 0)

    preds = torch.ones(4, 3, 8, 8, 8, dtype=torch.float)
    targets = torch.zeros_like(preds)
    loss = soft_dice(
        preds,
        targets,
        batch_dice=batch_dice,
        smooth_nom=0.0,
        smooth_denom=0.0,
    )

    if batch_dice:
        assert tuple(loss.shape) == (3,)
    else:
        assert tuple(loss.shape) == (4, 3)
    assert math.isclose(loss.mean().item(), 0)


@pytest.mark.parametrize("smooth_nom", [0.0, 1e-5])
@pytest.mark.parametrize("smooth_denom", [0.0, 1e-5])
@pytest.mark.parametrize("batch_dice", [True, False])
def test_dice_perfect_prediction(
    batch_dice: bool,
    smooth_nom: float,
    smooth_denom: float,
):
    torch.manual_seed(0)
    preds = (torch.rand(4, 3, 8, 8, 8) > 0.5).to(dtype=torch.float)
    targets = preds.clone()
    loss = soft_dice(
        preds,
        targets,
        batch_dice=batch_dice,
        smooth_nom=smooth_nom,
        smooth_denom=smooth_denom,
    )

    if batch_dice:
        assert tuple(loss.shape) == (3,)
    else:
        assert tuple(loss.shape) == (4, 3)

    assert math.isclose(loss.mean().item(), -1)
