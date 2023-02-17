import torch

from nndet.losses.regression.diou import DIoULoss
from nndet.losses.regression.giou import GIoULoss, GIoULossPaired


def test_giou_loss():
    boxes0_3d = torch.tensor([[0, 0, 2, 2, 0, 1], [0, 0, 2, 2, 0, 1]]).float()
    boxes1_3d = torch.tensor([[1, 1, 3, 3, 0, 1], [1, 1, 3, 3, 0, 1]]).float()
    loss_fn = GIoULoss(reduction="sum", loss_weight=3.0)

    computed_loss = loss_fn(boxes0_3d, boxes1_3d)
    expected_loss = torch.tensor(-((1.0 / 7.0) - (2.0 / 9.0)) * 2 * 3)
    assert computed_loss.allclose(expected_loss)


def test_giou_paired_loss():
    boxes0_3d = torch.tensor([[0, 0, 2, 2, 0, 1], [0, 0, 2, 2, 0, 1]]).float()
    boxes1_3d = torch.tensor([[1, 1, 3, 3, 0, 1], [1, 1, 3, 3, 0, 1]]).float()
    loss_fn = GIoULossPaired(reduction="sum", loss_weight=3.0)

    computed_loss = loss_fn(boxes0_3d, boxes1_3d)
    expected_loss = torch.tensor(-((1.0 / 7.0) - (2.0 / 9.0)) * 2 * 3)
    assert computed_loss.allclose(expected_loss)


def test_diou_loss():
    boxes0_3d = torch.tensor([[0, 0, 2, 2, 0, 2], [0, 0, 2, 2, 0, 2]]).float()
    boxes1_3d = torch.tensor([[1, 1, 3, 3, 1, 3], [1, 1, 3, 3, 1, 3]]).float()
    loss_fn = DIoULoss(reduction="sum", loss_weight=3.0, eps=0)

    # iou = 1 / 15
    # cd^2 = 3
    # diag^2 = 27

    computed_loss = loss_fn(boxes0_3d, boxes1_3d)
    expected_loss = torch.tensor((1 - 1 / 15 + 3 / 27) * 2 * 3)
    print(computed_loss)
    print(expected_loss)
    assert computed_loss.allclose(expected_loss)
