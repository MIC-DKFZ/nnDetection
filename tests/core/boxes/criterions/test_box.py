import torch

from nndet.core.boxes.criterions.box import GIoUCenterBoxCriterion, L1RegCriterion


def test_l1_reg_criterion_equal():
    pred_coords = torch.ones((2, 4), dtype=torch.float)
    target_boxes = torch.ones((2, 4), dtype=torch.float)

    criterion = L1RegCriterion(loss_weight=2)
    loss = criterion(pred_coords, target_boxes)

    assert loss.shape == (2, 2)
    assert torch.allclose(loss, torch.tensor([[0, 0], [0, 0]], dtype=torch.float))


def test_giou_center_box_criterion_equal():
    pred_coords = torch.ones((2, 4), dtype=torch.float)
    target_boxes = torch.ones((2, 4), dtype=torch.float)

    criterion = GIoUCenterBoxCriterion(loss_weight=2)
    loss = criterion(pred_coords, target_boxes)

    assert loss.shape == (2, 2)
    assert torch.allclose(loss, torch.tensor([[-2, -2], [-2, -2]], dtype=torch.float))
