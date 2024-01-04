import torch

from nndet.core.boxes.criterions.cls import (
    FocalClassCriterionSigmoid,
    SimpleClassCriterionSigmoid,
    SimpleClassCriterionSoftmax,
)


def test_simple_softmax_criterion():
    pred_logits = torch.ones((2, 4))
    target_labels = torch.tensor([1, 2])

    criterion = SimpleClassCriterionSoftmax(loss_weight=1)
    loss = criterion(pred_logits, target_labels)

    assert loss.shape == (2, 2)
    assert torch.allclose(loss, torch.tensor([[-0.25, -0.25], [-0.25, -0.25]]))


def test_simple_sigmoid_criterion():
    s1 = torch.functional.F.sigmoid(torch.tensor(1.0))

    pred_logits = torch.ones((2, 3))
    pred_logits[0, 0] = 0
    pred_logits[1, 1] = 0
    target_labels = torch.tensor([1, 2])

    criterion = SimpleClassCriterionSigmoid(loss_weight=1)
    loss = criterion(pred_logits, target_labels)

    assert loss.shape == (2, 2)
    assert torch.allclose(loss, torch.tensor([[-0.5, -s1.item()], [-s1.item(), -0.5]]))


def test_focal_sigmoid_criterion():
    pred_logits = torch.ones((2, 3))
    target_labels = torch.tensor([1, 2])

    criterion = FocalClassCriterionSigmoid(alpha=0.5, gamma=2, loss_weight=1, eps=0.01)
    loss = criterion(pred_logits, target_labels)

    s1 = torch.functional.F.sigmoid(torch.tensor(1.0))
    val_neg = 0.5 * (s1**2) * -torch.log(1 - s1 + 0.01)
    val_pos = 0.5 * ((1 - s1) ** 2) * -torch.log(s1 + 0.01)
    val = (val_pos - val_neg).item()

    expected_criterion = torch.tensor([[val, val], [val, val]])
    assert loss.shape == (2, 2)
    assert torch.allclose(loss, expected_criterion)
