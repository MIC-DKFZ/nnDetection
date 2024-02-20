import torch

from nndet.core.post.detr import MaxFGBoxPost, TopKBoxPost


def example_predictions():
    pred_probs = torch.zeros(2, 2, 3) + torch.tensor([0.1, 0.2, 0.3], dtype=torch.float)[None, None]
    pred_boxes = torch.zeros(2, 2, 6) + torch.arange(2, dtype=torch.float).unsqueeze_(1).unsqueeze_(0)
    return pred_probs, pred_boxes


def test_max_fg_box_post():
    pred_probs, pred_boxes = example_predictions()
    expected_boxes = [torch.zeros(2, 6, dtype=torch.float) + torch.arange(2).unsqueeze_(1).unsqueeze_(0)] * 2
    expected_probs = [torch.zeros(2, dtype=torch.float) + 0.3] * 2
    expected_labels = [torch.zeros(2, dtype=torch.long) + 2] * 2

    module = MaxFGBoxPost()
    module_boxes, module_probs, module_labels = module.process_batch(pred_probs, pred_boxes)

    assert len(module_boxes) == 2
    assert len(module_probs) == 2
    assert len(module_labels) == 2
    for idx in range(2):
        assert torch.allclose(module_boxes[idx], expected_boxes[idx])
        assert torch.allclose(module_probs[idx], expected_probs[idx])
        assert torch.allclose(module_labels[idx], expected_labels[idx])


def test_topk_box_post():
    pred_probs, pred_boxes = example_predictions()
    expected_boxes = [
        torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1],
            ]
        )
    ] * 2
    expected_probs = [torch.tensor([0.3, 0.3, 0.2, 0.2], dtype=torch.float)] * 2
    expected_labels = [torch.tensor([2, 2, 1, 1], dtype=torch.long)] * 2

    module = TopKBoxPost(topk=4)
    module_boxes, module_probs, module_labels = module.process_batch(pred_probs, pred_boxes)

    assert len(module_boxes) == 2
    assert len(module_probs) == 2
    assert len(module_labels) == 2
    for idx in range(2):
        assert torch.allclose(module_boxes[idx], expected_boxes[idx])
        assert torch.allclose(module_probs[idx], expected_probs[idx])
        assert torch.allclose(module_labels[idx], expected_labels[idx])
