from typing import List, Tuple

import torch

from nndet.core.boxes.matcher1to1.base import BaseMatcher


class BaseMatcherNoMatch(BaseMatcher):
    """
    Always returns 0 indices
    """

    @torch.no_grad()
    def match(
        self,
        pred_logits: torch.Tensor,
        pred_coords: torch.Tensor,
        target_boxes: List[torch.Tensor],
        target_labels: List[torch.Tensor],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        n = [min(p.shape[0], t.shape[0]) for p, t in zip(pred_logits, target_labels)]
        return [(torch.zeros(n_i, dtype=torch.long), torch.zeros(n_i, dtype=torch.long)) for n_i in n], {}


def test_forward():
    pred_logits = torch.zeros(3, 3, 2)
    pred_coords = torch.zeros(3, 3, 6)

    target_boxes = [torch.zeros(1, 6), torch.zeros(0, 6), torch.zeros(1, 6)]
    target_labels = [torch.ones(1), torch.zeros(0), torch.zeros(1, 6)]

    matcher = BaseMatcherNoMatch(None, None)

    indices, _ = matcher(pred_logits, pred_coords, target_boxes, target_labels)

    assert len(indices) == 3
    assert torch.allclose(indices[0][0], torch.tensor([0], dtype=torch.long))
    assert torch.allclose(indices[0][1], torch.tensor([0], dtype=torch.long))
    assert indices[1][0] is None
    assert indices[1][1] is None
    assert torch.allclose(indices[2][0], torch.tensor([0], dtype=torch.long))
    assert torch.allclose(indices[2][1], torch.tensor([0], dtype=torch.long))


def test_unmask_indices():
    matcher = BaseMatcherNoMatch(None, None)

    masked_indices = [
        (torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long)),
        (torch.tensor([1], dtype=torch.long), torch.tensor([1], dtype=torch.long)),
    ]
    mask = [True, False, False, True, False]

    indices = matcher.unmask_indices(mask, masked_indices)

    assert len(indices) == 5
    assert torch.allclose(indices[0][0], torch.tensor([0], dtype=torch.long))
    assert torch.allclose(indices[0][1], torch.tensor([0], dtype=torch.long))
    assert indices[1][0] is None
    assert indices[1][1] is None
    assert indices[2][0] is None
    assert indices[2][1] is None
    assert torch.allclose(indices[3][0], torch.tensor([1], dtype=torch.long))
    assert torch.allclose(indices[3][1], torch.tensor([1], dtype=torch.long))
    assert indices[4][0] is None
    assert indices[4][1] is None
