from typing import Optional, Sequence, Tuple, Union

import pytest
import torch

from nndet.core.post.box import (
    BoxPostprocessing,
    CrossLevelBoxPostprocessing,
    PerLevelBoxPostprocessing,
)


class NoClassBoxPostprocessing(BoxPostprocessing):
    """
    Return original boxes with label 0 for all boxes.
    """

    def process_image_class_agnostic(
        self,
        img_reps: torch.Tensor,
        img_probs: torch.Tensor,
        img_shape: Union[Tuple[int, int], Tuple[int, int, int]],
        num_anchors_per_level: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return img_reps, img_probs, torch.zeros(img_probs.shape, dtype=torch.long)

    def process_image_per_class(
        self,
        img_reps: torch.Tensor,
        img_probs: torch.Tensor,
        img_shape: Union[Tuple[int, int], Tuple[int, int, int]],
        num_anchors_per_level: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dims = img_reps.shape[1] // self.num_classes
        img_reps = img_reps.reshape(-1, dims)
        img_probs = img_probs.reshape(-1)
        return img_reps, img_probs, torch.zeros(img_probs.shape, dtype=torch.long)


@pytest.fixture
def example_predictions():
    boxes = torch.tensor(
        [
            [0, 0, 1, 1, 0, 1],
            [0, 0, 1, 1, 0, 1],
            [0, 0, 1, 1, 0, 1],
            [0, 0, 1, 1, 0, 1],
            [1, 1, 2, 2, 1, 2],
        ],
        dtype=torch.float32,
    )
    probs = torch.tensor([0.6, 0.9, 0.8, 0.7, 0.5]).reshape(-1, 1)
    return {
        "reps": [boxes],
        "probs": [probs],
        "image_shapes": [(10, 10, 10)],
        "num_anchors_per_level": [1, 5],
    }


def test_box_postprocessing(example_predictions):
    post = NoClassBoxPostprocessing(
        num_classes=1,
        nms_thresh=0.6,
        is_class_agnostic=True,
    )
    expected_boxes = torch.clone(example_predictions["reps"][0])
    expected_probs = torch.clone(example_predictions["probs"][0])
    expected_labels = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)
    output_boxes, output_probs, output_labels = post.process_batch(**example_predictions)

    assert len(output_boxes) == 1
    assert len(output_probs) == 1
    assert len(output_labels) == 1

    assert torch.allclose(expected_boxes, output_boxes[0])
    assert torch.allclose(expected_probs, output_probs[0])
    assert torch.allclose(expected_labels, output_labels[0])


@pytest.mark.parametrize("is_class_agnostic", [True, False])
def test_cross_level_box_postprocessing(example_predictions, is_class_agnostic):
    post = CrossLevelBoxPostprocessing(
        num_classes=1,
        nms_thresh=0.6,
        is_class_agnostic=is_class_agnostic,
    )
    output_boxes, output_probs, output_labels = post.process_batch(**example_predictions)

    expected_boxes = torch.tensor(
        [
            [0, 0, 1, 1, 0, 1],
            [1, 1, 2, 2, 1, 2],
        ],
        dtype=torch.float32,
    )
    expected_probs = torch.tensor([0.9, 0.5], dtype=torch.float32)
    expected_labels = torch.tensor([0, 0], dtype=torch.long)

    assert len(output_boxes) == 1
    assert len(output_probs) == 1
    assert len(output_labels) == 1

    assert torch.allclose(expected_boxes, output_boxes[0])
    assert torch.allclose(expected_probs, output_probs[0])
    assert torch.allclose(expected_labels, output_labels[0])


def test_per_level_box_postprocessing(example_predictions):
    post = PerLevelBoxPostprocessing(
        num_classes=1,
        nms_thresh=0.6,
        is_class_agnostic=True,
    )
    output_boxes, output_probs, output_labels = post.process_batch(**example_predictions)

    expected_boxes = torch.tensor(
        [
            [0, 0, 1, 1, 0, 1],
            [0, 0, 1, 1, 0, 1],
            [1, 1, 2, 2, 1, 2],
        ],
        dtype=torch.float32,
    )
    expected_probs = torch.tensor([0.9, 0.6, 0.5], dtype=torch.float32)
    expected_labels = torch.tensor([0, 0, 0], dtype=torch.long)

    assert len(output_boxes) == 1
    assert len(output_probs) == 1
    assert len(output_labels) == 1

    assert torch.allclose(expected_boxes, output_boxes[0])
    assert torch.allclose(expected_probs, output_probs[0])
    assert torch.allclose(expected_labels, output_labels[0])
