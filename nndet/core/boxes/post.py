from typing import Optional, Tuple, Union

import torch

from nndet.core.boxes.clip import clip_boxes_to_image_
from nndet.core.boxes.utils import remove_small_boxes as fn_remove_small_boxes
from nndet.core.boxes.nms import batched_nms


def post_image_single_class_regression(
    boxes: torch.Tensor, 
    probs: torch.Tensor,
    num_foreground_classes: int,
    image_shape: Union[Tuple[int, int], Tuple[int, int, int]],
    nms_thresh: float,
    topk_candidates: Optional[int] = None,
    score_thresh: Optional[float] = None,
    remove_small_boxes: Optional[float] = None,
    detections_per_img: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Postprocess bounding box deltas and probabilities for a single image
        Adapted from torchvision https://github.com/pytorch/vision

        Args:
            boxes: predicted deltas for proposals [N, dim * 2]
            probs: predicted logits for boxes [N, C]
            image_shape: shape of image

        Returns:
            Tensor: final boxes [R, dim * 2]
            Tensor: final scores (for final class) [R]
            Tensor: final class label [R]
        """
        assert boxes.shape[0] == probs.shape[0]
        boxes = clip_boxes_to_image_(boxes, image_shape)
        probs = probs.flatten()

        if topk_candidates is not None:
            num_topk = min(topk_candidates, boxes.size(0))
            probs, idx = probs.sort(descending=True)
            probs, idx = probs[:num_topk], idx[:num_topk]
        else:
            idx = torch.arange(probs.numel())

        if score_thresh is not None:
            keep_idxs = probs > score_thresh
            probs, idx = probs[keep_idxs], idx[keep_idxs]

        anchor_idxs = idx // num_foreground_classes
        labels = idx % num_foreground_classes
        boxes = boxes[anchor_idxs]

        if remove_small_boxes is not None:
            keep = fn_remove_small_boxes(boxes, min_size=remove_small_boxes)
            boxes, probs, labels = boxes[keep], probs[keep], labels[keep]

        keep = batched_nms(boxes, probs, labels, nms_thresh)

        if detections_per_img is not None:
            keep = keep[:detections_per_img]
        return boxes[keep], probs[keep], labels[keep]
