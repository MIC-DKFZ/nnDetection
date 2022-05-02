from abc import abstractmethod
from typing import List, Optional, Sequence, Tuple, Union

import torch

from nndet.core.boxes.clip import clip_boxes_to_image_
from nndet.core.boxes.nms import batched_nms
from nndet.core.boxes.ops import remove_small_boxes as fn_remove_small_boxes


class BoxPostprocessing:
    def __init__(
        self,
        num_foreground_classes: int,
        nms_thresh: float = 1.0,
        remove_small_boxes: Optional[float] = None,
        detections_per_img: Optional[int] = None,
        topk_candidates: Optional[int] = None,
        score_thresh: Optional[float] = None,
        class_agnostic: bool = True,
    ) -> None:
        """
        Provides an abstract interface to postprocess a batch of boxes
        from a detection model.

        Args:
            regress_class_agnostic: todo
        """
        super().__init__()
        self.num_foreground_classes = num_foreground_classes
        self.nms_thresh = nms_thresh
        self.remove_small_boxes = remove_small_boxes
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates
        self.score_thresh = score_thresh
        self.class_agnostic = class_agnostic

    def process_batch(
        self,
        reps: List[torch.Tensor],
        probs: List[torch.Tensor],
        image_shapes: List[Union[Tuple[int, int], Tuple[int, int, int]]],
        num_anchors_per_level: Optional[Sequence[int]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        all_reps, all_probs, all_labels = [], [], []
        for idx, img_shape in enumerate(image_shapes):
            if self.class_agnostic:
                _reps, _probs, _labels = self.process_image_class_agnostic(
                    img_reps=reps[idx],
                    img_probs=probs[idx],
                    img_shape=img_shape,
                )
            else:
                _reps, _probs, _labels = self.process_image_per_class(
                    img_reps=reps[idx],
                    img_probs=probs[idx],
                    img_shape=img_shape,
                )

            all_reps.append(_reps)
            all_probs.append(_probs)
            all_labels.append(_labels)
        return all_reps, all_probs, all_labels

    @abstractmethod
    def process_image_class_agnostic(
        self,
        img_reps: torch.Tensor,
        img_probs: torch.Tensor,
        img_shape: Union[Tuple[int, int], Tuple[int, int, int]],
        num_anchors_per_level: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def process_image_per_class(
        self,
        img_reps: torch.Tensor,
        img_probs: torch.Tensor,
        img_shape: Union[Tuple[int, int], Tuple[int, int, int]],
        num_anchors_per_level: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def nms(
        img_reps: torch.Tensor,
        img_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass


class CrossLevelBoxPostprocessing(BoxPostprocessing):
    def process_image_class_agnostic(
        self,
        img_reps: torch.Tensor,
        img_probs: torch.Tensor,
        img_shape: Union[Tuple[int, int], Tuple[int, int, int]],
        num_anchors_per_level: Optional[Sequence[int]] = None,
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
        assert img_reps.shape[0] == img_probs.shape[0]
        boxes = clip_boxes_to_image_(img_reps, img_shape)
        probs = img_probs.flatten()

        if self.topk_candidates is not None:
            num_topk = min(self.topk_candidates, boxes.size(0))
            probs, idx = probs.sort(descending=True)
            probs, idx = probs[:num_topk], idx[:num_topk]
        else:
            idx = torch.arange(probs.numel())

        if self.score_thresh is not None:
            keep_idxs = probs > self.score_thresh
            probs, idx = probs[keep_idxs], idx[keep_idxs]

        anchor_idxs = torch.div(idx, self.num_foreground_classes, rounding_mode="floor")
        labels = idx % self.num_foreground_classes

        boxes = boxes[anchor_idxs]
        if self.remove_small_boxes is not None:
            keep = fn_remove_small_boxes(boxes, min_size=self.remove_small_boxes)
            boxes, probs, labels = boxes[keep], probs[keep], labels[keep]
        boxes, probs, labels = self.nms(boxes, probs, labels)

        if self.detections_per_img is not None:
            boxes = boxes[: self.detections_per_img]
            probs = probs[: self.detections_per_img]
            labels = labels[: self.detections_per_img]
        return boxes, probs, labels

    def process_image_per_class(
        self,
        img_reps: torch.Tensor,
        img_probs: torch.Tensor,
        img_shape: Union[Tuple[int, int], Tuple[int, int, int]],
        num_anchors_per_level: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def nms(
        self,
        img_reps: torch.Tensor,
        img_probs: torch.Tensor,
        img_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        res = batched_nms(
            boxes=img_reps,
            scores=img_probs,
            labels=img_labels,
            iou_thresh=self.nms_thresh,
        )
        return res[:3]


# class NoNMSCrossLevelPostprocessing(CrossLevelBoxPostprocessing):
#     def nms(
#         self,
#         img_reps: torch.Tensor,
#         img_probs: torch.Tensor,
#         img_labels: torch.Tensor,
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         return (
#             img_reps,
#             img_probs,
#             img_labels,
#         )
