# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
#
# Parts of this code are from torchvision (https://github.com/pytorch/vision) licensed under
# SPDX-FileCopyrightText: Soumith Chintala 2016
# SPDX-License-Identifier: BSD-3-Clause

from abc import abstractmethod
from typing import List, Optional, Sequence, Tuple, Union

import torch

import nndet.core.ops_torch as ops_torch
from nndet.core.boxes.nms import batched_nms


class BoxPostprocessing:
    def __init__(
        self,
        num_foreground_classes: int,
        nms_thresh: float = 1.0,
        remove_small_boxes: Optional[float] = None,
        detections_per_img: Optional[int] = None,
        topk_candidates: Optional[int] = None,
        score_thresh: Optional[float] = None,
        is_class_agnostic: bool = True,
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
        self.is_class_agnostic = is_class_agnostic

    def process_batch(
        self,
        reps: List[torch.Tensor],
        probs: List[torch.Tensor],
        image_shapes: List[Union[Tuple[int, int], Tuple[int, int, int]]],
        num_anchors_per_level: Optional[Sequence[int]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        all_reps, all_probs, all_labels = [], [], []
        for idx, img_shape in enumerate(image_shapes):
            if self.is_class_agnostic:
                _reps, _probs, _labels = self.process_image_class_agnostic(
                    img_reps=reps[idx],
                    img_probs=probs[idx],
                    img_shape=img_shape,
                    num_anchors_per_level=num_anchors_per_level,
                )
            else:
                _reps, _probs, _labels = self.process_image_per_class(
                    img_reps=reps[idx],
                    img_probs=probs[idx],
                    img_shape=img_shape,
                    num_anchors_per_level=num_anchors_per_level,
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
            img_reps: predicted deltas for proposals [N, dim * 2]
            img_probs: predicted logits for boxes [N, C]
            img_shape: shape of image
            num_anchors_per_level: number of anchors per level

        Returns:
            Tensor: final boxes [R, dim * 2]
            Tensor: final scores (for final class) [R]
            Tensor: final class label [R]
        """
        assert img_reps.shape[0] == img_probs.shape[0]
        boxes = ops_torch.clip_boxes_to_image_(img_reps, img_shape)
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
            keep = ops_torch.remove_small_boxes(boxes, min_size=self.remove_small_boxes)
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
        """
        Postprocess bounding box deltas and probabilities for a single image
        Adapted from torchvision https://github.com/pytorch/vision

        Args:
            img_reps: predicted deltas for proposals [N, C * dim * 2]
            img_probs: predicted logits for boxes [N, C]
            img_shape: shape of image
            num_anchors_per_level: number of anchors per level

        Returns:
            Tensor: final boxes [R, dim * 2]
            Tensor: final scores (for final class) [R]
            Tensor: final class label [R]
        """
        assert img_reps.shape[0] == img_probs.shape[0]
        assert img_probs.shape[-1] == self.num_foreground_classes
        assert (img_reps.shape[1] == img_probs.shape[-1] * 6) or (img_reps.shape[1] == img_probs.shape[-1] * 4)

        img_labels = torch.arange(self.num_foreground_classes, device=img_probs.device)
        img_labels = img_labels.view(1, -1).expand_as(img_probs)  # [N, C]

        dims = img_reps.shape[1] // self.num_foreground_classes
        boxes = img_reps.reshape(-1, dims)  # [R, 2 * dims]
        probs = img_probs.reshape(-1)  # [R]
        labels = img_labels.reshape(-1)  # [R]

        boxes = ops_torch.clip_boxes_to_image_(boxes, img_shape)
        if self.topk_candidates is not None:
            num_topk = min(self.topk_candidates, boxes.size(0))
            probs, idx = probs.sort(descending=True)
            probs, idx = probs[:num_topk], idx[:num_topk]
        else:
            idx = torch.arange(probs.numel())

        if self.score_thresh is not None:
            keep_idxs = probs > self.score_thresh
            probs, idx = probs[keep_idxs], idx[keep_idxs]

        # filter boxes and labels
        boxes = boxes[idx]
        labels = labels[idx]

        if self.remove_small_boxes is not None:
            keep = ops_torch.remove_small_boxes(boxes, min_size=self.remove_small_boxes)
            boxes, probs, labels = boxes[keep], probs[keep], labels[keep]

        boxes, probs, labels = self.nms(boxes, probs, labels)

        if self.detections_per_img is not None:
            boxes = boxes[: self.detections_per_img]
            probs = probs[: self.detections_per_img]
            labels = labels[: self.detections_per_img]
        return boxes, probs, labels

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


class PerLevelBoxPostprocessing(BoxPostprocessing):
    """
    Warning:
        Only use this with a single class. otherwise the results will be off.
    """

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
        (Note: in contrast to torchvision this performs some
        operations per image which could be parallelized across the batch)

        Args:
            img_reps: predicted deltas for proposals [N, dim * 2]
            img_probs: predicted logits for boxes [N, C]
            img_shape: shape of image
            num_anchors_per_level: number of anchors per level

        Returns:
            Tensor: final boxes [R, dim * 2]
            Tensor: final scores (for final class) [R]
            Tensor: final class label [R]
        """
        assert img_reps.shape[0] == img_probs.shape[0]
        if img_probs.shape[1] != 1:
            raise ValueError(
                "PerLevelBoxPostprocessing is only supported with a "
                f"single class but found {img_probs.shape[1]} classes"
            )
        boxes = ops_torch.clip_boxes_to_image_(img_reps, img_shape)
        probs = img_probs.flatten()

        levels = [
            torch.full((n,), fill_value=level_idx, dtype=torch.long, device=probs.device)
            for level_idx, n in enumerate(num_anchors_per_level)
        ]
        levels = torch.cat(levels, 0)

        if self.topk_candidates is not None:
            idx = self.topk_per_level(probs, num_anchors_per_level)
        else:
            idx = torch.arange(probs.numel())

        boxes = boxes[idx]
        probs = probs[idx]
        levels = levels[idx]

        if self.score_thresh is not None:
            keep_mask = probs > self.score_thresh
            probs, boxes, levels = probs[keep_mask], boxes[keep_mask], levels[keep_mask]

        if self.remove_small_boxes is not None:
            keep = ops_torch.fn_remove_small_boxes(boxes, min_size=self.remove_small_boxes)
            boxes, probs, levels = boxes[keep], probs[keep], levels[keep]
        boxes, probs, _ = self.nms(boxes, probs, levels)

        if self.detections_per_img is not None:
            boxes = boxes[: self.detections_per_img]
            probs = probs[: self.detections_per_img]

        labels = torch.ones(probs.shape, dtype=torch.long, device=probs.device)
        return boxes, probs, labels

    def topk_per_level(
        self,
        probs: torch.Tensor,
        num_anchors_per_level: Sequence[int],
    ) -> torch.Tensor:
        all_idx = []
        idx_offset = 0
        for probs_per_level in probs.split(num_anchors_per_level):
            # select topk
            _topk = min(self.topk_candidates, probs_per_level.shape[0])
            _, sorted_idx_level = probs_per_level.topk(_topk)

            all_idx.append(sorted_idx_level + idx_offset)
            idx_offset = idx_offset + probs_per_level.shape[0]

        return torch.cat(all_idx)

    def nms(
        self,
        img_reps: torch.Tensor,
        img_probs: torch.Tensor,
        levels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        res = batched_nms(
            boxes=img_reps,
            scores=img_probs,
            labels=levels,
            iou_thresh=self.nms_thresh,
        )
        return res[:3]

    def process_image_per_class(
        self,
        img_reps: torch.Tensor,
        img_probs: torch.Tensor,
        img_shape: Union[Tuple[int, int], Tuple[int, int, int]],
        num_anchors_per_level: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError
