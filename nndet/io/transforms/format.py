# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Hashable, Optional

import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform

import nndet.core.ops_np as ops_np
from nndet.utils.enums import BoxPointMode


class Boxes2ObjectPointsTransform(AbstractTransform):
    def __init__(
        self,
        data_key: Hashable,
        box_coord_key: Hashable,
        point_key: Hashable,
        label_key: Optional[Hashable] = None,
        mode: str = "corners",
    ) -> None:
        """
        Convert bounding boxes into object points

        Args:
            data_key: specify where data is located in dict.
                Expects data to be of format [B, C, s_dims], where B is the batch
                size, C number of of channels and s_dims are up to three
                spatial dimensions. Defaults to "data".
            box_coord_key: specify where boxes are located in dict.
                Expects boxes to be of format List([R, #dims * 2]) where
                List represents the batch dimension, R is the number of objects
                per image and #dims is the number of spatial dimensions
            label_key: specify where seg is located in dict.
                Expects seg to be of format [B, C, s_dims], where B is the batch
                size, C number of of channels and s_dims are up to three
                spatial dimensions. Defaults to "data".
            point_key: specify where points should be located in dict.
                Expects points to be in the format List([R, L, dims + 1]) where
                the List is the batch dimension, R is the number of objects,
                L is the number of points per object and dims are the number
                of spatial dimensions
            mode: Define conversion mode from boxes to points.
                Defaults to "corner".
        """
        super().__init__()
        self.data_key = data_key
        self.box_coord_key = box_coord_key
        self.point_key = point_key
        self.label_key = label_key
        self.mode = BoxPointMode(mode)

    def __call__(self, **data_dict: dict) -> dict:
        boxes = data_dict[self.box_coord_key]
        batch_size = len(boxes)

        points = []
        for batch_idx in range(batch_size):
            if self.mode == BoxPointMode.CORNERS:
                _points = ops_np.boxes2corner_points(boxes[batch_idx])
            elif self.mode == BoxPointMode.CENTERS:
                _points = ops_np.boxes2center_points(boxes[batch_idx])
            else:
                raise ValueError(
                    f"BoxPoint mode {self.mode} is not supported for this transformation {self.__class__.__name__}"
                )
            points.append(_points)

        data_dict[self.point_key] = ops_np.points_to_homogeneous(points)
        return data_dict


class ObjectPoints2BoxesTransform(AbstractTransform):
    def __init__(
        self,
        data_key: Hashable,
        box_coord_key: Hashable,
        box_label_key: Hashable,
        point_key: Hashable,
        label_key: Optional[Hashable] = None,
        pop_points: bool = True,
    ) -> None:
        """
        Convert bounding boxes into object points and discards/clips
        bounding boxes to the image boundaries. Bounding boxes can
        be moved out of the image when spatial augmentations are applied
        to the images.

        Args:
            data_key: specify where data is located in dict.
                Expects data to be of format [B, C, s_dims], where B is the batch
                size, C number of of channels and s_dims are up to three
                spatial dimensions. Defaults to "data".
            box_coord_key: specify where boxes are located in dict.
                Expects boxes to be of format List([R, #dims * 2]) where
                List represents the batch dimension, R is the number of objects
                in the image and #dims is the number of spatial dimensions
            box_label_key: specify where box labels are located in dict.
                Used to load and save the labels of the final boxes.
                Exptects label to be of format List([R]) where List represents
                the batch dimennsion and R is the number of object in the image
            label_key: specify where seg is located in dict.
                Expects seg to be of format [B, C, s_dims], where B is the batch
                size, C number of of channels and s_dims are up to three
                spatial dimensions. Defaults to "data".
            point_key: specify where points should be located in dict.
                Expects points to be in the format List([R, L, dims + 1]) where
                the List is the batch dimension, R is the number of objects,
                L is the number of points per object and dims are the number
                of spatial dimensions
            pop_points: remove point key from dictionary
        """
        super().__init__()
        self.data_key = data_key
        self.box_coord_key = box_coord_key
        self.box_label_key = box_label_key
        self.point_key = point_key
        self.label_key = label_key
        self.pop_points = pop_points

    def __call__(self, **data_dict: dict) -> dict:
        if self.pop_points:
            points = ops_np.points_to_cartesian(data_dict.pop(self.point_key))
        else:
            points = ops_np.points_to_cartesian(data_dict[self.point_key])

        batch_size = len(points)
        boxes_label_batch = data_dict[self.box_label_key]
        img_shape = data_dict[self.data_key].shape[2:]
        dim = len(img_shape)

        boxes_coords = []
        boxes_labels = []
        for batch_idx in range(batch_size):
            if points[batch_idx].size == 0:
                boxes_coords.append(np.array([[]]).reshape(0, dim * 2))
                boxes_labels.append(np.array([]))
            else:
                boxes_sample = ops_np.object_points2boxes(points[batch_idx])
                crop_boxes = ops_np.clip_boxes_to_image(boxes_sample, img_shape=img_shape)
                # remove small boxes (everything outside of the crop has size 0 or 1)
                keep = ops_np.remove_small_boxes(crop_boxes, min_size=2)
                boxes_coords.append(boxes_sample[keep])
                boxes_labels.append(boxes_label_batch[batch_idx][keep])

        data_dict[self.box_coord_key] = boxes_coords
        data_dict[self.box_label_key] = boxes_labels
        return data_dict
