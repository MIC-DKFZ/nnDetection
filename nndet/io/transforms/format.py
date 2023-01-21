from typing import Hashable, Optional

from batchgenerators.transforms.abstract_transforms import AbstractTransform


class Boxes2ObjectPoints(AbstractTransform):
    def __init__(
        self,
        data_key: Hashable,
        box_coord_key: Hashable,
        point_key: Hashable,
        label_key: Optional[Hashable] = None,
        mode: str = "corner",
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
        self.mode = mode


class ObjectPoints2Boxes(AbstractTransform):
    def __init__(
        self,
        data_key: Hashable,
        box_coord_key: Hashable,
        box_label_key: Hashable,
        point_key: Hashable,
        label_key: Optional[Hashable] = None,
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
        """
        super().__init__()
        self.data_key = data_key
        self.box_coord_key = box_coord_key
        self.box_label_key = box_label_key
        self.point_key = point_key
        self.label_key = label_key
