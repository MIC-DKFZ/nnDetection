from abc import abstractmethod
from typing import Dict, List, Union

import numpy as np

from nndet.core.boxes.ops_np import box_size_np
from nndet.utils.typing import ND_TUPLE_INT


class FGCrop:
    patch_size_generator: ND_TUPLE_INT
    need_to_pad: ND_TUPLE_INT
    patch_size_final: ND_TUPLE_INT

    @abstractmethod
    def get_fg_crop(
        self,
        case_data: np.ndarray,
        case_seg: np.ndarray,
        properties: dict,
        case_id: str,
        instance_id: int,
        candidates: Union[Dict, None] = None,
    ) -> List[slice]:
        """
        Sample foreground patches from precomputed boxes

        Args:
            case_data: case data (this should be a memmap!)
            case_seg: case segmentation (this should be a memmap!)
            properties: properties of case
            case_id: identifier of case
            instance_id: instance index to sample
            candidates: candidate positions to sample foreground from.
                Should not be None for this case.

        Returns:
            List[slice]: determined crop
        """
        raise NotImplementedError


class InsideFGCrop3D(FGCrop):
    def get_fg_crop(
        self,
        case_data: np.ndarray,
        case_seg: np.ndarray,
        properties: dict,
        case_id: str,
        instance_id: int,
        candidates: Union[Dict, None] = None,
    ) -> List[slice]:
        """
        Sample foreground patches from precomputed boxes

        Args:
            case_data: case data (this should be a memmap!)
            case_seg: case segmentation (this should be a memmap!)
            properties: properties of case
            case_id: identifier of case
            instance_id: instance index to sample
            candidates: candidate positions to sample foreground from.
                Should not be None for this case.

        Returns:
            List[slice]: determined crop
        """
        assert candidates is not None
        # some instances might get lost during resampling so we need to find the correct index
        idx = candidates["instances"].index(instance_id)
        box = candidates["boxes"][idx]  # [6]
        origin0 = np.random.randint(int(box[0]) + 1, int(box[2])) - (
            self.patch_size_generator[0] // 2
        )
        origin1 = np.random.randint(int(box[1]) + 1, int(box[3])) - (
            self.patch_size_generator[1] // 2
        )
        origin2 = np.random.randint(int(box[4]) + 1, int(box[5])) - (
            self.patch_size_generator[2] // 2
        )
        return [
            slice(origin0, origin0 + self.patch_size_generator[0]),
            slice(origin1, origin1 + self.patch_size_generator[1]),
            slice(origin2, origin2 + self.patch_size_generator[2]),
        ]


class OffsetFGCrop3D(FGCrop):
    offset_prob: float
    offset_magn: float

    def get_fg_crop(
        self,
        case_data: np.ndarray,
        case_seg: np.ndarray,
        properties: dict,
        case_id: str,
        instance_id: int,
        candidates: Union[Dict, None],
    ) -> List[slice]:
        """
        Sample foreground patches from precomputed boxes

        Args:
            case_data: case data (this should be a memmap!)
            case_seg: case segmentation (this should be a memmap!)
            properties: properties of case
            case_id: identifier of case
            instance_id: instance index to sample
            candidates: candidate positions to sample foreground from.
                Should not be None for this case.

        Returns:
            List[slice]: determined crop
        """
        spatial_shape = case_data.shape[1:]
        # some instances might get lost during resampling so we need to find the correct index
        idx = candidates["instances"].index(instance_id)
        box = candidates["boxes"][[idx]]  # [1, 6]
        box_size = box_size_np(box)[0]
        box = box[0]

        origins = []
        offset_rand = np.random.rand(1)
        for i, (ilb, ulb) in enumerate([(0, 2), (1, 3), (4, 5)]):
            if (offset_rand > self.offset_prob) or (
                box_size[i] >= self.patch_size_final[i]
            ):
                # no offset prob | object is bigger than patch
                center = np.random.randint(int(box[ilb]) + 1, int(box[ulb]))
                origins.append(center - (self.patch_size_generator[i] // 2))
            elif spatial_shape[i] <= self.patch_size_generator[i]:
                # patch larger than scan
                # we center the slice and pad the rest
                origins.append(-(self.need_to_pad[i] // 2))
            else:
                # create best effort offset
                patch_upper_bound = spatial_shape[i] - self.patch_size_final[i]
                lower_bound = np.clip(
                    box[ilb] - (self.patch_size_final[i] - box_size[i]),
                    a_min=0,
                    a_max=patch_upper_bound,
                )
                upper_bound = np.clip(box[ilb], a_min=0, a_max=patch_upper_bound)

                _d = (upper_bound - lower_bound) / 2
                lower_bound = lower_bound + round((1.0 - self.offset_magn) * _d)
                upper_bound = upper_bound - round((1.0 - self.offset_magn) * _d)
                assert upper_bound >= lower_bound

                if lower_bound == upper_bound:
                    _origin = int(lower_bound)
                else:
                    _origin = np.random.randint(lower_bound, upper_bound)
                origins.append(_origin - (self.need_to_pad[i] // 2))

        assert len(origins) == 3
        return [
            slice(origins[0], origins[0] + self.patch_size_generator[0]),
            slice(origins[1], origins[1] + self.patch_size_generator[1]),
            slice(origins[2], origins[2] + self.patch_size_generator[2]),
        ]


class OffsetFGCrop3DV2(FGCrop):
    """
    Fixes center crop
    Uses a different offset magnitude mechanism; if magnitude is set to 0 the
    object is centered inside the patch. if magnitude is set to 1 offset
    is maximized. Float values in between will dynamically vary the offset
    strength.
    """

    offset_prob: float
    offset_magn: float
    max_size_pct: float

    def get_fg_crop(
        self,
        case_data: np.ndarray,
        case_seg: np.ndarray,
        properties: dict,
        case_id: str,
        instance_id: int,
        candidates: Union[Dict, None],
    ) -> List[slice]:
        """
        Sample foreground patches from precomputed boxes
        (fixes the centering mechanism)

        Args:
            case_data: case data (this should be a memmap!)
            case_seg: case segmentation (this should be a memmap!)
            properties: properties of case
            case_id: identifier of case
            instance_id: instance index to sample
            candidates: candidate positions to sample foreground from.
                Should not be None for this case.

        Returns:
            List[slice]: determined crop
        """
        spatial_shape = case_data.shape[1:]
        # some instances might get lost during resampling so we need to find the correct index
        idx = candidates["instances"].index(instance_id)
        box = candidates["boxes"][[idx]]  # [1, 6]
        box_size = box_size_np(box)[0]
        box = box[0]

        origins = []
        offset_rand = np.random.rand(1)
        for i, (ilb, ulb) in enumerate([(0, 2), (1, 3), (4, 5)]):
            if (offset_rand > self.offset_prob) or (
                box_size[i] >= (self.max_size_pct * self.patch_size_final[i])
            ):
                # no offset prob | object is bigger than patch
                # print("inbox", box_size, self.patch_size_final)
                origins.append(
                    self._inside_box(
                        ps=self.patch_size_final[i],
                        psg=self.patch_size_generator[i],
                        box_lower=int(box[ilb]),
                        box_upper=int(box[ulb]),
                    )
                )
            elif spatial_shape[i] <= self.patch_size_generator[i]:
                # print("center")
                # patch larger than scan
                # we center the slice and pad the rest
                origins.append(
                    self._center_data(
                        ps=self.patch_size_final[i],
                        psg=self.patch_size_generator[i],
                        spatial_size=spatial_shape[i],
                    )
                )
            else:
                # print("offset")
                # create best effort offset
                origins.append(
                    self._offset_box(
                        ps=self.patch_size_final[i],
                        ntp=self.need_to_pad[i],
                        spatial_size=spatial_shape[i],
                        box_lower=int(box[ilb]),
                        box_upper=int(box[ulb]),
                    )
                )

        assert len(origins) == 3
        return [
            slice(origins[0], origins[0] + self.patch_size_generator[0]),
            slice(origins[1], origins[1] + self.patch_size_generator[1]),
            slice(origins[2], origins[2] + self.patch_size_generator[2]),
        ]

    def _center_data(
        self,
        ps: int,
        psg: int,
        spatial_size: int,
    ) -> int:
        """
        Center data inside generator patch.
        All inputs to this function refer to one axis.

        Args:
            ps: patch size for network (after aug crop)
            psg: patch size to extract by dataloader
            spatial_size: size of data

        Returns:
            int: lower boundary of patch to extract
        """
        center = spatial_size // 2
        return center - (psg // 2)

    def _inside_box(
        self,
        ps: int,
        psg: int,
        box_lower: int,
        box_upper: int,
    ) -> int:
        """
        Select random point inside box as center of patch
        All inputs to this function refer to one axis.

        Args:
            ps: patch size for network (after aug crop)
            psg: patch size to extract by dataloader
            box_lower: lower bound of box
            box_upper: upper bound of box

        Returns:
            int: lower boundary of patch to extract
        """
        center = np.random.randint(box_lower + 1, box_upper)
        return center - (psg // 2)

    def _offset_box(
        self,
        ps: int,
        ntp: int,
        spatial_size: int,
        box_lower: int,
        box_upper: int,
    ) -> int:
        """
        Try to offset the object randomly while keeping the whole object
        inside the patch.
        All inputs to this function refer to one axis.

        Args:
            ps: patch size for network (after aug crop)
            ntp: amount that needs to be padded (difference between network
                patch size and dataloader patch size)
            spatial_size: size of data
            box_lower: lower bound of box
            box_upper: upper bound of box
            box_size: size of bounding box

        Returns:
            int: lower boundary of patch to extract
        """
        patch_upper_bound = spatial_size - ps  # keep patch inside scan
        lower_bound = np.clip(
            # lower + ps = box_upper => lower = box_upper - ps
            box_upper - ps,
            a_min=-1,
            a_max=patch_upper_bound,
        )
        upper_bound = np.clip(box_lower, a_min=-1, a_max=patch_upper_bound)

        if self.offset_magn < 1.0:
            # if offset margin is smaller 1.0, the bound are moved towards the centralized position
            centered = box_lower - (ps / 2) + (box_upper - box_lower) / 2
            lower_bound = lower_bound + round(
                (1.0 - self.offset_magn) * (centered - lower_bound)
            )
            upper_bound = upper_bound - round(
                (1.0 - self.offset_magn) * (upper_bound - centered)
            )
        assert upper_bound >= lower_bound

        if lower_bound == upper_bound:
            _origin = int(lower_bound)
        else:
            _origin = np.random.randint(lower_bound, upper_bound)
        return _origin - (ntp // 2)


class OffsetFGCrop2D(FGCrop):
    def get_fg_crop(
        self,
        case_data: np.ndarray,
        case_seg: np.ndarray,
        properties: dict,
        case_id: str,
        instance_id: int,
        candidates: Union[Dict, None],
    ) -> List[slice]:
        """
        Sample foreground patches from precomputed boxes

        Args:
            case_data: case data (this should be a memmap!)
            case_seg: case segmentation (this should be a memmap!)
            properties: properties of case
            case_id: identifier of case
            instance_id: instance index to sample
            candidates: candidate positions to sample foreground from.
                Should not be None for this case.

        Returns:
            List[slice]: determined crop
        """
        spatial_shape = case_data.shape[2:]
        # some instances might get lost during resampling so we need to find the correct index
        idx = candidates["instances"].index(instance_id)
        box = candidates["boxes"][[idx]]  # [1, 6]
        box_size = box_size_np(box)[0, 1:]
        box = box[0]

        slice_idx = np.random.randint(int(box[0]) + 1, int(box[2]))

        origins = []
        for i, (ib, ib2) in enumerate([(1, 3), (4, 5)]):
            if (
                spatial_shape[i] <= self.patch_size_generator[i]
            ):  # patch larger than scan
                # we center the slice and pad the rest
                origins.append(-(self.need_to_pad[i] // 2))
            elif (
                box_size[i] >= self.patch_size_final[i]
            ):  # selected instance is larger than patch
                # we can not offset, we select our center point inside the bounding box and hope for the best
                center = np.random.randint(int(box[ib]) + 1, int(box[ib2]))
                origins.append(center - (self.patch_size_generator[i] // 2))
            else:  # create best effort offset
                patch_upper_bound = spatial_shape[i] - self.patch_size_final[i]
                lower_bound = np.clip(
                    box[ib] - (self.patch_size_final[i] - box_size[i]),
                    a_min=0,
                    a_max=patch_upper_bound,
                )
                upper_bound = np.clip(box[ib], a_min=0, a_max=patch_upper_bound)

                if lower_bound == upper_bound:
                    _origin = int(lower_bound)
                else:
                    _origin = np.random.randint(lower_bound, upper_bound)

                origins.append(_origin - (self.need_to_pad[i] // 2))

        return [
            slice(slice_idx, slice_idx + 1),
            slice(origins[0], origins[0] + self.patch_size_generator[0]),
            slice(origins[1], origins[1] + self.patch_size_generator[1]),
        ]
