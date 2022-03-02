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
    offset_magn: float  # TODO

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
            if offset_rand > self.offset_prob:
                # no offset should be applied
                origins.append(
                    np.random.randint(int(box[0]) + 1, int(box[2]))
                    - (self.patch_size_generator[0] // 2)
                )
                continue

            if (
                spatial_shape[i] <= self.patch_size_generator[i]
            ):  # patch larger than scan
                # we center the slice and pad the rest
                origins.append(-(self.need_to_pad[i] // 2))
            elif (
                box_size[i] >= self.patch_size_final[i]
            ):  # selected instance is larger than patch
                # we can not offset, we select our center point inside the bounding box and hope for the best
                center = np.random.randint(int(box[ilb]) + 1, int(box[ulb]))
                origins.append(center - (self.patch_size_generator[0] // 2))
            else:  # create best effort offset
                patch_upper_bound = spatial_shape[i] - self.patch_size_final[i]
                lower_bound = np.clip(
                    box[ilb] - (self.patch_size_final[i] - box_size[i]),
                    a_min=0,
                    a_max=patch_upper_bound,
                )
                upper_bound = np.clip(box[ilb], a_min=0, a_max=patch_upper_bound)

                if lower_bound == upper_bound:
                    _origin = int(lower_bound)
                else:
                    _origin = np.random.randint(lower_bound, upper_bound)

                origins.append(_origin - (self.need_to_pad[i] // 2))

        return [
            slice(origins[0], origins[0] + self.patch_size_generator[0]),
            slice(origins[1], origins[1] + self.patch_size_generator[1]),
            slice(origins[2], origins[2] + self.patch_size_generator[2]),
        ]


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
                origins.append(center - (self.patch_size_generator[0] // 2))
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
