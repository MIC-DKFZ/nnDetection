from abc import abstractmethod
from typing import Dict, List, Union

import numpy as np


class BGCrop:
    @abstractmethod
    def get_bg_crop(
        self,
        case_data: np.ndarray,
        case_seg: np.ndarray,
        properties: dict,
        case_id: str,
        candidates: Union[Dict, None],
    ) -> List[slice]:
        """
        Extract slices for (random) background crop

        Args:
            case_data: case data (this should be a memmap!)
            case_seg: case segmentation (this should be a memmap!)
            properties: properties of case
            case_id: identifier of case
            candidates: foreground candidates. Is not used in this
                specific implementation and thus None

        Returns:
            List[slice]: determined crop
        """
        raise NotImplementedError


class RandomBGCrop3D(BGCrop):
    def get_bg_crop(
        self,
        case_data: np.ndarray,
        case_seg: np.ndarray,
        properties: dict,
        case_id: str,
        candidates: Union[Dict, None],
    ) -> List[slice]:
        """
        Extract slices for (random) background crop

        Args:
            case_data: case data (this should be a memmap!)
            case_seg: case segmentation (this should be a memmap!)
            properties: properties of case
            case_id: identifier of case
            candidates: foreground candidates. Is not used in this
                specific implementation and thus None

        Returns:
            List[slice]: determined crop
        """
        data_shape = case_data.shape[1:]

        crop = []
        for ps, ds, _pad in zip(
            self.patch_size_generator, data_shape, self.need_to_pad
        ):
            pad = _pad
            if pad + ds < ps:
                pad = ps - ds
            origin = np.random.randint(
                -(pad // 2), ds + (pad // 2) + (pad % 2) - ps + 1
            )
            crop.append(slice(origin, origin + ps))
        return crop
