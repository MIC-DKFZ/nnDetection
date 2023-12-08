# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

"""
Direct copy of nnU-Net, all credits to https://github.com/MIC-DKFZ/nnUNet
"""

from batchgenerators.transforms.abstract_transforms import AbstractTransform


class MaskTransform(AbstractTransform):
    def __init__(self, dct_for_where_it_was_used, mask_idx_in_seg=1, set_outside_to=0, data_key="data", seg_key="seg"):
        """
        data[mask < 0] = 0
        Sets everything outside the mask to 0. CAREFUL! outside is defined as < 0, not =0 (in the Mask)!!!

        Direct copy of nnU-Net, all credits to https://github.com/MIC-DKFZ/nnUNet

        :param dct_for_where_it_was_used:
        :param mask_idx_in_seg:
        :param set_outside_to:
        :param data_key:
        :param seg_key:
        """
        self.dct_for_where_it_was_used = dct_for_where_it_was_used
        self.seg_key = seg_key
        self.data_key = data_key
        self.set_outside_to = set_outside_to
        self.mask_idx_in_seg = mask_idx_in_seg

    def __call__(self, **data_dict):
        seg = data_dict.get(self.seg_key)
        if seg is None or seg.shape[1] < self.mask_idx_in_seg:
            raise Warning("mask not found, seg may be missing or seg[:, mask_idx_in_seg] may not exist")
        data = data_dict.get(self.data_key)
        for b in range(data.shape[0]):
            mask = seg[b, self.mask_idx_in_seg]
            for c in range(data.shape[1]):
                if self.dct_for_where_it_was_used[c]:
                    data[b, c][mask < 0] = self.set_outside_to
        data_dict[self.data_key] = data
        return data_dict


def convert_3d_to_2d_generator(data_dict):
    """
    Direct copy of nnU-Net, all credits to https://github.com/MIC-DKFZ/nnUNet
    """
    shp = data_dict["data"].shape
    data_dict["data"] = data_dict["data"].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
    data_dict["orig_shape_data"] = shp
    shp = data_dict["seg"].shape
    data_dict["seg"] = data_dict["seg"].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
    data_dict["orig_shape_seg"] = shp
    return data_dict


def convert_2d_to_3d_generator(data_dict):
    """
    Direct copy of nnU-Net, all credits to https://github.com/MIC-DKFZ/nnUNet
    """
    shp = data_dict["orig_shape_data"]
    current_shape = data_dict["data"].shape
    data_dict["data"] = data_dict["data"].reshape((shp[0], shp[1], shp[2], current_shape[-2], current_shape[-1]))
    shp = data_dict["orig_shape_seg"]
    current_shape_seg = data_dict["seg"].shape
    data_dict["seg"] = data_dict["seg"].reshape((shp[0], shp[1], shp[2], current_shape_seg[-2], current_shape_seg[-1]))
    return data_dict


class Convert3DTo2DTransform(AbstractTransform):
    """
    Direct copy of nnU-Net, all credits to https://github.com/MIC-DKFZ/nnUNet
    """

    def __init__(self):
        pass

    def __call__(self, **data_dict):
        return convert_3d_to_2d_generator(data_dict)


class Convert2DTo3DTransform(AbstractTransform):
    """
    Direct copy of nnU-Net, all credits to https://github.com/MIC-DKFZ/nnUNet
    """

    def __init__(self):
        pass

    def __call__(self, **data_dict):
        return convert_2d_to_3d_generator(data_dict)
