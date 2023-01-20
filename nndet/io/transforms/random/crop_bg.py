from typing import List, Optional, Tuple

import numpy as np
from batchgenerators.augmentations.crop_and_pad_augmentations import (
    get_lbs_for_center_crop,
    get_lbs_for_random_crop,
)


def crop(
    data: np.ndarray,
    seg: Optional[np.ndarray] = None,
    crop_size=128,
    margins=(0, 0, 0),
    crop_type="center",
    pad_mode="constant",
    pad_kwargs={"constant_values": 0},
    pad_mode_seg="constant",
    pad_kwargs_seg={"constant_values": 0},
    points: Optional[List[np.ndarray]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[List[np.ndarray]]]:
    """
    Note: this is a direct adaptation from batchgenerators where points support
    was inserted and is only used for the spatial transform.

    crops data and seg (seg may be None) to crop_size. Whether this will be achieved via center or random crop is
    determined by crop_type. Margin will be respected only for random_crop and will prevent the crops form being closer
    than margin to the respective image border. crop_size can be larger than data_shape - margin -> data/seg will be
    padded with zeros in that case. margins can be negative -> results in padding of data/seg followed by cropping with
    margin=0 for the appropriate axes

    :param data: b, c, x, y(, z)
    :param seg:
    :param crop_size:
    :param margins: distance from each border, can be int or list/tuple of ints (one element for each dimension).
    Can be negative (data/seg will be padded if needed)
    :param crop_type: random or center
    :return:
    """
    if not isinstance(data, (list, tuple, np.ndarray)):
        raise TypeError("data has to be either a numpy array or a list")

    data_shape = tuple([len(data)] + list(data[0].shape))
    data_dtype = data[0].dtype
    dim = len(data_shape) - 2

    if seg is not None:
        seg_shape = tuple([len(seg)] + list(seg[0].shape))
        seg_dtype = seg[0].dtype

        if not isinstance(seg, (list, tuple, np.ndarray)):
            raise TypeError("data has to be either a numpy array or a list")

        assert all(
            [i == j for i, j in zip(seg_shape[2:], data_shape[2:])]
        ), "data and seg must have the same spatial " "dimensions. Data: %s, seg: %s" % (
            str(data_shape),
            str(seg_shape),
        )

    if type(crop_size) not in (tuple, list, np.ndarray):
        crop_size = [crop_size] * dim
    else:
        assert len(crop_size) == len(data_shape) - 2, (
            "If you provide a list/tuple as center crop make sure it has the same dimension as your " "data (2d/3d)"
        )

    if not isinstance(margins, (np.ndarray, tuple, list)):
        margins = [margins] * dim

    data_return = np.zeros([data_shape[0], data_shape[1]] + list(crop_size), dtype=data_dtype)
    if seg is not None:
        seg_return = np.zeros([seg_shape[0], seg_shape[1]] + list(crop_size), dtype=seg_dtype)
    else:
        seg_return = None
    points_return = None

    for b in range(data_shape[0]):
        data_shape_here = [data_shape[0]] + list(data[b].shape)
        if seg is not None:
            seg_shape_here = [seg_shape[0]] + list(seg[b].shape)

        if crop_type == "center":
            lbs = get_lbs_for_center_crop(crop_size, data_shape_here)
        elif crop_type == "random":
            lbs = get_lbs_for_random_crop(crop_size, data_shape_here, margins)
        else:
            raise NotImplementedError("crop_type must be either center or random")

        need_to_pad = [[0, 0]] + [
            [
                abs(min(0, lbs[d])),
                abs(min(0, data_shape_here[d + 2] - (lbs[d] + crop_size[d]))),
            ]
            for d in range(dim)
        ]

        lbs_offset = np.array(lbs)  # used for point offset

        # we should crop first, then pad -> reduces i/o for memmaps, reduces RAM usage and improves speed
        ubs = [min(lbs[d] + crop_size[d], data_shape_here[d + 2]) for d in range(dim)]
        lbs = [max(0, lbs[d]) for d in range(dim)]

        slicer_data = [slice(0, data_shape_here[1])] + [slice(lbs[d], ubs[d]) for d in range(dim)]
        data_cropped = data[b][tuple(slicer_data)]

        if seg_return is not None:
            slicer_seg = [slice(0, seg_shape_here[1])] + [slice(lbs[d], ubs[d]) for d in range(dim)]
            seg_cropped = seg[b][tuple(slicer_seg)]

        if any([i > 0 for j in need_to_pad for i in j]):
            data_return[b] = np.pad(data_cropped, need_to_pad, pad_mode, **pad_kwargs)
            if seg_return is not None:
                seg_return[b] = np.pad(seg_cropped, need_to_pad, pad_mode_seg, **pad_kwargs_seg)
        else:
            data_return[b] = data_cropped
            if seg_return is not None:
                seg_return[b] = seg_cropped

        if points is not None:
            points_return = [p - lbs_offset[None, None] for p in points]
    return data_return, seg_return, points_return
