"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import typing
import itertools
import numpy as np

from loguru import logger

from skimage.measure import regionprops
import SimpleITK as sitk


def center_crop_object_mask(mask: np.ndarray, cshape: typing.Union[tuple, int],
                            ) -> typing.List[tuple]:
    """
    Creates indices to crop patches around individual objects in mask

    Args
        mask: mask where objects have diffrent numbers. Objects need to be numbered
            consequtively from one to n, with 0 as background.
        cshape: size of individual crops. Needs to be divisible by two.
            Otherwise crops do not have the expected size.
            If cshape is a int, crops will have the same size in every dimension.

    Returns
        list[tuple]: each crop generates one tuple with indices

    Raises
        TypeError: raised if mask and patches define different dimensionalities
        TypeError: raised if `cshape` is larger than mask

    See Also
        :func:`save_get_crop`

    Warnings
        The returned crops are not checked for image boundaries. Slices
        with negative indices and indices which extend over the mask boundaries
        are possible! To correct for this, use `save_get_crop` which handles
        this exceptions.
    """
    if isinstance(cshape, int):
        cshape = tuple([cshape] * mask.ndim)

    if mask.ndim != len(cshape):
        raise TypeError("Size of crops needs to be defined for "
                        "every dimension")
    if any(np.subtract(mask.shape, cshape) < 0):
        raise TypeError("Patches must be smaller than data.")

    if mask.max() == 0:
        # no objects in mask
        return []

    all_centroids = [i['centroid'] for i in regionprops(mask.astype(np.int32))]
    crops = []
    for centroid in all_centroids:
        crops.append(tuple(slice(int(c) - (s // 2), int(c) + (s // 2))
                           for c, s in zip(centroid, cshape)))
    return crops


def center_crop_object_seg(seg: np.ndarray, cshape: typing.Union[tuple, int],
                           **kwargs) -> typing.List[tuple]:
    """
    Creates indices to crop patches around individual objects in segmentation.
    Objects are determined by region growing with connected threshold.

    Args
        seg: semantic segmentation of objects.
        cshape: size of individual crops. Needs to be divisible by two.
            Otherwise crops do not have the expected size.
            If cshape is a int, crops will have the same size in every dimension.
        kwargs: additional keyword arguments passed to `center_crop_objects_mask`

    Returns
        list[tuple]: each crop generates one tuple with indices

    See Also
        :func:`save_get_crop`

    Warnings
        The returned crops are not checked for image boundaries. Slices
        with negative indices and indices which extend over the mask boundaries
        are possible! To correct for this, use `save_get_crop` which handles
        this exceptions.
    """
    _mask, _ = create_mask_from_seg(seg)
    return center_crop_object_mask(_mask, cshape=cshape, **kwargs)


def create_mask_from_seg(seg: np.ndarray) -> typing.Tuple[np.ndarray, list]:
    """
    Create a mask where objects are enumerated from 1, ..., n.
    Objects are determined by region growing with connected threshold.

    Args
        seg: semantic segmentation array

    Returns
        np.ndarray: mask with objects
        list: classes to objects (ascending order)
    """
    _seg = np.copy(seg).astype(np.int32)
    _seg_sitk = sitk.GetImageFromArray(_seg)
    _mask = np.zeros_like(seg).astype(np.int32)
    _obj_cls = []
    _obj = 1

    while _seg.max() > 0:
        # choose one seed in segmentation
        seed = np.transpose(np.nonzero(_seg))[0]
        # invert coordinates for sitk
        seed_sitk = tuple(seed[:: -1].tolist())
        seed = tuple(seed)
        # region growing
        seg_con = sitk.ConnectedThreshold(_seg_sitk,
                                          seedList=[seed_sitk],
                                          lower=int(_seg[seed]),
                                          upper=int(_seg[seed]))
        seg_con = sitk.GetArrayFromImage(seg_con).astype(bool)

        # add object to mask
        _mask[seg_con] = _obj
        _obj_cls.append(_seg[seed])
        # remove object from segmentation
        _seg[seg_con] = 0

        _obj += 1

        # objects should never overlap
        assert _mask.max() < _obj
    return _mask, _obj_cls


def create_grid(cshape: typing.Union[typing.Sequence[int], int],
                dshape: typing.Sequence[int],
                overlap: typing.Union[typing.Sequence[int], int] = 0,
                mode='fixed',
                center_boarder: bool = False,
                **kwargs,
                ) -> typing.List[typing.Tuple[slice]]:
    """
    Create indices for a grid

    Args
        cshape: size of individual patches
        dshape: shape of data
        overlap: overlap between patches. If `overlap` is an integer is is applied
            to all dimensions.
        mode: defines how borders should be handled, by default 'fixed'.
            `fixed` created patches without special handling of borders, thus
            the last patch might exceed `dshape`
            `symmetric` moves patches such that the the first and last patch are
            equally overlapping of dshape (when combined with padding, the last and
            first patch would have the same amount of padding)
        center_boarder: adds additional crops at the boarders which have the
            boarder as their center

    Returns
        typing.List[slice]: slices to extract patches

    Raises
        TypeError: raised if `cshape` and `dshape` do not have the same length
        TypeError: raised if `overlap` and `dshape` do not have the same length
        TypeError: raised if `cshape` is larger than `dshape`
        TypeError: raised if `overlap` is larger than `cshape`

    Warnings
        The returned crops are can exceed the image boundaries. Slices
        with negative indices and indices which extend over the image
        boundary at the start. To correct for this, use `save_get_crop`
        which handles exceptions at borders.
    """
    _mode_fn = {
        "fixed": _fixed_slices,
        "symmetric": _symmetric_slices,
    }

    if len(dshape) == 3 and len(cshape) == 2:
        logger.info("Creating 2d grid.")
        slices_3d = dshape[0]
        dshape = dshape[1:]
    else:
        slices_3d = None

    # create tuples from shapes
    if isinstance(cshape, int):
        cshape = tuple([cshape] * len(dshape))
    if isinstance(overlap, int):
        overlap = tuple([overlap] * len(dshape))

    # check shapes
    if len(cshape) != len(dshape):
        raise TypeError(
            "cshape and dshape must be defined for same dimensionality.")
    if len(overlap) != len(dshape):
        raise TypeError(
            "overlap and dshape must be defined for same dimensionality.")
    if any(np.subtract(dshape, cshape) < 0):
        axes = np.nonzero(np.subtract(dshape, cshape) < 0)
        logger.warning(f"Found patch size which is bigger than data: data {dshape} patch {cshape}")
    if any(np.subtract(cshape, overlap) < 0):
        raise TypeError("Overlap must be smaller than size of patches.")

    grid_slices = [_mode_fn[mode](psize, dlim, ov, **kwargs)
                   for psize, dlim, ov in zip(cshape, dshape, overlap)]

    if center_boarder:
        for idx, (psize, dlim, ov) in enumerate(zip(cshape, dshape, overlap)):
            lower_bound_start = int(-0.5 * psize)
            upper_bound_start = dlim - int(0.5 * psize)
            grid_slices[idx] = tuple([
                slice(lower_bound_start, lower_bound_start + psize),
                *grid_slices[idx],
                slice(upper_bound_start, upper_bound_start + psize),
            ])

    if slices_3d is not None:
        grid_slices = [tuple([slice(i, i + 1) for i in range(slices_3d)])] + grid_slices
    grid = list(itertools.product(*grid_slices))
    return grid


def _fixed_slices(psize: int, dlim: int, overlap: int, start: int = 0) -> typing.Tuple[slice]:
    """
    Creates fixed slicing of a single axis. Only last patch exceeds dlim.

    Args
        psize: size of patch
        dlim: size of data
        overlap: overlap between patches
        start: where to start patches, by default 0

    Returns
        typing.List[slice]: ordered slices for a single axis
    """
    upper_limit = 0
    lower_limit = start
    idx = 0
    crops = []

    while upper_limit < dlim:
        if idx != 0:
            lower_limit = lower_limit - overlap

        upper_limit = lower_limit + psize
        crops.append(slice(lower_limit, upper_limit))
        lower_limit = upper_limit
        idx += 1
    return tuple(crops)


def _symmetric_slices(psize: int, dlim: int, overlap: int) -> typing.Tuple[slice]:
    """
    Creates symmetric slicing of a single axis. First and last patch exceed
    data borders.

    Args
        psize: size of patch
        dlim: size of data
        overlap: overlap between patches
        start: where to patches, by default 0

    Returns
        typing.List[slice]: ordered slices for a single axis
    """
    if psize >= dlim:
        return _fixed_slices(psize, dlim, overlap, start=-(psize - dlim) // 2)

    pmod = dlim % (psize - overlap)
    start = (pmod - psize) // 2
    return _fixed_slices(psize, dlim, overlap, start=start)


def save_get_crop(data: np.ndarray,
                  crop: typing.Sequence[slice],
                  mode: str = "shift",
                  **kwargs,
                  ) -> typing.Tuple[np.ndarray,
                                    typing.Tuple[int],
                                    typing.Tuple[slice]]:
    """
    Safely extract crops from data

    Args
        data: list or tuple with data where patches are extracted from
        crop: contains the coordiates of a single crop as slices
        mode: Handling of borders when crops are outside of data, by default "shift".
            Following modes are supported: "shift" crops are shifted inside the
            data | other modes are identical to `np.pad`
        kwargs: additional keyword arguments passed to `np.pad`

    Returns
        list[np.ndarray]: crops from data
        Tuple[int]: origin offset of crop with regard to data origin (can be
            used to offset bounding boxes)
        Tuple[slice]: crop from data used to extract information

    See Also
        :func:`center_crop_objects_mask`, :func:`center_crop_objects_seg`

    Warnings
        This functions only supports positive indexing. Negative indices are
        interpreted like they were outside the lower boundary!
    """
    if len(crop) > data.ndim:
        raise TypeError(
            "crop must have smaller or same dimensionality as data.")
    if mode == 'shift':
        # move slices if necessary
        return _shifted_crop(data, crop)
    else:
        # use np.pad if necessary
        return _padded_crop(data, crop, mode, **kwargs)


def _shifted_crop(data: np.ndarray,
                  crop: typing.Sequence[slice],
                  ) -> typing.Tuple[np.ndarray,
                                    typing.Tuple[int],
                                    typing.Tuple[slice]]:
    """
    Created shifted crops to handle borders

    Args
        data: crop is extracted from data
        crop: defines boundaries of crops

    Returns
        List[np.ndarray]: list of crops
        Tuple[int]: origin offset of crop with regard to data origin (can be
            used to offset bounding boxes)
        Tuple[slice]: crop from data used to extract information

    Raises
        TypeError: raised if patchsize is bigger than data

    Warnings
        This functions only supports positive indexing. Negative indices are
        interpreted like they were outside the lower boundary!
    """
    shifted_crop = []
    dshape = tuple(data.shape)

    # index from back, so batch and channel dimensions must not be defined
    axis = data.ndim - len(crop)

    for idx, crop_dim in enumerate(crop):
        if crop_dim.start < 0:
            # start is negative, thus it is subtracted from stop
            new_slice = slice(0, crop_dim.stop - crop_dim.start, crop_dim.step)
            if new_slice.stop > dshape[axis + idx]:
                raise RuntimeError(
                    "Patch is bigger than entire data. shift "
                    "is not supported in this case.")
            shifted_crop.append(new_slice)
        elif crop_dim.stop > dshape[axis + idx]:
            new_slice = \
                slice(crop_dim.start - (crop_dim.stop - dshape[axis + idx]),
                      dshape[axis + idx], crop_dim.step)
            if new_slice.start < 0:
                raise RuntimeError(
                    "Patch is bigger than entire data. shift "
                    "is not supported in this case.")
            shifted_crop.append(new_slice)
        else:
            shifted_crop.append(crop_dim)
    origin = [int(x.start) for x in shifted_crop]
    return data[tuple([..., *shifted_crop])], origin, shifted_crop


def _padded_crop(data: np.ndarray,
                 crop: typing.Sequence[slice],
                 mode: str,
                 **kwargs,
                 ) -> typing.Tuple[np.ndarray,
                                   typing.Tuple[int],
                                   typing.Tuple[slice]]:
    """
    Extract patch from data and pad accordingly

    Args
        data: crop is extracted from data
        crop: defines boundaries of crops
        mode: mode for padding. See `np.pad` for more details
        kwargs: additional keyword arguments passed to :func:`np.pad`

    Returns
        typing.List[np.ndarray]: list of crops
        Tuple[int]: origin offset of crop with regard to data origin (can be
            used to offset bounding boxes)
        Tuple[slice]: crop from data used to extract information
    """
    clipped_crop = []
    dshape = tuple(data.shape)

    # index from back, so batch and channel dimensions must not be defined
    axis = data.ndim - len(crop)
    padding = [(0, 0)] * axis if axis > 0 else []

    for idx, crop_dim in enumerate(crop):
        lower_pad = 0
        upper_pad = 0
        lower_bound = crop_dim.start
        upper_bound = crop_dim.stop

        # handle lower bound
        if lower_bound < 0:
            lower_pad = -lower_bound
            lower_bound = 0
        # handle upper bound
        if upper_bound > dshape[axis + idx]:
            upper_pad = upper_bound - dshape[axis + idx]
            upper_bound = dshape[axis + idx]
        padding.append((lower_pad, upper_pad))
        clipped_crop.append(slice(lower_bound, upper_bound, crop_dim.step))
    origin = [int(x.start) for x in crop]
    return (np.pad(data[tuple([..., *clipped_crop])], pad_width=padding, mode=mode, **kwargs),
            origin,
            clipped_crop,
            )

