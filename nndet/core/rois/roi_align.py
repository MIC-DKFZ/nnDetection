from typing import Sequence, Tuple, Union

import torch
from torch import Tensor

from nndet._C import roi_align as roi_align_3d
from nndet.core.boxes.ops import expand_to_boxes


def roi_align(
    input: Tensor,
    boxes: Tensor,
    output_size: Tuple[int],
    spatial_scale: Union[float, Tuple[float]] = 1.0,
    sampling_ratio: int = -1,
    aligned: bool = False,
) -> Tensor:
    # TODO: aligned implementation
    # TODO:
    assert input.device == boxes.device

    # apply scaling here, will be moved to cuda function down the road
    if isinstance(spatial_scale, Sequence):
        _scale = torch.Tensor(spatial_scale).to(boxes)
        boxes[:, 1:] = boxes[:, 1:] * expand_to_boxes(_scale)
    else:
        boxes[:, 1:] = boxes[:, 1:] * spatial_scale

    if input.is_cuda:
        if boxes.shape[1] == 4:
            raise NotImplementedError
        else:
            pool_fn = roi_align_3d
    else:
        raise NotImplementedError

    boxes = boxes.to(dtype=input.dtype)

    # print(boxes)
    return pool_fn(
        input.contiguous(),
        boxes.contiguous(),
        1.0,
        output_size[0],
        output_size[1],
        output_size[2],
        sampling_ratio,
    )
