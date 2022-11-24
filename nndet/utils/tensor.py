# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict

import torch
import re
import numpy as np
from torch import Tensor

from collections import abc
from torch._six import string_classes
from typing import Sequence, Union, Any, Mapping, Callable, List

np_str_obj_array_pattern = re.compile(r'[SaUO]')


def make_onehot_batch(labels: torch.Tensor, n_classes: torch.Tensor) -> torch.Tensor:
    """
    Create onehot encoding of labels
    
    Args:
        labels: label tensor to enode [N, dims]
        n_classes: number of classes
    
    Returns:
        Tensor: onehot encoded tensor [N, C, dims]; N: batch size,
            C: number of classes, dims: spatial dimensions
    """
    idx = labels.to(dtype=torch.long)

    new_shape = [labels.shape[0], n_classes, *labels.shape[1:]]
    labels_onehot = torch.zeros(*new_shape, device=labels.device,
                                dtype=labels.dtype)
    labels_onehot.scatter_(1, idx.unsqueeze(dim=1), 1)
    return labels_onehot


def to_dtype(inp: Any, dtype: Callable) -> Any:
    """
    helper function to convert a sequence of arguments to a specific type

    Args:
        inp (Any): any object which can be converted by dtype, if sequence is detected
            dtype is applied to individual arguments
        dtype (Callable): callable to convert arguments

    Returns:
        Any: converted input
    """
    if isinstance(inp, Sequence):
        return type(inp)([dtype(i) for i in inp])
    else:
        return dtype(inp)


def to_device(inp: Union[Sequence[torch.Tensor], torch.Tensor, Mapping[str, torch.Tensor]],
              device: Union[torch.device, str], detach: bool = False,
              **kwargs) -> \
        Union[Sequence[torch.Tensor], torch.Tensor, Mapping[str, torch.Tensor]]:
    """
    Push tensor or sequence of tensors to device

    Args:
        inp (Union[Sequence[torch.Tensor], torch.Tensor]): tensor or sequence of tensors
        device (Union[torch.device, str]): target device
        detach: detach tensor before moving it to new device
        **kwargs: keyword arguments passed to `to` function

    Returns:
        Union[Sequence[torch.Tensor], torch.Tensor]: tensor or seq. of tenors at target device
    """
    if isinstance(inp, torch.Tensor):
        if detach:
            return inp.detach().to(device=device, **kwargs)
        else:
            return inp.to(device=device, **kwargs)
    elif isinstance(inp, Sequence):
        old_type = type(inp)
        return old_type([to_device(i, device=device, detach=detach, **kwargs)
                         for i in inp])
    elif isinstance(inp, Mapping):
        old_type = type(inp)
        return old_type({key: to_device(item, device=device, detach=detach, **kwargs)
                         for key, item in inp.items()})
    else:
        return inp


def to_numpy(inp: Union[Sequence[torch.Tensor], torch.Tensor, Any]) -> \
        Union[Sequence[np.ndarray], np.ndarray, Any]:
    """
    Convert a tensor or sequence of tensors to numpy array/s

    Args:
        inp (Union[Sequence[torch.Tensor], torch.Tensor]): tensor or sequence of tensors

    Returns:
        Union[Sequence[np.ndarray], np.ndarray]: array or seq. of arrays at target device
         (non tensor entries are forwarded as they are)
    """
    if isinstance(inp, (tuple, list)):
        old_type = type(inp)
        return old_type([to_numpy(i) for i in inp])
    elif isinstance(inp, dict) and not isinstance(inp, defaultdict):
        old_type = type(inp)
        return old_type({k: to_numpy(i) for k, i in inp.items()})
    elif isinstance(inp, torch.Tensor):
        return inp.detach().cpu().numpy()
    else:
        return inp


def to_tensor(inp: Any) -> Any:
    """
    Convert arrays, seq, mappings to torch tensor
    https://github.com/pytorch/pytorch/blob/f522bde1213fcc46f2857f79be7b1b01ddf302a6/torch/utils/data/_utils/collate.py

    Args:
        inp: np.ndarrays, mappings, sequences are converted to tensors,
            rest is passed through this function

    Returns:
        Any: converted data
    """
    elem_type = type(inp)
    if isinstance(inp, torch.Tensor):
        return inp
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        # array of string classes and object
        if elem_type.__name__ == 'ndarray' \
                and np_str_obj_array_pattern.search(inp.dtype.str) is not None:
            return inp
        return torch.as_tensor(inp)
    elif isinstance(inp, abc.Mapping):
        return {key: to_tensor(inp[key]) for key in inp}
    elif isinstance(inp, tuple) and hasattr(inp, '_fields'):  # namedtuple
        return elem_type(*(to_tensor(d) for d in inp))
    elif isinstance(inp, abc.Sequence) and not isinstance(inp, string_classes):
        return [to_tensor(d) for d in inp]
    else:
        return inp


def cat(t: Union[List[Tensor], Tensor], *args, **kwrags):
    if not isinstance(t, (list, Tensor)):
        raise ValueError(f"Can only concatenate lists and tensors.")

    if isinstance(t, Tensor):
        return t
    elif len(t) == 1:
        return t[0]
    else:
        return torch.cat(t, *args, **kwrags)
