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

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from nndet.io.transforms.base import AbstractTransform


class FindInstances(AbstractTransform):
    def __init__(
        self, instance_key: str, save_key: str = "present_instances", **kwargs
    ):
        super().__init__(grad=False)
        self.instance_key = instance_key
        self.save_key = save_key

    def forward(self, **data) -> dict:
        present_instances = []
        for instance_element in data[self.instance_key].split(1):
            tmp = instance_element.to(dtype=torch.int).unique(sorted=True)
            tmp = tmp[tmp > 0]
            present_instances.append(tmp)
        data[self.save_key] = present_instances
        return data


class Instances2Boxes(AbstractTransform):
    def __init__(
        self,
        instance_key: str,
        map_key: str,
        box_key: str,
        class_key: str,
        grad: bool = False,
        present_instances: Optional[str] = None,
        **kwargs,
    ):
        """
        Convert instance segmentation to bounding boxes

        Args:
            instance_key: key where instance segmentation is located
            map_key: key where mapping from instances to classes is located
                (should be a dict which keys(instances) to items(classes))
            box_key: key where boxes should be saved
            class_key: key where classes of instances will be saved
            grad: enable gradient computation inside transformation
            present_instances: key where precomputed present instances are
                saved. If `None` it will compute the present instance new.
        """
        super().__init__(grad=grad, **kwargs)
        self.class_key = class_key
        self.box_key = box_key
        self.map_key = map_key
        self.instance_key = instance_key
        self.present_instances = present_instances

    def forward(self, **data) -> dict:
        """
        Extract boxes from instances

        Args:
            data: batch dict

        Returns:
            dict: processed batch
        """
        data[self.box_key] = []
        data[self.class_key] = []
        for batch_idx, instance_element in enumerate(data[self.instance_key].split(1)):
            _present_instances = (
                data[self.present_instances][batch_idx]
                if self.present_instances is not None
                else None
            )
            _boxes, instance_idx = instances_to_boxes(
                instance_element,
                instance_element.ndim - 2,
                instances=_present_instances,
            )

            _classes = get_instance_class_from_properties(
                instance_idx, data[self.map_key][batch_idx]
            )
            _classes = _classes.to(device=_boxes.device)

            data[self.box_key].append(_boxes)
            data[self.class_key].append(_classes)
        return data


def instances_to_boxes(
    seg: Tensor,
    dim: int,
    instances: Optional[Sequence[int]] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Convert instance segmentation to bounding boxes (not batched)

    Args:
        seg: instance segmentation of individual classes [..., dims]
        dim: number of spatial dimensions to create bounding box for
            Conversion will always start from the last dimension. If `None`,
            all dimensions are used.

    Returns:
        Tensor: bounding boxes
            (x1, y1, x2, y2, (z1, z2)) List[Tensor[N, dim * 2]]
        Tensor: if `instances` is `None`: sorted instance indices
            Otherwise will pass through instances
    """
    if dim > 3:
        raise ValueError("Only supports bounding boxes up to three dimensions.")

    boxes = []
    _seg = seg.detach()

    if instances is None:
        instances = _seg.unique(sorted=True)
        instances = instances[instances > 0]

    for _idx in instances:
        instance_idx = (_seg == _idx).nonzero(as_tuple=False)

        _mins = instance_idx[:, -dim:].min(dim=0)[0]
        _maxs = instance_idx[:, -dim:].max(dim=0)[0]

        box = [
            _mins[-dim] - 1,
            _mins[(-dim) + 1] - 1,
            _maxs[-dim] + 1,
            _maxs[(-dim) + 1] + 1,
        ]
        if dim > 2:
            box = box + [_mins[(-dim) + 2] - 1, _maxs[(-dim) + 2] + 1]
        boxes.append(torch.tensor(box))

    if boxes:
        boxes = torch.stack(boxes)
    else:
        boxes = torch.tensor([[]])
    return boxes.to(dtype=torch.float, device=seg.device), instances


def instances_to_boxes_np(
    seg: np.ndarray,
    dim: int,
    instances: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert instance segmentation to bounding boxes (not batched)

    Args:
        seg: instance segmentation of individual classes [..., dims]
        dim: number of spatial dimensions to create bounding box for
            Conversion will always start from the last dimension. If `None`,
            all dimensions are used.

    Returns:
        np.ndarray: bounding boxes
            `(x1, y1, x2, y2, (z1, z2))` `List[Tensor[N, dim * 2]]`
        np.ndarray: if `instances` is `None`: sorted instance indices
            Otherwise will pass through instances
    """
    if dim > 3:
        raise ValueError("Only supports bounding boxes up to three dimensions.")

    boxes = []
    if instances is None:
        instances = np.unique(seg)
        instances = instances[instances > 0]

    for _idx in instances:
        instance_idx = np.stack(np.nonzero(seg == _idx), axis=1)
        _mins = np.min(instance_idx[:, -dim:], axis=0)
        _maxs = np.max(instance_idx[:, -dim:], axis=0)

        box = [
            _mins[-dim] - 1,
            _mins[(-dim) + 1] - 1,
            _maxs[-dim] + 1,
            _maxs[(-dim) + 1] + 1,
        ]
        if dim > 2:
            box = box + [_mins[(-dim) + 2] - 1, _maxs[(-dim) + 2] + 1]
        boxes.append(np.array(box))

    if boxes:
        boxes = np.stack(boxes)
    else:
        boxes = np.array([[]])
    return boxes, instances


def get_instance_class_from_properties(
    instance_idx: torch.Tensor,
    map_dict: Dict[Union[str, int], Union[str, int]],
) -> Tensor:
    """
    Extract instance classes form mapping dict

    Args:
        instance_idx: instance ids present in segmentaion
        map_dict: dict mapping instance ids (keys) to classes

    Returns:
        Tensor: extracted instance classes
    """
    _map_dict = {int(k): int(i) for k, i in map_dict.items()}
    classes = [int(_map_dict[int(idx.detach().item())]) for idx in instance_idx]
    return torch.tensor(classes, device=instance_idx.device)


def get_instance_class_from_properties_seq(
    instance_idx: Sequence[int],
    map_dict: Dict[Union[str, int], Union[str, int]],
) -> Sequence[int]:
    """
    Extract instance classes form mapping dict

    Args:
        instance_idx: instance ids present in segmentaion
        map_dict: dict mapping instance ids (keys) to classes

    Returns:
        Sequence[int]: extracted instance classes
    """
    _map_dict = {int(k): int(i) for k, i in map_dict.items()}
    classes = [int(_map_dict[int(idx)]) for idx in instance_idx]
    return classes


class Instances2Segmentation(AbstractTransform):
    def __init__(
        self,
        instance_key: str,
        map_key: str,
        seg_key: str = None,
        add_background: bool = True,
        grad: bool = False,
        present_instances: Optional[str] = None,
    ):
        """
        Convert instances to semantic segmentation

        Args:
            instance_key: key where instance segmentation is located
            map_key: key where mapping from instances to classes is located
            seg_key: key where segmentation should be saved; If None, the
                instance key will be overwritten
            add_background: adds +1 to classes from mapping for background
            grad: enable gradient propagation through transformation
            present_instances: key where precomputed present instances are
                saved. If None it will compute the present instance new.
        """
        super().__init__(grad=grad)
        self.add_background = add_background
        self.seg_key = seg_key if seg_key is not None else instance_key
        self.map_key = map_key
        self.instance_key = instance_key
        self.present_instances = present_instances

    def forward(self, **data) -> dict:
        """
        Convert instance segmentation to semantic segmentation

        Args:
            **data: batch dict

        Returns:
            dict: processed batch
        """
        semantic = torch.zeros_like(data[self.instance_key])
        for batch_idx in range(semantic.shape[0]):
            _present_instances = (
                data[self.present_instances][batch_idx]
                if self.present_instances is not None
                else None
            )
            instances_to_segmentation(
                data[self.instance_key][batch_idx],
                data[self.map_key][batch_idx],
                add_background=self.add_background,
                instance_idx=_present_instances,
                out=semantic[batch_idx],
            )
        data[self.seg_key] = semantic
        return data


def instances_to_segmentation(
    instances: Tensor,
    mapping: Dict[str, Union[str, int]],
    add_background: bool = True,
    instance_idx: Optional[Tensor] = None,
    out: Tensor = None,
) -> Tensor:
    """
    Convert instances to semantic segmentation

    Args:
        instances: instance segmentation; foreground classes > 0; [dims]
        mapping: mapping from each instance to class
        add_background: adds +1 to classes from mapping for background
            Should be enabled if classes in mapping start from zero and
            diabled otherwise
        out: optional output tensor where results are saved
        instance_idx: precomputed instance ids present in sample. If None
            the instances ids will be computed

    Returns:
        Tensor: semantic segmentation
    """
    mapping = {int(key): int(item) for key, item in mapping.items()}
    if out is None:
        out = torch.zeros_like(instances)

    if instance_idx is None:
        instance_idx = instances.unique(sorted=True)
        instance_idx = instance_idx[instance_idx > 0]

    for instance_id in instance_idx:
        _cls = mapping[instance_id.item()]
        if add_background:
            _cls += 1
        out[instances == instance_id] = _cls
    return out


def instances_to_segmentation_np(
    instances: np.ndarray,
    mapping: Dict[Union[str, int], Union[str, int]],
    add_background: bool = True,
    out: np.ndarray = None,
) -> np.ndarray:
    """
    Convert instances to semantic segmentation

    Args:
        instances: instance segmentation; foreground classes > 0; [dims]
        mapping: mapping from each instance to class
        add_background: adds +1 to classes from mapping for background
            Should be enabled if classes in mapping start from zero and
            diabled otherwise
        out: optional output tensor where results are saved

    Returns:
        Tensor: semantic segmentation
    """
    mapping = {int(key): int(item) for key, item in mapping.items()}
    if out is None:
        out = np.zeros_like(instances)

    instance_idx = np.unique(instances)
    instance_idx = instance_idx[instance_idx > 0]
    for instance_id in instance_idx:
        _cls = mapping[instance_id]
        if add_background:
            _cls += 1
        out[instances == instance_id] = _cls
    return out


class Instances2Fg(AbstractTransform):
    def __init__(
        self,
        instance_key: str,
        seg_key: str = None,
        grad: bool = False,
    ):
        """
        Convert instances to foreground segmentation

        Args:
            instance_key: key where instance segmentation is located
            seg_key: key where fg segmentation should be saved; If None, the
                instance key will be overwritten
            grad: enable gradient propagation through transformation
        """
        super().__init__(grad=grad)
        self.seg_key = seg_key if seg_key is not None else instance_key
        self.instance_key = instance_key

    def forward(self, **data) -> dict:
        """
        Convert instance segmentation to fg semantic segmentation

        Args:
            **data: batch dict

        Returns:
            dict: processed batch
        """
        semantic_fg = instances_to_fg(data[self.instance_key])
        data[self.seg_key] = semantic_fg
        return data


def instances_to_fg(
    instances: Tensor,
    out: Optional[Tensor] = None,
) -> Tensor:
    """
    Convert instances to foreground segmentation (0 = background)

    Args:
        instances: numbered array of instances, shape [dims]
        out: Optional output array. Defaults to None.

    Returns:
        Tensor: array of same shape as input with foregronud segmentation
    """
    if out is None:
        out = torch.zeros_like(instances)
    out = (instances > 0).to(instances)
    return out


def instances_to_fg_np(
    instances: np.ndarray,
    out: np.ndarray = None,
) -> np.ndarray:
    """
    Convert instances to foreground segmentation (0 = background)

    Args:
        instances: numbered array of instances, shape [dims]
        out: Optional output array. Defaults to None.

    Returns:
        np.ndarray: array of same shape as input with foregronud segmentation
    """
    if out is None:
        out = np.zeros_like(instances)
    out = (instances > 0).astype(instances.dtype)
    return out


def instances_to_binary_masks(
    instances: Tensor,
    instance_idx: Optional[Tensor] = None,
    out: Tensor = None,
) -> Tensor:
    """
    Convert numbered instances to binary masks. If no instances are present
    an empty array will be returend.

    Args:
        instances: numbered instances, shape [1, dims] (1 = channel dimension)
        instance_idx: Provide instance indices which are present.
            By providing the instances beforehand, running an additional
            unique operation will be avoided. Defaults to None.
        out: Optional output array. Defaults to None.

    Returns:
        Tensor: binary masks of instance, shape [R, dims] where R are the number
            of isntances. If no instances are present, an empty array is
            returned.
    """
    if instance_idx is None:
        instance_idx = instances.unique(sorted=True)
        instance_idx = instance_idx[instance_idx > 0]

    if instance_idx.numel() > 0:
        num_instances = instance_idx.shape[0]
    else:
        # no instances present return empty tensor
        num_instances = 0
        out = torch.tensor([]).to(instances)
        return out

    if out is None:
        out = torch.zeros(
            num_instances,
            *instances.shape[1:],
            dtype=instances.dtype,
            device=instances.device,
        )

    for _c in range(num_instances):
        out[_c] = instances == instance_idx[_c]
    return out


def instances_to_binary_masks_np(
    instances: np.ndarray,
    instance_idx: Optional[np.ndarray] = None,
    out: np.ndarray = None,
) -> np.ndarray:
    """
    Convert numbered instances to binary masks. If no instances are present
    an empty array will be returend.

    Args:
        instances: numbered instances, shape [1, dims] (1 = channel dimension)
        instance_idx: Provide instance indices which are present.
            By providing the instances beforehand, running an additional
            unique operation will be avoided. Defaults to None.
        out: Optional output array. Defaults to None.

    Returns:
        np.ndarray: binary masks of instance, shape [R, dims] where R are the
            number of isntances. If no instances are present, an empty array is
            returned.
    """
    if instance_idx is None:
        instance_idx = np.unique(instances)
        instance_idx = instance_idx[instance_idx > 0]

    if instance_idx.size > 0:
        num_instances = instance_idx.shape[0]
    else:
        # no instances present return empty tensor
        num_instances = 0
        out = np.array([]).astype(instances.dtype)
        return out

    if out is None:
        out = np.zeros(
            (num_instances, *instances.shape[1:]),
            dtype=instances.dtype,
        )

    for _c in range(num_instances):
        out[_c] = instances == instance_idx[_c]
    return out


class Instances2BinaryMasks(AbstractTransform):
    def __init__(
        self,
        instance_key: str,
        binary_mask_key: str,
        grad: bool = False,
        present_instances: Optional[str] = None,
        **kwargs,
    ):
        """
        Convert numbered instances to binary masks. If no instances are present
        an empty array will be inserted.

        Args:
            instance_key: Key where instances are saved
            binary_mask_key: Key where binary masks should be saved
            grad: Requries grad. Defaults to False.
            present_instances: Key where instance indices are present.
                By providing the instances beforehand, running an additional
                unique operation will be avoided. Defaults to None.
        """
        super().__init__(grad=grad, **kwargs)
        self.binary_mask_key = binary_mask_key
        self.instance_key = instance_key
        self.present_instances = present_instances

    def forward(self, **data) -> dict:
        """
        Convert numbered instances to binary masks

        Args:
            **data: batch dict

        Returns:
            dict: processed batch
        """
        data[self.binary_mask_key] = []
        for batch_idx, instance_element in enumerate(data[self.instance_key].split(1)):
            _present_instances = (
                data[self.present_instances][batch_idx]
                if self.present_instances is not None
                else None
            )
            data[self.binary_mask_key].append(
                instances_to_binary_masks(
                    instances=instance_element,
                    instance_idx=_present_instances,
                )
            )
        return data
