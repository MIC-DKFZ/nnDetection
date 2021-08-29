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

import numpy as np

from scipy.ndimage import label
from typing import Dict, Sequence, Union, Tuple, Optional

from nndet.io.transforms.instances import get_bbox_np


def seg_to_instances(
    seg: np.ndarray,
    min_num_voxel: int = 0,
    ) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Use connected components with ones matrix to created
    instances from segmentation

    Args:
        seg: semantic segmentation [spatial dims]
        min_num_voxel: minimum number of voxels of an instance

    Returns:
        np.ndarray: instance segmentation
        Dict[int, int]: mapping from instances to classes
    """
    structure = np.ones([3] * seg.ndim)
    
    unique_classes = np.unique(seg)
    unique_classes = unique_classes[unique_classes > 0]

    instances = np.zeros_like(seg)
    instance_classes = {}

    i = 1
    for uc in unique_classes:
        binary_class_mask = (seg == uc)
        instances_temp, _ = label(binary_class_mask, structure=structure)

        instance_ids = np.unique(instances_temp)
        instance_ids = instance_ids[instance_ids > 0]

        for iid in instance_ids:
            instance_binary_mask = instances_temp == iid
            
            if min_num_voxel > 0:
                if instance_binary_mask.sum() < min_num_voxel:  # remove small instances
                    continue

            instances[instance_binary_mask] = i  # save instance to final mask
            instance_classes[int(i)] = uc
            i = i + 1  # bump instance index
    return instances, instance_classes


def seg_to_instances_voted(
    seg: np.ndarray,
    min_num_voxel: int = 0,
    ) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Conntected component analysis is performed on foreground
    (independent of exact class) and the final class
    is determined via majority voting.

    Args:
        seg: semantic segmentation [spatial dims]
        min_num_voxel: minimum number of voxels of an instance

    Returns:
        np.ndarray: instance segmentation
        Dict[int, int]: mapping from instances to classes
    """
    structure = np.ones([3] * seg.ndim)

    binary_fg_mask = (seg > 0).astype(int)
    instances_temp, _ = label(binary_fg_mask, structure=structure)

    instance_ids = np.unique(instances_temp)
    instance_ids = instance_ids[instance_ids > 0]

    instances = np.zeros_like(seg)
    instance_classes = {}

    i = 1
    for iid in instance_ids:
        instance_binary_mask = instances_temp == iid
        
        if min_num_voxel > 0:
            if instance_binary_mask.sum() < min_num_voxel:  # remove small instances
                continue

        instances[instance_binary_mask] = i  # save instance to final mask
        cls_id, cls_count = np.unique(
            seg[instance_binary_mask], return_counts=True) # count classes in region
        majority_voted_class = cls_id[np.argmax(cls_count)] # select class with most votes

        assert 0 not in cls_id
        assert majority_voted_class > 0

        instance_classes[int(i)] = majority_voted_class
        i = i + 1  # bump instance index
    return instances, instance_classes


def remove_classes(
    seg: np.ndarray,
    rm_classes: Sequence[int],
    classes: Dict[int, int] = None,
    background: int = 0,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[int, int]]]:
    """
    Remove classes from segmentation (also works on instances
    but instance ids may not be consecutive anymore)

    Args:
        seg: segmentation [spatial dims]
        rm_classes: classes which should be removed
        classes: optional mapping from instances from segmentation to classes
        background: background value

    Returns:
        np.ndarray: segmentation where classes are removed
        Dict[int, int]: updated mapping from instances to classes
    """
    for rc in rm_classes:
        seg[seg == rc] = background
        if classes is not None:
            classes.pop(rc)
    if classes is None:
        return seg
    else:
        return seg, classes


def reorder_classes(
    seg: np.ndarray,
    class_mapping: Dict[int, int],
    ) -> np.ndarray:
    """
    Reorders classes in segmentation

    Args:
        seg: segmentation
        class_mapping: mapping from source id to new id

    Returns:
        np.ndarray: remapped segmentation
    """
    for source_id, target_id in class_mapping.items():
        seg[seg == source_id] = target_id
    return seg


def compute_score_from_seg(
    instances: np.ndarray,
    instance_classes: Dict[int, int],
    probs: np.ndarray,
    aggregation: str = "max",
    ) -> np.ndarray:
    """
    Combine scores for each instance given an instance mask and instance logits

    Args:
        instances: instance segmentation [dims]; dims can be arbitrary dimensions
        instance_classes: assign each instance id to a class (id -> class)
        probs: predicted probabilities for each class [C, dims];
            C = number of classes, dims need to have the same dimensions as
            instances
        aggregation: defines the aggregation method for the probabilities.
            One of 'max', 'mean'

    Returns:
        Sequence[float]: Probability for each instance
    """
    instance_classes = {int(key): int(item) for key, item in instance_classes.items()}
    instance_ids = list(instance_classes.keys())
    instance_scores = []
    for iid in instance_ids:
        ic = instance_classes[iid]
        instance_mask = instances == iid
        instance_probs = probs[ic][instance_mask]

        if aggregation == "max":
            _score = np.max(instance_probs)
        elif aggregation == "mean":
            _score = np.mean(instance_probs)
        elif aggregation == "median":
            _score = np.median(instance_probs)
        elif aggregation == "percentile95":
            _score = np.percentile(instance_probs, 95)
        else:
            raise ValueError(f"Aggregation {aggregation} is not aggregation")
        instance_scores.append(_score)
    return np.asarray(instance_scores)


def softmax_to_instances(
    probs: np.ndarray,
    aggregation: str,
    stuff: Optional[Sequence[int]] = None,
    min_num_voxel: int = 0,
    min_threshold: Optional[float] = None,
    ) -> dict:
    """
    Compute instance segmentation results from a semantic segmentation
    argmax -> remove stuff classes -> connected components ->
    aggregate score inside each instance

    Args:
        probs: Predicted probabilities for each class [C, dims];
            C = number of classes, dims can be arbitrary dimensions
        aggregation: defines the aggregation method for the probabilities.
            One of 'max', 'mean'
        stuff: stuff classes to be ignored during conversion.
        min_num_voxel: minimum number of voxels of an instance
        min_threshold: if None argmax is used. If a threshold is provided
            it is used as a probability threshold for the foreground class.
            if multiple foreground classes exceed the threshold, the
            foreground class with the largest probability is selected.

    Returns:
        dict: predictions
            `pred_instances`: instance segmentation [dims]
            `pred_boxes`: predicted bounding boxes [2 * spatial dims]
            `pred_labels`: predicted class for each instance/box
            `pred_scores`: predicted score for each instance/box
    """
    if probs.shape[0] < 2:
        raise ValueError("softmax_to_instances only works for softmax probabilities")

    if min_threshold is not None:
        if probs.shape[0] > 2:
            cluster_map = np.max(probs[1:], axis=0) > min_threshold
            class_map = np.argmax(probs[1:], axis=0) + 1

            seg = np.zeros_like(probs[0])
            seg[cluster_map] = class_map[cluster_map]
        else:
            seg = probs[1] > min_threshold
    else:
        seg = np.argmax(probs, axis=0)

    if stuff is not None:
        for s in stuff:
            seg[seg == s] = 0

    instances, instance_classes = seg_to_instances_voted(seg, min_num_voxel=min_num_voxel)

    instance_scores = compute_score_from_seg(
        instances, instance_classes, probs, aggregation=aggregation,
        )
    instance_classes = {int(key): int(item) - 1 for key, item in instance_classes.items()}
    tmp = get_bbox_np(instances[None], instance_classes)
    instance_boxes = tmp["boxes"]
    instance_classes_seq = tmp["classes"]

    return {
        "pred_instances": instances,
        "pred_boxes": instance_boxes,
        "pred_labels": instance_classes_seq,
        "pred_scores": instance_scores,
        }
