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

import pickle
import numpy as np

from loguru import logger
from collections import OrderedDict, defaultdict
from multiprocessing.pool import Pool
from itertools import repeat
from typing import Dict, Sequence, List, Tuple


from nndet.io.load import load_case_cropped
from nndet.planning import DatasetAnalyzer
from nndet.core.boxes import box_iou_np


def analyze_instances(analyzer: DatasetAnalyzer) -> dict:
    """
    Analyze instance segmentations
    
    Args:
        analyzer (DatasetAnalyzer): calling analyzer
    
    Returns:
        dict: extracted properties
    """
    class_dct = analyzer.data_info["labels"]
    all_classes = np.array([int(i) for i in class_dct.keys()])

    if analyzer.overwrite or not analyzer.props_per_case_file.is_file():
        props_per_case = run_analyze_instances(analyzer, all_classes)
    else:
        with open(analyzer.props_per_case_file, "rb") as f:
            props_per_case = pickle.load(f)

    output = {'class_dct': class_dct,
              'all_classes': all_classes,
              'instance_props_per_patient': props_per_case
              }
    output.update(analyze_instances_data_set(props_per_case))
    return output


def run_analyze_instances(analyzer: DatasetAnalyzer,
                          all_classes: Sequence[int],
                          save: bool = True,
                          ):
    """
    Analyze all instance segmentation from data set

    Args:
        analyzer: calling analyzer
        all_classes: all classes present in dataset
        save: save properties per case as pickle
            file :param:`analyzer.props_per_case_file`

    Returns:
        Dict: extract properties per case id [case_id, property_dict]
    """
    props_per_case = OrderedDict()
    if analyzer.num_processes == 0:
        props = [analyze_instances_per_case(*args) for args in zip(
                 repeat(analyzer), analyzer.case_ids, repeat(all_classes))]
    else:
        with Pool(analyzer.num_processes) as p:
            props = p.starmap(analyze_instances_per_case, zip(
                repeat(analyzer), analyzer.case_ids, repeat(all_classes)))

    # props = [analyze_instances_per_case(analyzer, cid, all_classes) for cid in analyzer.case_ids]

    for case_id, prop in zip(analyzer.case_ids, props):
        props_per_case[case_id] = prop

    if save:
        with open(analyzer.props_per_case_file, "wb") as f:
            pickle.dump(props_per_case, f)
    return props_per_case


def analyze_instances_data_set(props_per_case: OrderedDict) -> dict:
    """
    Compute properties of instances over whole dataset

    Args:
        props_per_case: properties per case
            `num_instances`: see :func:`count_instances`
            `class_ious`: see :func:`instance_class_and_region_sizes`
            `all_ious`: see :func:`instance_class_and_region_sizes`

    Returns:
        Dict: properties extracted from whole dataset
            `num_instances`(Dict[int, int]): number of instances per class
            `class_ious`(Dict[int, np.ndarray]): all flattened IoUs of
                instances of the same class
            `all_ious`(Dict[int, np.ndarray]): all flattened IoUs of
                instances regardless their class
    """
    data_props = {}
    num_instances = defaultdict(int)
    class_ious = defaultdict(list)
    for case_id, case_props in props_per_case.items():
        for cls, count in case_props["num_instances"].items():
            num_instances[cls] += count

        for cls, ious in case_props["class_ious"].items():
            class_ious[cls].append(ious.flatten())
    data_props["num_instances"] = num_instances
    for cls in class_ious.keys():
        class_ious[cls] = np.concatenate(class_ious[cls])
    data_props["class_ious"] = class_ious
    data_props["all_ious"] = np.concatenate([case_props["all_ious"].flatten()
                                             for _, case_props in props_per_case.items()])
    return data_props


def analyze_instances_per_case(analyzer: DatasetAnalyzer,
                               case_id: str,
                               all_classes: Sequence[int],
                               ):
    """
    Analyze a single case

    Args:
        analyzer: calling analyzer
        case_id: case identifier
        all_classes: all classes present in dataset

    Returns:
        Dict[str, Any]: properties extracted per case. See:
            `num_instanes` (Dict[int, int]): number of instance per class
            `has_classes` (Sequence[int]): classes present in this case
            `volume_per_class(Dict[int, float])`: volume per class (sum of
                all instance volume corresponding to class)
                [all_classes, volume]
            `region_volume_per_class`(Dict[int, List[float]]): volume of
                each instance (sorted to corresponding class)
                [all_classes, list(region_class_volume)]
            `boxes`(np.ndarray): bounding boxes (x1, y1, x2, y2, (z1, z2))[N, dims * 2]
            `all_ious`(np.ndarray): IoU values between all boxes independent of class
            `class_ious`(Dict[int, np.ndarray]): IoU values of boxes with respect to classes
    """
    logger.info(f"Processing instance properties of case {case_id}")
    _, iseg, props = load_case_cropped(analyzer.cropped_data_dir, case_id)
    props["num_instances"] = count_instances(props, all_classes)
    props["has_classes"] = list(set(props["instances"].values()))
    props["volume_per_class"], props["region_volume_per_class"] = \
        instance_class_and_region_sizes(case_id, iseg, props, all_classes)
    props["boxes"] = iseg_to_boxes(iseg)
    props["all_ious"], props["class_ious"] = case_ious(props["boxes"], props)
    return props


def count_instances(props: dict, all_classes: Sequence[int]) -> Dict[int, int]:
    """
    Count instace classes inside one case

    Args:
        props: additional properties
            `instances` (Dict[int, int]): maps each instance to a numerical class
        all_classes: all classes in dataset

    Returns:
        Dict[int, int]: number of instance per class [all_classes, count]
    """
    instance_classes = list(map(int, props["instances"].values()))
    return {int(c): instance_classes.count(int(c)) for c in all_classes}


def instance_class_and_region_sizes(
        case_id: str,
        iseg: np.ndarray,
        props: dict,
        all_classes: Sequence[int],
        ) -> Tuple[
        Dict[int, float], Dict[int, List[float]]]:
    """
    Compute physical volume of all instances
    Classes which are not present in case are 0 or an empty list.

    Args:
        iseg: instance segmentation
        props: additional properties
            `itk_spacing` (Sequence[float]): spacing information
            `instances` (Dict[int, int]): maps each instance to a numerical class
        all_classes: all classes in dataset

    Returns:
        Dict[int, float]: volume per class (sum of all instance volume
            corresponding to class) [all_classes, volume]
        Dict[int, List[float]]: volume of each instance (sorted to
            corresponding class) [all_classes, list(region_class_volume)]
    """
    vol_per_voxel = np.prod(props['itk_spacing'])
    instance_classes = {int(key): int(item) for key, item in props["instances"].items()}

    volume_per_class = OrderedDict(zip(all_classes, [0] * len(all_classes)))
    region_volume_per_class = OrderedDict()

    ids = np.unique(iseg)
    ids = ids[ids > 0]
    if len(ids) != len(list(instance_classes.keys())):
        logger.warning(f"Instance lost. Found {instance_classes} in "
                       f"properties but {ids} in seg of {case_id}.")
    volumer_per_instance = {c: np.sum(iseg == c) * vol_per_voxel for c in ids}

    for instance_id, instance_vol in volumer_per_instance.items():
        i_cls = instance_classes[instance_id]
        volume_per_class[i_cls] += instance_vol
        if i_cls in region_volume_per_class:
            region_volume_per_class[i_cls].append(instance_vol)
        else:
            region_volume_per_class[i_cls] = [instance_vol]
    return volume_per_class, region_volume_per_class


def iseg_to_boxes(iseg: np.ndarray) -> np.ndarray:
    """
    Convert instance segmentations to bounding boxes

    Args:
        iseg: instance segmentation [dims] (NO channel dim)

    Returns:
        (np.ndarray): bounding boxes (x1, y1, x2, y2, (z1, z2))[N, dims * 2]
            (order of boxes corresponds to instance ids)
    
    Notes:
        Please refer to `nndet.io.transforms.instances` for the function
        and don't use this one.
    """
    boxes = []
    ids = np.unique(iseg)
    ids = ids[ids > 0]
    for instance_id in ids:
        instance_idx = np.argwhere(iseg == instance_id)
        coord_list = [np.min(instance_idx[:, 0]) - 1,
                      np.min(instance_idx[:, 1]) - 1,
                      np.max(instance_idx[:, 0]) + 1,
                      np.max(instance_idx[:, 1]) + 1]
        if instance_idx.shape[1] == 3:
            coord_list.extend([np.min(instance_idx[:, 2]) - 1,
                               np.max(instance_idx[:, 2]) + 1])
        boxes.append(coord_list)

    if boxes:
        return np.stack(boxes)
    else:
        return []


def case_ious(boxes: np.ndarray, props: dict) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    Compute IoU values for a single case (Evaluated both settings: all
    bounding boxes and bounding boxes corresponding to a specific class)

    Args:
        boxes: bounding boxes of a single case (x1, y1, x2, y2, (z1, z2))[N, dims*2]
        props: additional properties
            `instances` (Dict[int, int]): maps each instance to a numerical class

    Returns:
        np.ndarray: IoU values of all bounding boxes
        Dict[int, np.ndarray]: IoU values of bounding boxes which correspond
            to a specific class
    """
    if not isinstance(boxes, list):
        all_ious = compute_each_iou(boxes)

        class_ious = OrderedDict()
        case_classes = list(set(map(int, props["instances"].values())))
        # boxes are sorted by instance id
        case_instances = sorted(list(map(int, props["instances"].keys())))

        for _c in case_classes:
            cls_box_indices = [int(props["instances"][str(ci)]) == _c for ci in case_instances]
            class_ious[_c] = compute_each_iou(boxes[cls_box_indices])
    else:
        all_ious = np.array([])
        class_ious = {}
    return all_ious, class_ious


def compute_each_iou(boxes: np.ndarray):
    """
    Compute IoU values from each box to each box (except the same box)

    Args:
        boxes: bounding boxes (x1, y1, x2, y2, (z1, z2))[N, dims*2]

    Returns:
        np.ndarray: computed IoUs [N-1, N-1]
    """
    ious = box_iou_np(boxes, boxes)
    # remove diagonal elements because they are always one
    ious = ious[~np.eye(ious.shape[0], dtype=bool)].reshape(ious.shape[0], -1)
    return ious
