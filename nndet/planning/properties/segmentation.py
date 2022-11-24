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
from itertools import repeat
from collections import OrderedDict
from skimage.morphology import label
from multiprocessing import Pool
from typing import Dict, List, Sequence, Tuple, Callable

from nndet.planning.analyzer import DatasetAnalyzer
from nndet.io.load import load_case_cropped


def analyze_segmentations(analyzer: DatasetAnalyzer) -> dict:
    """
    Analyze segmentation of dataset (if overwrite is disabled and analysis was already run, 
    this function will only load the results)
    
    Args:
        analyzer: analyzer which calls this function
    
    Returns:
        Dict:
            `class_dct`(np.ndarray): contains all present classes
            `all_classes`(np.ndarray): values of all foreground classes
            `segmentation_props_per_patient`: result from :func:`run_analyze_segmentation`
    """
    class_dct = analyzer.data_info["labels"]
    all_classes = np.array([int(i) for i in class_dct.keys()])

    if analyzer.overwrite or not analyzer.props_per_case_file.is_file():
        props_per_case = run_analyze_segmentation(analyzer, all_classes)
    else:
        with open(analyzer.props_per_case_file, "rb") as f:
            props_per_case = pickle.load(f)
    return {'class_dct': class_dct, 'all_classes': all_classes,
            'segmentation_props_per_patient': props_per_case}


def analyze_segmentation_per_case(analyzer: DatasetAnalyzer, case_id: str,
                                  all_classes: Sequence[int]) -> Dict:
    """
    1) what class is in this training case?
    2) what is the size distribution for each class?
    3) what is the region size of each class?
    4) check if all in one region

    Args:
        analyzer: calling analyzer
        case_id: case identifier
        all_classes: all present classes in dataset

    Returns:
        Dict: region and class properties of case
            `has_classes` (np.ndarray): present classes in case
            `only_one_region` (Dict[Tuple[int], bool]):
                contains information if individual classes are only present as a single region;
                analyses if all classes build a single region;
                can be indexed by the respective tuple of classes
            `volume_per_class` ([Dict]): physical colume per class
            `region_volume_per_class` (Dict[List]): physical volume per class per region
    """
    logger.info(f"Processing segmentation properties of case {case_id}")
    _, seg, props = load_case_cropped(analyzer.cropped_data_dir, case_id)
    vol_per_voxel = np.prod(props['itk_spacing'])

    unique_classes = np.unique(seg)

    regions = [list(all_classes)]
    for c in all_classes:
        regions.append((c, ))
    all_in_one_region = check_if_all_in_one_region(seg, regions)

    volume_per_class, region_sizes = collect_class_and_region_sizes(
        seg, all_classes, vol_per_voxel)

    return {"has_classes": unique_classes, "only_one_region": all_in_one_region,
            "volume_per_class": volume_per_class, "region_volume_per_class": region_sizes}


def run_analyze_segmentation(
    analyzer: DatasetAnalyzer, all_classes: Sequence[int],
    save: bool = True, 
    analyze_fn: Callable[[DatasetAnalyzer, str, Sequence[int]], Dict] = analyze_segmentation_per_case) \
        -> Dict[str, Dict]:
    """
    Analyze segmentations of all cases in analyzer
    
    Args:
        analyzer: analyzer which called this function
        all_classes: values of all classes
        save: Saves results as a file. Defaults to True.
            (name is specified by `props_per_case_file` from analyzer)
        analyze_fn: callable
            to compute needed properties of a single segmentation case. Takes
            the calling analyzer, the case id and a sequence of integers representing
            all classes in the dataset and should return a single dict
    
    Returns:
        Dict[Dict]: computed properties per case
    """
    props_per_case = OrderedDict()

    if analyzer.num_processes == 0:
        props = [analyze_fn(*args) for args in zip(
            repeat(analyzer), analyzer.case_ids, repeat(all_classes))]
    else:
        with Pool(analyzer.num_processes) as p:        
                props = p.starmap(analyze_fn, zip(
                    repeat(analyzer), analyzer.case_ids, repeat(all_classes)))

    for case_id, prop in zip(analyzer.case_ids, props):
        props_per_case[case_id] = prop

    if save:
        with open(analyzer.props_per_case_file, "wb") as f:
            pickle.dump(props_per_case, f)
    return props_per_case


def check_if_all_in_one_region(seg: np.ndarray,
                               regions: Sequence[Sequence[int]]) -> Dict[Tuple[int], bool]:
    """
    Check if regions are splited over multiple instances or are all connected
    
    Args:
        seg: segmentation
        regions: Sequence of multiple regions to analyze.
            Each region can contain multiple classes
    
    Returns:
        Dict[Tuple[int], bool]: result for each region
    """
    res = OrderedDict()
    for r in regions:
        new_seg = np.zeros(seg.shape)
        for c in r:
            new_seg[seg == c] = 1
        labelmap, numlabels = label(new_seg, return_num=True)
        if numlabels != 1:
            res[tuple(r)] = False
        else:
            res[tuple(r)] = True
    return res


def collect_class_and_region_sizes(seg: np.ndarray, all_classes: Sequence[int],
                                   vol_per_voxel: float) -> (Dict, Dict[str, Dict]):
    """
    Collect class and region sizes from segmentation
    
    Args:
        seg: segmentation
        all_classes: array with all classes
        vol_per_voxel: physical volume per voxel
    
    Returns:
        Dict: volume per class (dict index corresponds to class)
        Dict[List]: sizes of each region; 
            first dict indexes thes class while second dict indexed the regions
    """
    volume_per_class = OrderedDict()
    region_volume_per_class = OrderedDict()
    for c in all_classes:
        volume_per_class[c] = np.sum(seg == c) * vol_per_voxel
        
        region_volume_per_class[c] = []
        labelmap, numregions = label(seg == c, return_num=True)
        for l in range(1, numregions + 1):
            region_volume_per_class[c].append(np.sum(labelmap == l) * vol_per_voxel)
    return volume_per_class, region_volume_per_class
