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
from multiprocessing import Pool
from collections import OrderedDict
from typing import Union, Sequence, Dict

from nndet.planning.analyzer import DatasetAnalyzer
from nndet.io.load import load_case_cropped


def get_modalities(analyzer: DatasetAnalyzer) -> dict:
    """
    Extract modalities from analyzer data info
    
    Args:
        analyzer: calling analyzer; need to provide `modalities` dict in :param:`data_info`
    
    Returns:
        dict: extract modalities
            `modalities` (Dict[int, str]): modalities 
    """
    modalities = analyzer.data_info["modalities"]
    modalities = {int(k): modalities[k] for k in modalities.keys()}
    return {"modalities": modalities}


def analyze_intensities(analyzer: DatasetAnalyzer) -> dict:
    """
    Either recompute or load intensity statistics from dataset
    
    Args:
        analyzer: calling analyer; need to provide a dictionary where 
            modalities are named in :param:`data_info` in key `modalities`

    Returns:
        Dict: 
            `intensity_properties`: result of :func:`run_collect_intensity_properties`
    """
    num_modalities = len(analyzer.data_info["modalities"].keys())
    
    if analyzer.overwrite or not analyzer.intensity_properties_file.is_file():
        results = run_collect_intensity_properties(analyzer, num_modalities)
    else:
        with open(analyzer.intensity_properties_file, 'rb') as f:
            results = pickle.load(f)
    return {'intensity_properties': results}


def run_collect_intensity_properties(analyzer: DatasetAnalyzer,
                                     num_modalities: int, save: bool = True) -> Dict[int, Dict]:
    """
    Collect intensity properties over forground from whole dataset
    
    Args:
        analyzer: calling analyzer
        num_modalities: number of modalities
        save (optional): Save result in `analyzer.intensity_properties_file`. Defaults to True.
    
    Returns:
        Dict[int, Dict]: Intensity properties of foreground over the dataset.
            Evaluated statistics: `median`; `mean`; `std`; `min`; `max`; `percentile_99_5`; `percentile_00_5`
            `local_props`: contains a dict (with case ids) where statistics where computed per case
    """
    
    results = OrderedDict()
    for mod_id in range(num_modalities):
        logger.info(f"Processing intensity values of modality {mod_id}")
        results[mod_id] = OrderedDict()

        if analyzer.num_processes == 0:
            voxels = [get_voxels_in_foreground(*args) for args in
                      zip(repeat(analyzer), analyzer.case_ids, repeat(mod_id))]
            local_props = [compute_stats(v) for v in voxels]
        else:
            with Pool(analyzer.num_processes) as p:
                voxels = p.starmap(get_voxels_in_foreground,
                                    zip(repeat(analyzer), analyzer.case_ids, repeat(mod_id)))
                local_props = p.map(compute_stats, voxels)

        props_per_case = OrderedDict()
        for case_id, lp in zip(analyzer.case_ids, local_props):
            props_per_case[case_id] = lp

        all_voxels = []
        for iv in voxels:
            all_voxels += iv
        results[mod_id]['local_props'] = props_per_case
        results[mod_id].update(compute_stats(all_voxels))

    if save:
        with open(analyzer.intensity_properties_file, 'wb') as f:
            pickle.dump(results, f)
    return results


def get_voxels_in_foreground(analyzer: DatasetAnalyzer, case_id: str,
                             modality_id: int, subsample: int = 10) -> list:
    """
    Get voxels from foreground
    
    Args:
        analyzer: calling analyzer
        case_id: case identifier
        modality_id: modality to choose for analyses
        subsample (optional): Subsample voxels for computational purposes. Defaults to 10.
    
    Returns:
        list: foreground voxels
    """
    data, seg, props = load_case_cropped(analyzer.cropped_data_dir, case_id)
    modality = data[modality_id]
    mask = seg > 0
    voxels = list(modality[mask.astype(bool)][::subsample])  # no need to take every voxel
    return voxels


def compute_stats(voxels: Union[Sequence, np.ndarray]):
    """
    Compute statistics of voxels
    
    Args:
        voxels: input voxels
    
    Returns:
        Dict[str, np.ndarray]: computed statistics
            `median`; `mean`; `std`; `min`; `max`; `percentile_99_5`; `percentile_00_5`
    """
    if len(voxels) == 0:
        stats = {"median": np.nan, "mean": np.nan, "std": np.nan, "min": np.nan,
                 "max": np.nan, "percentile_99_5": np.nan, "percentile_00_5": np.nan,
                }
    else:
        stats = {
            "median": np.median(voxels),
            "mean": np.mean(voxels),
            "std": np.std(voxels),
            "min": np.min(voxels),
            "max": np.max(voxels),
            "percentile_99_5": np.percentile(voxels, 99.5),
            "percentile_00_5": np.percentile(voxels, 00.5),
        }
    return stats

