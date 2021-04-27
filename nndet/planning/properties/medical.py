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

from typing import Dict, List
from collections import defaultdict, OrderedDict

from nndet.io.load import load_properties_of_cropped
from nndet.planning.analyzer import DatasetAnalyzer


def get_sizes_and_spacings_after_cropping(analyzer: DatasetAnalyzer) -> Dict[str, List]:
    """
    Load all sizes and spacings after cropping
    
    Args:
        analyzer: analyzer which calls this property
    
    Returns:
        Dict[str, List]: loaded sizes and spacings inside list
            `all_sizes`: contains all sizes
            `all_spacings`: contains all spacings
    """
    output = defaultdict(list)
    for case_id in analyzer.case_ids:
        properties = load_properties_of_cropped(analyzer.cropped_data_dir / case_id)
        output['all_sizes'].append(properties["size_after_cropping"])
        output['all_spacings'].append(properties["original_spacing"])
    return output


def get_size_reduction_by_cropping(analyzer: DatasetAnalyzer) -> Dict[str, Dict]:
    """
    Compute all size reductions of each case
    
    Args:
        analyzer: analzer which calls this property
    
    Returns:
        Dict: computed size reductions
            `size_reductions`: dictionary with each case id and reduction
    """
    size_reduction = OrderedDict()
    for case_id in analyzer.case_ids:
        props = load_properties_of_cropped(analyzer.cropped_data_dir / case_id)
        shape_before_crop = props["original_size_of_raw_data"]
        shape_after_crop = props['size_after_cropping']
        size_red = np.prod(shape_after_crop) / np.prod(shape_before_crop)
        size_reduction[case_id] = size_red
    return {"size_reductions": size_reduction}
