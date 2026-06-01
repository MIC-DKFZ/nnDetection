# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict, defaultdict
from typing import Dict, List

import numpy as np

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
        output["all_sizes"].append(properties["size_after_cropping"])
        output["all_spacings"].append(properties["original_spacing"])
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
        shape_after_crop = props["size_after_cropping"]
        size_red = np.prod(shape_after_crop) / np.prod(shape_before_crop)
        size_reduction[case_id] = size_red
    return {"size_reductions": size_reduction}
