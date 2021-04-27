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

from nndet.planning.properties import (
    get_sizes_and_spacings_after_cropping,
    get_size_reduction_by_cropping,
    get_modalities,
    analyze_segmentations,
    analyze_intensities,
    analyze_instances,
)


def medical_segmentation_props(intensity_properties: bool = True):
    """
    Default set for analysis of medical segmentation images
    
    Args:
        intensity_properties (optional): analyze intensity properties. Defaults to True.
    
    Returns:
        Sequence[Callable]: properties to calculate. Results can be summarized as follows:
    
    See Also:
        :func:`nndet.planning.medical.get_sizes_and_spacings_after_cropping`,
        :func:`nndet.planning.medical.get_size_reduction_by_cropping`,
        :func:`nndet.planning.intensity.get_modalities`,
        :func:`nndet.planning.intensity.analyze_intensities`,
        :func:`nndet.planning.segmentation.analyze_segmentations`,
    """
    props = [
        get_sizes_and_spacings_after_cropping,
        get_size_reduction_by_cropping,
        get_modalities,
        analyze_segmentations,
    ]

    if intensity_properties:
        props.append(analyze_intensities)
    else:
        props.append(lambda x: {'intensity_properties': None})
    return props


def medical_instance_props(intensity_properties: bool = True):
    """
    Default set for analysis of medical instance segmentation images

    Args:
        intensity_properties (optional): analyze intensity properties. Defaults to True.

    Returns:
        Sequence[Callable]: properties to calculate. Results can be summarized as follows:

    See Also:
        :func:`nndet.planning.medical.get_sizes_and_spacings_after_cropping`,
        :func:`nndet.planning.medical.get_size_reduction_by_cropping`,
        :func:`nndet.planning.intensity.get_modalities`,
        :func:`nndet.planning.intensity.analyze_intensities`,
        :func:`nndet.planning.instance.analyze_instances`,
    """
    props = [
        get_sizes_and_spacings_after_cropping,
        get_size_reduction_by_cropping,
        get_modalities,
        analyze_instances,
    ]

    if intensity_properties:
        props.append(analyze_intensities)
    else:
        props.append(lambda x: {'intensity_properties': None})
    return props
