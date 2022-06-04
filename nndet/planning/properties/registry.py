# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from nndet.planning.properties import (
    analyze_instances,
    analyze_intensities,
    analyze_segmentations,
    get_modalities,
    get_size_reduction_by_cropping,
    get_sizes_and_spacings_after_cropping,
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
        props.append(lambda x: {"intensity_properties": None})
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
        props.append(lambda x: {"intensity_properties": None})
    return props
