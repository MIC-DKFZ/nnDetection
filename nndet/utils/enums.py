# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


# IO
class SelectionMode(Enum):
    UNIFORM = "uniform"
    SQRT = "sqrt"


# Model Specific
class BoxRegressionMode(Enum):
    ENCODE = "encode"
    DECODE = "decode"
    DUAL = "dual"


# Postprocessing / Inference
class ModelNMS(Enum):
    NMS = "batched_nms"
    WNMS = "batched_weighted_nms"


class EnsembleNMS(Enum):
    NMS = "batched_nms"
    WBC = "batched_wbc"


class DimBoxMerger(Enum):
    GREEDYIOU = "GreedyIoUBoxMerger"
    VOTELABELGREEDYIOU = "VoteLabelGreedyIoUBoxMerger"


class PoolingMode(Enum):
    CONV_KERNEL = "conv_kernel"
    CONV_STRIDE = "conv_stride"
    MAX_KERNEL = "max_kernel"
    MAX_STRIDE = "max_stride"
    AVG_KERNEL = "avg_kernel"
    AVG_STRIDE = "avg_stride"
    BLOCK = "block"


class InterpolationMode(Enum):
    TRANSPOSE = "transpose"
    NEAREST = "nearest"
    LINEAR = "linear"
    CUBIC = "cubic"


class LoadModels(Enum):
    ALL = "all"
    BEST = "best"
    LAST = "last"


class AuxLossNorm(Enum):
    NONE = "none"
    MEAN = "mean"
    REDUCED = "reduced"


class BoxPointMode(Enum):
    CORNERS = "corners"
    CENTERS = "centers"


class FFNRegWeightInit(Enum):
    NONE = "none"  # default weight initialisation by linear layer
    ZERO = "zero"  # set weight and bias to zero
    ZERO_BIAS = "zero_bias"  # set bias to zero


class AnnotationStyle(Enum):
    WEAK = "weak"
    SEG = "seg"
