from enum import Enum


# IO
class SelectionMode(Enum):
    UNIFORM = "uniform"
    SQRT = "sqrt"


# Model Specific
class BoxRegressionMode(Enum):
    ENCODE = "encode"
    DECODE = "decode"


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


class InterpolationMode(Enum):
    TRANSPOSE = "transpose"
    NEAREST = "nearest"
    LINEAR = "linear"
    CUBIC = "cubic"
