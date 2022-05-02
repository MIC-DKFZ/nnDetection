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
