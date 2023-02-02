from torch import nn

# Define two Linear Classifiers for CE Loss or Focal Loss, simply to initialize with the same parameter


class LinearClassifierCE(nn.Linear):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__(in_features=in_features, out_features=num_classes + 1)


class LinearClassifierFocalLoss(nn.Linear):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__(in_features=in_features, out_features=num_classes)
