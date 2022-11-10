from typing import Optional

import torch


class FFNClassifier(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        internal_channels: int,
        num_classes: int,
        num_layers: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()

        self.fc = self._build_module(**kwargs)

        self.in_channels = in_channels
        self.internal_channels = internal_channels
        self.num_classes = num_classes
        self.num_layers = num_layers

        self.loss: Optional[torch.nn.Module] = None
        self.logits_convert_fn: Optional[torch.nn.Module] = None
        self.init_weights()

    def _build_module(self, **kwargs):
        modules = []
        for idx in range(self.num_layers):
            in_features = self.in_channels if idx == 0 else self.internal_channels
            out_features = self.num_classes if idx == self.num_layers - 1 else self.internal_channels

            modules.append(
                torch.nn.Linear(
                    in_features=in_features,
                    out_features=out_features,
                )
            )

        if len(modules) == 1:
            return modules[0]
        else:
            return torch.nn.Sequential(*modules)

    def init_weights(self):
        pass

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)

    def compute_loss(
        self,
        pred_logits: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Base classifier with cross entropy loss (in general hard negative
        example mining should be done before this)

        Args:
            pred_logits (Tensor): predicted logits
            targets (Tensor): classification targets

        Returns:
            Tensor: classification loss
        """
        return self.loss(pred_logits, targets, **kwargs)

    def logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to probabilities

        Args:
            logits (Tensor): bounding box logits [N, C]
                N = number of anchors, C=number of foreground classes

        Returns:
            Tensor: probabilities
        """
        return self.logits_convert_fn(logits)

    def postprocess_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to probabilities and remove potential background class

        Args:
            logits: predicted logits #TODO: shape

        Returns:
            torch.Tensor: converted logits #TODO:shape
        """
        raise NotImplementedError


# TODO: classifier CE
# BCE classifier
# BCE + Focal Init
# Focal Loss


# batch_pred_scores_fg = F.softmax(pred_detection["pred_logits"], dim=-1)[
