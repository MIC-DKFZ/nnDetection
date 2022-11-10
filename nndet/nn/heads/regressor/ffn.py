from typing import Dict, Optional

import torch


class FFNRegressor(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        internal_channels: int,
        dim: int,
        num_layers: int = 1,
        # TODO: fc wrapper
        # TODO: do_norm & do_act
        **kwargs,
    ) -> None:
        super().__init__()

        self.mlp = self._build_module(**kwargs)

        self.in_channels = in_channels
        self.internal_channels = internal_channels
        self.dim = dim
        self.num_layers = num_layers

        self.loss_name: str = "ffn_reg_spec"
        self.box_loss_name: str = "ffn_reg_box"
        self.loss: Optional[torch.nn.Module] = None
        self.box_loss: Optional[torch.nn.Module] = None
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
        """
        Forward feature through module

        Args:
            features: input feature [D, B, R, C] where D=number of decoder
                layers, B=batch size, R=number of predictions, C=number of
                channels

        Returns:
            torch.Tensor: output prediction [D, B, R, dims * 2] where
                D=number of decoder layers, B=batch size, R=number of
                predictions, dims=number of spatial dimensions
        """
        return self.mlp(features)

    def compute_loss(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for given predictions and targets.

        #TODO: shapes
        Args:
            preds: predictions in specific format (in DETR like
                networks it usually is in center format)
            targets: targets in specific format (in DETR like
                networks it usually is in center format)
            pred_boxes: predicted bounding boxes (point format)
            target_boxes: target bounding boxes (point format)

        Returns:
            Dict[str, torch.Tensor]: regression loss(es) saved in
                `self.loss_name` and `self.box_loss_name` which are
                defined by the module
        """
        losses = {}
        if self.loss is not None:
            losses[self.loss_name] = self.loss(preds, targets, **kwargs)
        if self.box_loss is not None:
            losses[self.box_loss_name] = self.box_loss(pred_boxes, target_boxes, **kwargs)
        return losses


# TODO:
# L1 loss
# GIoU loss
# L1 + GIoU loss

# uniform init to model medical patch sampling prior
