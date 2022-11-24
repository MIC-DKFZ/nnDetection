from typing import Dict, Optional

import torch

from nndet.losses.regression.giou import GIoULossPaired
from nndet.losses.regression.smoothl1 import SmoothL1Loss
from nndet.utils.typing import LINEARSEQ


class FFNRegressor(torch.nn.Module):
    def __init__(
        self,
        linear: LINEARSEQ,
        in_channels: int,
        internal_channels: int,
        dim: int,
        num_layers: int = 1,
        add_norm: bool = False,
        dropout_rate: float = 0.0,
        **kwargs,
    ) -> None:
        """
        Feed forward network head (usually used in DETR like models)

        Args:
            linear: generator object to obtain linear layer blocks
            in_channels: number of input channels
            internal_channels: number of internal channels to use
            dim: number of spatial dimensions
            num_layers: Number of linear layers to use. Defaults to 1.
            add_norm: Add normalisation layers. Defaults to False.
            dropout_rate: Dropout probability in last layer. Defaults to 0.0.
            kwargs: passed to linear generator class
        """
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"Need at least one linear layer in FFN head got {num_layers}!")

        self.in_channels = in_channels
        self.internal_channels = internal_channels
        self.num_layers = num_layers
        self.dim = dim

        self.mlp = self._build_module(
            linear=linear,
            add_norm=add_norm,
            dropout_rate=dropout_rate,
            **kwargs,
        )

        self.loss_name: str = "ffn_reg_spec"
        self.box_loss_name: str = "ffn_reg_box"
        self.loss: Optional[torch.nn.Module] = None
        self.box_loss: Optional[torch.nn.Module] = None
        self.init_weights()

        # normalize predictions
        self.logits_convert_fn = torch.nn.Sigmoid()

    def _build_module(
        self,
        linear: LINEARSEQ,
        add_norm: bool,
        dropout_rate: float,
        **kwargs,
    ) -> torch.nn.Module:
        """
        Build FFN module

        Args:
            linear: generator object to obtain linear layer blocks
            add_norm: Add normalisation layers. Defaults to False.
            dropout_rate: Dropout probability in last layer. Defaults to 0.0.
            kwargs: passed to linear generator class

        Returns:
            torch.nn.Module: created module
        """
        modules = []
        for idx in range(self.num_layers):
            in_channels = self.in_channels if idx == 0 else self.internal_channels
            out_channels = self.dim * 2 if idx == self.num_layers - 1 else self.internal_channels
            # no norm and act in last layer
            add_norm = add_norm if idx < self.num_layers - 1 else False
            add_act = True if idx < self.num_layers - 1 else False
            # dropout in last layer
            dropout_rate = 0.0 if idx < self.num_layers - 1 else dropout_rate

            modules.append(
                linear(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    add_norm=add_norm,
                    add_act=add_act,
                    **kwargs,
                )
            )

        if len(modules) == 1:
            return modules[0]
        else:
            return torch.nn.Sequential(*modules)

    def init_weights(self):
        """
        Init weights
        """
        pass

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward feature through module

        Args:
            features: input feature [D, B, R, C] where D=number of decoder
                layers, B=batch size, R=number of predictions, C=number of
                channels

        Returns:
            torch.Tensor: output prediction, normalized image coordinates
                [D, B, R, dims * 2] where D=number of decoder layers,
                B=batch size, R=number of predictions, dims=number of
                spatial dimensions
        """
        return self.logits_convert_fn(self.mlp(features))
        # return self.mlp(features)

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

        Args:
            preds: predictions in specific format (in DETR like
                networks it usually is in center format) [N, dims * 2]
                where N=number of boxes, dims=number of spatial dimensions
            targets: targets in specific format (in DETR like
                networks it usually is in center format) [N, dims * 2]
                where N=number of boxes, dims=number of spatial dimensions
            pred_boxes: predicted bounding boxes (point format)
                [N, dims * 2] where N=number of boxes, dims=number of
                spatial dimensions
            target_boxes: target bounding boxes (point format)
                [N, dims * 2] where N=number of boxes, dims=number of
                spatial dimensions

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


class L1FFNRegressor(FFNRegressor):
    def __init__(
        self,
        linear: LINEARSEQ,
        in_channels: int,
        internal_channels: int,
        dim: int,
        num_layers: int = 1,
        add_norm: bool = False,
        dropout_rate: float = 0.0,
        # Loss paramter
        beta: float = 1.0,
        reduction: Optional[str] = "sum",
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        **kwargs,
    ) -> None:
        """
        Feed forward network head (usually used in DETR like models)
        trained with L1 loss

        Args:
            linear: generator object to obtain linear layer blocks
            in_channels: number of input channels
            internal_channels: number of internal channels to use
            dim: number of spatial dimensions
            num_layers: Number of linear layers to use. Defaults to 1.
            add_norm: Add normalisation layers. Defaults to False.
            dropout_rate: Dropout probability in last layer. Defaults to 0.0.
            beta: L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            reduction: reduction to apply to loss. 'sum' | 'mean' | 'none'
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            kwargs: passed to linear generator class
        """
        super().__init__(
            linear=linear,
            in_channels=in_channels,
            internal_channels=internal_channels,
            dim=dim,
            num_layers=num_layers,
            add_norm=add_norm,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        self.loss_name = "ffn_reg_l1"
        self.loss = SmoothL1Loss(
            beta=beta,
            reduction=reduction,
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
        )


class GIoUFFNRegressor(FFNRegressor):
    def __init__(
        self,
        linear: LINEARSEQ,
        in_channels: int,
        internal_channels: int,
        dim: int,
        num_layers: int = 1,
        add_norm: bool = False,
        dropout_rate: float = 0.0,
        # Loss paramter
        reduction: Optional[str] = "sum",
        box_loss_weight: float = 1.0,
        box_loss_fp32: bool = False,
        **kwargs,
    ) -> None:
        """
        Feed forward network head (usually used in DETR like models)
        trained with GIoU loss

        Args:
            linear: generator object to obtain linear layer blocks
            in_channels: number of input channels
            internal_channels: number of internal channels to use
            dim: number of spatial dimensions
            num_layers: Number of linear layers to use. Defaults to 1.
            add_norm: Add normalisation layers. Defaults to False.
            dropout_rate: Dropout probability in last layer. Defaults to 0.0.
            reduction: reduction to apply to loss. 'sum' | 'mean' | 'none'
            box_loss_weight: scalar to balance multiple losses
            box_loss_fp32: If True, loss is forced to be computed in float32
            kwargs: passed to linear generator class
        """
        super().__init__(
            linear=linear,
            in_channels=in_channels,
            internal_channels=internal_channels,
            dim=dim,
            num_layers=num_layers,
            add_norm=add_norm,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        self.box_loss_name = "ffn_reg_giou"
        self.box_loss = GIoULossPaired(
            reduction=reduction,
            loss_weight=box_loss_weight,
            loss_fp32=box_loss_fp32,
        )


class L1GIoUFFNRegressor(FFNRegressor):
    def __init__(
        self,
        linear: LINEARSEQ,
        in_channels: int,
        internal_channels: int,
        dim: int,
        num_layers: int = 1,
        add_norm: bool = False,
        dropout_rate: float = 0.0,
        # Loss paramter
        beta: float = 1.0,
        reduction: Optional[str] = "sum",
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        box_loss_weight: float = 1.0,
        box_loss_fp32: bool = False,
        **kwargs,
    ) -> None:
        """
        Feed forward network head (usually used in DETR like models)
        trained with L1 + GIoU loss

        Args:
            linear: generator object to obtain linear layer blocks
            in_channels: number of input channels
            internal_channels: number of internal channels to use
            dim: number of spatial dimensions
            num_layers: Number of linear layers to use. Defaults to 1.
            add_norm: Add normalisation layers. Defaults to False.
            dropout_rate: Dropout probability in last layer. Defaults to 0.0.
            beta: L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            reduction: reduction to apply to loss. 'sum' | 'mean' | 'none'
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            kwargs: passed to linear generator class
        """
        super().__init__(
            linear=linear,
            in_channels=in_channels,
            internal_channels=internal_channels,
            dim=dim,
            num_layers=num_layers,
            add_norm=add_norm,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        self.loss_name = "ffn_reg_l1"
        self.loss = SmoothL1Loss(
            beta=beta,
            reduction=reduction,
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
        )
        self.box_loss_name = "ffn_reg_giou"
        self.box_loss = GIoULossPaired(
            reduction=reduction,
            loss_weight=box_loss_weight,
            loss_fp32=box_loss_fp32,
        )
