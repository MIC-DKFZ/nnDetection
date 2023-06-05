import copy
import math
from abc import abstractmethod
from typing import Dict, Optional

import torch
from loguru import logger

from nndet.losses.classification.ce import BCELoss, CELoss
from nndet.losses.classification.focal import BFocalLoss
from nndet.utils.typing import LINEARSEQ


class FFNClassifier(torch.nn.Module):
    def __init__(
        self,
        linear: LINEARSEQ,
        in_channels: int,
        internal_channels: int,
        num_classes: int,
        num_layers: int = 1,
        add_norm: bool = False,
        dropout_rate: float = 0.0,
        num_decoder_layers: int = 0,
        share_mlp: bool = True,
        use_encoder_mlp: bool = False,
        class_agnostic_aux: bool = False,
        binary_classes: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Feed forward network head (usually used in DETR like models)

        Args:
            linear: generator object to obtain linear layer blocks
            in_channels: number of input channels
            internal_channels: number of internal channels to use
            num_classes: number of foreground classes
            num_layers: Number of linear layers to use. Defaults to 1.
            add_norm: Add normalisation layers. Defaults to False.
            dropout_rate: Dropout probability in last layer. Defaults to 0.0.
            kwargs: passed to linear generator class
        """
        super().__init__()

        self.in_channels = in_channels
        self.internal_channels = internal_channels
        self.num_classes = num_classes
        self.num_layers = num_layers

        self.mlp = self._build_module(
            linear=linear,
            output_channels=self.num_classes,
            add_norm=add_norm,
            dropout_rate=dropout_rate,
            **kwargs,
        )

        self.encoder_mlp = None
        self.aux_mlp = None
        self.share_mlp = share_mlp
        self.num_decoder_layers = num_decoder_layers
        if not self.share_mlp:
            if not class_agnostic_aux:
                aux_mlp = self.mlp
            else:
                aux_mlp = self._build_module(linear, binary_classes, add_norm, dropout_rate, **kwargs)
            self.aux_mlp = torch.nn.ModuleList([copy.deepcopy(aux_mlp) for i in range(num_decoder_layers - 1)])
            if use_encoder_mlp:
                self.encoder_mlp = copy.deepcopy(aux_mlp)
        elif use_encoder_mlp:
            self.encoder_mlp = self.mlp

        self.loss_name: str = "ffn_cls"
        self.loss: Optional[torch.nn.Module] = None
        self.logits_convert_fn: Optional[torch.nn.Module] = None
        self.init_weights()

    def _build_module(
        self,
        linear: LINEARSEQ,
        output_channels: int,
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
            out_channels = output_channels if idx == self.num_layers - 1 else self.internal_channels
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
        return torch.nn.Sequential(*modules)

    def init_weights(self):
        """
        Init weights
        """
        pass

    def forward(self, features: torch.Tensor, layer: Optional[int] = None) -> torch.Tensor:
        """
        Forward feature through module

        Args:
            layer: index to know which MLP to use
            features: input feature [D, B, R, C] where D=number of decoder
                layers, B=batch size, R=number of predictions, C=number of
                channels

        Returns:
            torch.Tensor: output prediction [D, B, R, num_classes] where
                D=number of decoder layers, B=batch size, R=number of
                predictions, num_classes=number of classes
        """
        # If mlps are shared, or no layer is given, or the last layer is accessed, return the main mlp
        if self.share_mlp or layer is None or layer == self.num_decoder_layers - 1:
            return self.mlp(features)
        # else it is a not shared aux layer
        return self.aux_mlp[layer](features)

    def compute_loss(
        self,
        pred_logits: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for given predictions and targets

        Args:
            pred_logits: predicted logits [N, R, num_classes] where
                N=batch size, R=number of boxes, C=number of classes
            targets: classification targets [N, R] where N=batch size
                (numerical classes as expected by torch CE loss),
                R=number of predictions

        Returns:
            Dict[str, torch.Tensor]: classification loss saved in
                key `self.loss_name` which is defined by module
        """
        return {self.loss_name: self.loss(pred_logits, targets, **kwargs)}

    @abstractmethod
    def postprocess_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to probabilities and remove potential background class

        Args:
            logits: predicted logits [N, R, C] where N=batch size,
                R=number of boxes, C=number of classes

        Returns:
            torch.Tensor: converted logits [N, R, C] where N=batch size,
                R=number of boxes, C=number of classes
        """
        raise NotImplementedError

    def logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to probabilities

        Args:
            logits: predicted logits [N, R, C] where N=batch size,
                R=number of boxes, C=number of classes

        Returns:
            torch.Tensor: converted logits [N, R, C] where N=batch size,
                R=number of boxes, C=number of classes
        """
        return self.logits_convert_fn(logits)


class SoftmaxFFNClassifier(FFNClassifier):
    def __init__(
        self,
        linear: LINEARSEQ,
        in_channels: int,
        internal_channels: int,
        num_classes: int,
        num_layers: int = 1,
        add_norm: bool = False,
        dropout_rate: float = 0.0,
        **kwargs,
    ) -> None:
        """
        Feed forward network head (usually used in DETR like models)
        Uses softmax for nonlinearity

        Args:
            linear: generator object to obtain linear layer blocks
            in_channels: number of input channels
            internal_channels: number of internal channels to use
            num_classes: number of foreground classes
            num_layers: Number of linear layers to use. Defaults to 1.
            add_norm: Add normalisation layers. Defaults to False.
            dropout_rate: Dropout probability in last layer. Defaults to 0.0.
            kwargs: passed to linear generator class
        """
        super().__init__(
            linear=linear,
            in_channels=in_channels,
            internal_channels=internal_channels,
            num_classes=num_classes + 1,  # add one background softmax class
            num_layers=num_layers,
            add_norm=add_norm,
            dropout_rate=dropout_rate,
            binary_classes=2,
            **kwargs,
        )
        self.logits_convert_fn = torch.nn.Softmax(dim=-1)

    def postprocess_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to probabilities and remove potential background class

        Args:
            logits: predicted logits [N, R, C] where N=batch size,
                R=number of boxes, C=number of classes

        Returns:
            torch.Tensor: converted logits [N, R, C] where N=batch size,
                R=number of boxes, C=number of classes
        """
        return self.logits_to_probs(logits=logits)[..., 1:]  # remove background class


class SigmoidFFNClassifier(FFNClassifier):
    def __init__(
        self,
        linear: LINEARSEQ,
        in_channels: int,
        internal_channels: int,
        num_classes: int,
        num_layers: int = 1,
        add_norm: bool = False,
        dropout_rate: float = 0.0,
        prior_prob: Optional[float] = None,
        **kwargs,
    ) -> None:
        """
        Feed forward network head (usually used in DETR like models)
        Uses sigmoid for nonlinearity

        Args:
            linear: generator object to obtain linear layer blocks
            in_channels: number of input channels
            internal_channels: number of internal channels to use
            num_classes: number of foreground classes
            num_layers: Number of linear layers to use. Defaults to 1.
            add_norm: Add normalisation layers. Defaults to False.
            dropout_rate: Dropout probability in last layer. Defaults to 0.0.
            prior_prob: initialize final layer with given prior probability
            kwargs: passed to linear generator class
        """
        self.prior_prob = prior_prob

        super().__init__(
            linear=linear,
            in_channels=in_channels,
            internal_channels=internal_channels,
            num_classes=num_classes,
            num_layers=num_layers,
            add_norm=add_norm,
            dropout_rate=dropout_rate,
            binary_classes=1,
            **kwargs,
        )
        self.logits_convert_fn = torch.nn.Sigmoid()

    def postprocess_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to probabilities and remove potential background class

        Args:
            logits: predicted logits [N, R, C] where N=batch size,
                R=number of boxes, C=number of classes

        Returns:
            torch.Tensor: converted logits [N, R, C] where N=batch size,
                R=number of boxes, C=number of classes
        """
        return self.logits_to_probs(logits=logits)

    def init_weights(self) -> None:
        """
        Init weights with prior prob
        """
        if self.prior_prob is not None:
            logger.info(f"Init FFN classifier weights: prior prob {self.prior_prob}")
            # Use prior in model initialization to improve stability
            bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
            torch.nn.init.constant_(self.mlp[-1][-1].bias, bias_value)
            if self.aux_mlp is not None and not self.share_mlp:
                for mlp in self.aux_mlp:
                    torch.nn.init.constant_(mlp[-1][-1].bias, bias_value)
            if self.encoder_mlp is not None and not self.share_mlp:
                torch.nn.init.constant_(self.encoder_mlp[-1][-1].bias, bias_value)
        else:
            logger.info("Init FFN classifier weights: default")


class CEFFNClassifier(SoftmaxFFNClassifier):
    def __init__(
        self,
        linear: LINEARSEQ,
        in_channels: int,
        internal_channels: int,
        num_classes: int,
        num_layers: int = 1,
        add_norm: bool = False,
        dropout_rate: float = 0.0,
        weight: Optional[torch.Tensor] = None,
        background_weight: Optional[float] = None,
        reduction: str = "sum",
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        **kwargs,
    ) -> None:
        """
        Feed forward network head (usually used in DETR like models)
        Computes the CrossEntropy loss (softmax based)

        Args:
            linear: generator object to obtain linear layer blocks
            in_channels: number of input channels
            internal_channels: number of internal channels to use
            num_classes: number of foreground classes
            num_layers: Number of linear layers to use. Defaults to 1.
            add_norm: Add normalisation layers. Defaults to False.
            dropout_rate: Dropout probability in last layer. Defaults to 0.0.
            weight: weight in cross entrpoy loss (see pytorch for more info)
            background_weight: weight for background class. Can only be used if
                no other weight is provided.
            reduction: reduction to apply to loss. 'sum' | 'mean' | 'none'
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            kwargs: passed to linear generator class
        """
        super().__init__(
            linear=linear,
            in_channels=in_channels,
            internal_channels=internal_channels,
            num_classes=num_classes,
            num_layers=num_layers,
            add_norm=add_norm,
            dropout_rate=dropout_rate,
            **kwargs,
        )

        if background_weight is not None:
            if weight is not None:
                raise ValueError("Received background weight and weight tensor for CE loss")
            weight = torch.ones(self.num_classes, device="cuda")
            weight[0] = background_weight

        self.loss_name = "ffn_ce"
        self.loss = CELoss(
            weight=weight,
            reduction=reduction,
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
        )


class BCEFFNClassifier(SigmoidFFNClassifier):
    def __init__(
        self,
        linear: LINEARSEQ,
        in_channels: int,
        internal_channels: int,
        num_classes: int,
        num_layers: int = 1,
        add_norm: bool = False,
        dropout_rate: float = 0.0,
        prior_prob: Optional[float] = None,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "sum",
        smoothing: float = 0.0,
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        **kwargs,
    ) -> None:
        """
        Feed forward network head (usually used in DETR like models)
        Computes the BinaryCrossEntropy loss (sigmoid based)

        Args:
            linear: generator object to obtain linear layer blocks
            in_channels: number of input channels
            internal_channels: number of internal channels to use
            num_classes: number of foreground classes
            num_layers: Number of linear layers to use. Defaults to 1.
            add_norm: Add normalisation layers. Defaults to False.
            dropout_rate: Dropout probability in last layer. Defaults to 0.0.
            prior_prob: initialize final layer with given prior probability
            weight: weight in BCEWithLogitsLoss (see pytorch for more info)
            reduction: reduction to apply to loss. 'sum' | 'mean' | 'none'
            smoothing:  label smoothing
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            kwargs: passed to linear generator class
        """
        super().__init__(
            linear=linear,
            in_channels=in_channels,
            internal_channels=internal_channels,
            num_classes=num_classes,
            num_layers=num_layers,
            add_norm=add_norm,
            dropout_rate=dropout_rate,
            prior_prob=prior_prob,
            **kwargs,
        )
        self.loss_name = "ffn_bce"
        self.loss = BCELoss(
            weight=weight,
            reduction=reduction,
            smoothing=smoothing,
            loss_weight=loss_weight,
            loss_fp32=loss_fp32,
        )


class FocalFFNClassifier(SigmoidFFNClassifier):
    def __init__(
        self,
        linear: LINEARSEQ,
        in_channels: int,
        internal_channels: int,
        num_classes: int,
        num_layers: int = 1,
        add_norm: bool = False,
        dropout_rate: float = 0.0,
        prior_prob: Optional[float] = None,
        gamma: float = 2,
        alpha: float = -1,
        reduction: str = "sum",
        smoothing: float = 0.0,
        loss_weight: float = 1.0,
        loss_fp32: bool = False,
        **kwargs,
    ) -> None:
        """
        Feed forward network head (usually used in DETR like models)
        Computes the FocalLoss loss (sigmoid based)

        Args:
            linear: generator object to obtain linear layer blocks
            in_channels: number of input channels
            internal_channels: number of internal channels to use
            num_classes: number of foreground classes
            num_layers: Number of linear layers to use. Defaults to 1.
            add_norm: Add normalisation layers. Defaults to False.
            dropout_rate: Dropout probability in last layer. Defaults to 0.0.
            prior_prob: initialize final layer with given prior probability
            gamma: focal loss gamma
            alpha: focal loss alpha
            reduction: reduction to apply to loss. 'sum' | 'mean' | 'none'
            smoothing:  label smoothing
            loss_weight: scalar to balance multiple losses
            loss_fp32: If True, loss is forced to be computed in float32
            kwargs: passed to linear generator class
        """
        super().__init__(
            linear=linear,
            in_channels=in_channels,
            internal_channels=internal_channels,
            num_classes=num_classes,
            num_layers=num_layers,
            add_norm=add_norm,
            dropout_rate=dropout_rate,
            prior_prob=prior_prob,
            **kwargs,
        )
        self.loss_name = "ffn_focal"
        self.loss = BFocalLoss(
            gamma=gamma,
            alpha=alpha,
            loss_fp32=loss_fp32,
            loss_weight=loss_weight,
            reduction=reduction,
            smoothing=smoothing,
        )
