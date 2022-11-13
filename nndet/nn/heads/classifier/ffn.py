import math
from abc import abstractmethod
from typing import Dict, Optional

import torch
from loguru import logger

from nndet.losses.classification.bce import BinaryCrossEntropyLoss
from nndet.losses.classification.ce import CrossEntropyLoss
from nndet.losses.classification.focal import FocalLossWithLogits
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
            add_norm=add_norm,
            dropout_rate=dropout_rate,
            **kwargs,
        )

        self.loss_name: str = "ffn_cls"
        self.loss: Optional[torch.nn.Module] = None
        self.logits_convert_fn: Optional[torch.nn.Module] = None
        self.init_weights()

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
            out_channels = self.num_classes if idx == self.num_layers - 1 else self.internal_channels
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
            torch.Tensor: output prediction [D, B, R, num_classes] where
                D=number of decoder layers, B=batch size, R=number of
                predictions, num_classes=number of classes
        """
        return self.mlp(features)

    def compute_loss(
        self,
        pred_logits: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for given predictions and targets

        Args:
            pred_logits: predicted logits [N, num_classes, R] where
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
        return self.logits_to_probs(logits=logits)[..., :-1]  # remove background class


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
            for layer in self.modules():
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    if layer.bias is not None:
                        torch.nn.init.constant_(layer.bias, 0)

            # Use prior in model initialization to improve stability
            bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
            torch.nn.init.constant_(self.mlp[-1].fc.bias, bias_value)
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
        self.loss = CrossEntropyLoss(
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
        self.loss = BinaryCrossEntropyLoss(
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
        self.loss = FocalLossWithLogits(
            gamma=gamma,
            alpha=alpha,
            loss_fp32=loss_fp32,
            loss_weight=loss_weight,
            reduction=reduction,
            smoothing=smoothing,
        )
