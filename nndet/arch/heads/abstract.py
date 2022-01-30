from abc import abstractmethod
from typing import Dict, List, Optional, TypeVar

import torch
import torch.nn as nn
from torch import Tensor

from nndet.core.boxes import BoxCoderND

CONV_TYPES = (nn.Conv2d, nn.Conv3d)


class Classifier(nn.Module):
    @abstractmethod
    def compute_loss(
        self,
        pred_logits: Tensor,
        targets: Tensor,
        **kwargs,
    ) -> Tensor:
        """
        Compute classification loss (cross entropy loss)

        Args:
            pred_logits (Tensor): predicted logits
            targets (Tensor): classification targets

        Returns:
            Tensor: classification loss
        """
        raise NotImplementedError

    @abstractmethod
    def logits_to_probs(
        self,
        logits: Tensor,
    ) -> Tensor:
        """
        Convert bounding box logits to probabilities

        Args:
            logits (Tensor): bounding box logits
                [N, C], C=number of classes

        Returns:
            Tensor: probabilities; [N, C], C=number of classes
        """
        raise NotImplementedError


class Regressor(nn.Module):
    @abstractmethod
    def compute_loss(
        self,
        pred_deltas: Tensor,
        target_deltas: Tensor,
        **kwargs,
    ) -> Tensor:
        """
        Compute regression loss

        Args:
            pred_deltas (Tensor): predicted bounding box deltas
                [N, (num_classes), dim * 2]
            target_deltas (Tensor): target bounding box deltas
                [N,  dim * 2]

        Returns:
            Tensor: loss
        """
        raise NotImplementedError

    @classmethod
    def class_agnostic(cls):
        """
        True if anchors are regressed in a class agnostic manner.
        False if anchors are regressed for each class separately.
        """
        return True


ClassifierType = TypeVar("ClassifierType", bound=Classifier)
RegressorType = TypeVar("RegressorType", bound=Regressor)


class BaseHead(nn.Module):
    """
    Provides an abstract interface for an module which takes
    inputs and computed its own loss
    """

    def __init__(
        self,
        classifier: ClassifierType,
        regressor: RegressorType,
        coder: BoxCoderND,
        shared: Optional[torch.nn.Module] = None,
    ):
        """
        Provides an abstract interface for an module which takes
        inputs and computed its own loss

        Args:
            classifier: classifier module
            regressor: regression module
            coder: Module to encoder/decoder box delta wrt to anchors/proposals
            shared: optional shared module which is applied to before the
                classifier and regression head
        """
        super().__init__()
        self.classifier = classifier
        self.regressor = regressor
        self.shared = shared
        self.coder = coder

    @abstractmethod
    def forward(
        self,
        x: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute forward pass

        Args
            x: feature maps
        """
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute loss
        """
        raise NotImplementedError

    @abstractmethod
    def postprocess_for_inference(
        self,
        prediction: Dict[str, torch.Tensor],
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Postprocess predictions for inference
        e.g. convert logits to probs; decode box deltas

        Args:
            Dict[str, torch.Tensor]: predictions from this head
            List[torch.Tensor]: anchors per image

        Returns:
            Dict[str, torch.Tensor]: postprocessed predictions
        """
        raise NotImplementedError

    @property
    def class_agnostic(self) -> bool:
        """
        Return if regression is performed per class or not.
        True => each anchor is regressed for each class separately
        False => each anchor is regressed once
        """
        return self.regressor.class_agnostic


HeadType = TypeVar("HeadType", bound=BaseHead)


class RoIConv1x1View(torch.nn.Module):
    def __init__(self, dim: int):
        """
        Small helper class to reshape input tensors for RoI processing
        of 1x1 convs immitating fully connected layers.

        Args:
            dim: number of spatial dimensions
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], -1, *[1] * self.dim)
