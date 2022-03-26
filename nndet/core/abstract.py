"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor


class AbstractDetector(torch.nn.Module):
    @classmethod
    @abstractmethod
    def from_config_plan(
        cls,
        model_cfg: dict,
        plan_arch: dict,
        plan_anchors: dict,
        log_num_anchors: str = None,
        **kwargs,
    ):
        raise NotImplementedError

    @abstractmethod
    def train_step(
        self,
        images: Tensor,
        targets: dict,
        predict: bool,
        batch_num: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a single training step -> only losses are computed

        Args:
            images: images to process
            targets: labels for training
            batch_num: batch index inside epoch

        Returns:
            Dict[str, torch.Tensor]: losses
        """
        raise NotImplementedError

    @abstractmethod
    def validation_step(
        self,
        images: Tensor,
        targets: dict,
        batch_num: bool,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict]]:
        """
        Perform a single validation step -> losses (without grad) and
        predictions are computed

        Args:
            images: images to process
            targets: labels for training
            batch_num: batch index inside epoch

        Returns:
            Dict[str, torch.Tensor]: losses
            Optional[Dict]: predictions; only if `predict=True`
        """
        raise NotImplementedError

    @abstractmethod
    def inference_step(
        self,
        images: Tensor,
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Perform a single inference step -> only predictions are computed

        Args:
            images: images to process
            *args: positional arguments
            **kwargs: keyword arguments

        Returns:
            Dict: predictions
        """
        raise NotImplementedError


class AbstractOneStageDetector(AbstractDetector):
    def train_step_with_features(
        self,
        images: Tensor,
        targets: dict,
        predict: bool,
        batch_num: int,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict], List[torch.Tensor]]:
        """
        Perform a single training step and return feature maps
        Only needed for one stage detectors

        Args:
            images: images to process
            targets: labels for training
            predict: compute final predictions which should be used for metric evaluation
            batch_num: batch index inside epoch

        Returns:
            Dict[str, torch.Tensor]: losses
            Optional[Dict]: predictions; only if `predict=True`
            List[torch.Tensor]: feature maps from backbones
        """
        raise NotImplementedError

    def inference_step_with_features(
        self,
        images: Tensor,
        **kwargs,
    ) -> Union[Dict[str, Any], List[torch.Tensor]]:
        """
        Perform a single inference step
        Only needed for one stage detectors

        Args:
            images: images to process
            *args: positional arguments
            **kwargs: keyword arguments

        Returns:
            Dict: predictions
            List[torch.Tensor]: feature maps from backbones
        """
        raise NotImplementedError
