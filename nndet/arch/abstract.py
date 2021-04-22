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

from typing import Dict, Tuple, Any, Optional

import torch
from abc import abstractmethod

from torch import Tensor


class AbstractModel(torch.nn.Module):
    @classmethod
    @abstractmethod
    def from_config_plan(cls,
                         model_cfg: dict,
                         plan_arch: dict,
                         plan_anchors: dict,
                         log_num_anchors: str = None,
                         **kwargs,
                         ):
        raise NotImplementedError

    @abstractmethod
    def train_step(self,
                   images: Tensor,
                   targets: dict,
                   evaluation: bool,
                   batch_num: int,
                   ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict]]:
        """
        Perform a single training step

        Args:
            images: images to process
            targets: labels for training
            evaluation (bool): compute final predictions which should be used for metric evaluation
            batch_num (int): batch index inside epoch

        Returns:
            torch.Tensor: final loss for back propagation
            Dict: predictions for metric calculation
            Dict[str, torch.Tensor]: scalars for logging (e.g. individual loss components)
        """
        raise NotImplementedError

    @abstractmethod
    def inference_step(self,
                       images: Tensor,
                       *args,
                       **kwargs,
                       ) -> Dict[str, Any]:
        """
        Perform a single training step

        Args:
            images: images to process
            *args: positional arguments
            **kwargs: keyword arguments

        Returns:
            Dict: predictions for metric calculation
        """
        raise NotImplementedError
