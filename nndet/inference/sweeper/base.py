# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple


class Sweeper(ABC):
    evaluator_cls = None

    def __init__(
        self,
        classes: Sequence[str],
        pred_dir: os.PathLike,
        gt_dir: os.PathLike,
        target_metric: str,
        save_dir: Optional[os.PathLike] = None,
    ):
        """
        Sweep multiple parameters and compute evaluation metrics
        to determine the best set of parameters

        Args:
            evaluation: reference to an evaluation objects
            pred_dir: directory where predicted data is saved
            device: device to use for internal computations
        """
        self.classes = classes
        self.save_dir = save_dir if save_dir is None else Path(save_dir)
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.target_metric = target_metric

        self.device = "cpu"

        self.pred_dir = Path(pred_dir)
        self.gt_dir = Path(gt_dir)

    @abstractmethod
    def run_postprocessing_sweep(
        self,
        restore: bool = True,
    ) -> Tuple[Dict, Dict]:
        """
        Run parameter sweeps to determine best parameters
        accoring to target metric

        Args:
            target_metric: metric to optimize

        Returns:
            Dict: determined parameters
            Dict: final results with parameters
        """
        raise NotImplementedError
