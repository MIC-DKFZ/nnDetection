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

from abc import ABC, abstractmethod
from pathlib import Path
import time
from typing import Callable, Tuple, Dict, Sequence, Any, Optional, TypeVar

import numpy as np
from loguru import logger

from nndet.io.paths import Pathlike
from nndet.io.load import save_json
from nndet.utils.info import maybe_verbose_iterable
from nndet.utils import to_numpy
from nndet.evaluator.registry import BoxEvaluator


class Sweeper(ABC):
    def __init__(self,
                 classes: Sequence[str],
                 pred_dir: Pathlike,
                 gt_dir: Pathlike,
                 target_metric: str,
                 save_dir: Optional[Pathlike] = None,
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
    def run_postprocessing_sweep(self,
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


class BoxSweeper(Sweeper):
    def __init__(self,
                 classes: Sequence[str],
                 pred_dir: Pathlike,
                 gt_dir: Pathlike,
                 target_metric: str,
                 ensembler_cls: Callable,
                 save_dir: Optional[Pathlike] = None,
                 ) -> None:
        """
        Run sweep over parameters and select the best

        Args:
            classes: classes present in dataset 
            pred_dir: directory where predictions are saved
            gt_dir: directory where ground truth is saved
            target_metric: metric to optimize
            ensembler_cls: ensembler class used during prediction
            save_dir: Directory to save results. Defaults to None.
        """
        super().__init__(classes=classes,
                         pred_dir=pred_dir,
                         gt_dir=gt_dir,
                         target_metric=target_metric,
                         save_dir=save_dir,
                         )

        self.evaluator_cls = BoxEvaluator
        self.ensembler_cls = ensembler_cls

    def run_postprocessing_sweep(self):
        """
        Sequentially search for the best parameters

        Returns:
            Dict: final parameters to run inference on new cases
            Dict:
                `det_scores`: detection score metrics
                `det_curves`: detection curves
        """
        state, sweep_params = self.ensembler_cls.sweep_parameters()
        num_cases = self.ensembler_cls.get_case_ids(self.pred_dir)
        logger.info(f"Running parameter sweep on {num_cases} cases to optimize "
                    f"{self.target_metric} with initial state {state}.")

        best_score = float('-inf')
        for param_name, values in sweep_params.items():
            best_value, _best_score = self.run_parameter(
                values=values,
                param_name=param_name,
                state=state,
                )
            state[param_name] = best_value

            if _best_score < best_score:
                logger.error("ERROR: Something went wrong during sweeping. "
                             "Results were modified inplace! "
                             f"Previous: {best_score} now {_best_score}")
            best_score = _best_score

        logger.info(f"\n\n Determined {state} with best sweeping score {best_score} {self.target_metric}\n\n")
        return state

    def run_parameter(self,
                      values: Sequence[Any],
                      param_name: str,
                      state: Dict[str, Any],
                      ):
        """
        Evaluate parameters and select the best
        
        Args:
            values: values to evaluate
            param_name: name of parameter
            state: different state parameters
        """
        cache = []
        overview = {}
        for value in values:
            logger.info(f"Running sweep {param_name}={value}")
            tic = time.perf_counter()
            metric_scores = self._evaluate_value(state=state, **{param_name: value})
            overview[f"{param_name}_{value}".replace(".", "_")] = {
                "state": str(state),
                "overwrite": {param_name: str(value)},
                "scores": str(metric_scores),
            }
            cache.append(metric_scores[self.target_metric])
            toc = time.perf_counter()
            logger.info(f"Sweep took {toc - tic} s")

        best_idx = np.argmax(cache)
        best_value = values[best_idx]
        best_score = cache[best_idx]

        if self.save_dir is not None:
            overview[f"best_{param_name}"] = {"value": str(best_value), "score": str(best_score)}
            save_json(overview, self.save_dir / f"sweep_{param_name}.json")
        return best_value, best_score

    def _evaluate_value(self,
                        state: Dict[str, Any],
                        **overwrite,
                        ):
        """
        Evalaute a single value

        Args:
            state: state for ensembler
            overwrite: state overwrites

        Returns:
            Dict: scalar metrics
        """
        evaluator = self.evaluator_cls.create(classes=self.classes,
                                              fast=True,
                                              verbose=False,
                                              save_dir=None,
                                              )

        for case_id in maybe_verbose_iterable(self.ensembler_cls.get_case_ids(self.pred_dir)):
            ensembler = self.ensembler_cls.from_checkpoint(
                base_dir=self.pred_dir, case_id=case_id, device=self.device,
                )
            ensembler.update_parameters(**state)
            ensembler.update_parameters(**overwrite)

            pred = to_numpy(ensembler.get_case_result(restore=False))
            gt = np.load(str(self.gt_dir / f"{case_id}_boxes_gt.npz"), allow_pickle=True)

            evaluator.run_online_evaluation(
                pred_boxes=[pred["pred_boxes"]], pred_classes=[pred["pred_labels"]],
                pred_scores=[pred["pred_scores"]], gt_boxes=[gt["boxes"]],
                gt_classes=[gt["classes"]], gt_ignore=None,
            )

        metric_scores, _ = evaluator.finish_online_evaluation()
        return metric_scores


SweeperType = TypeVar('SweeperType', bound=Sweeper)
