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

from collections import defaultdict
from typing import Dict, Sequence, Callable, Tuple, Union, Mapping, Optional

import numpy as np
from loguru import logger
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, \
    f1_score, precision_score, recall_score, roc_auc_score

from nndet.evaluator import AbstractEvaluator
from nndet.utils.info import experimental


__all__ = ["CaseEvaluator"]


class _CaseEvaluator(AbstractEvaluator):
    @experimental
    def __init__(self,
                 classes: Sequence[Union[str, int]],
                 score_metrics_scalar: Mapping[str, Callable] = None,
                 class_metrics_scalar: Mapping[str, Callable] = None,
                 score_metrics_curve: Mapping[str, Callable] = None,
                 class_metrics_curve: Mapping[str, Callable] = None,
                 target_class: Optional[int] = None,
                 ):
        """
        Compute case level evaluation metrics
        Predictions for individual instances are aggregated by using the
        max of the predicted score for each class. Final class prediction
        is computed by an argmax over that scores. The mappings of the
        metrics are later used as the keys of the result dict.

        Note this implementation is experimental and might change in the
        future.

        Args:
            classes: class present in whole dataset
            score_metrics_scalar: metrics which accept ground truth classes [N]
                and prediction scores [N, C] for evaluation; N is the nunber
                of cases and C is the number of classes. The output should
                be a scalar.
            class_metrics_scalar: metrics which accept ground truth classes [N]
                and prediction classes [N] for evaluation; N is the nunber
                of cases and C is the number of classes. The output should 
                be a scalar.
            score_metrics_curve: metrics which accept ground truth classes [N]
                and prediction scores [N, C] for evaluation; N is the nunber
                of cases and C is the number of classes. The output should be
                an array like object.
            class_metrics_curve: metrics which accept ground truth classes [N]
                and prediction classes [N] for evaluation; N is the nunber
                of cases and C is the number of classes. The output should 
                be an array like object.
            target_class: target class for case evaluation (internally
                results are evaluated in a binary case target class vs rest).
                If None, fall back to fg vs bg
        """
        self.results_list = defaultdict(list)

        self.score_metrics_scalar = score_metrics_scalar if score_metrics_scalar is not None else {}
        self.class_metrics_scalar = class_metrics_scalar if class_metrics_scalar is not None else {}
        self.score_metrics_curve = score_metrics_curve if score_metrics_curve is not None else {}
        self.class_metrics_curve = class_metrics_curve if class_metrics_curve is not None else {}

        if isinstance(target_class, str):
            raise ValueError(f"Need integer value of target class not the name!")

        self.target_class = int(target_class)
        self.classes = classes
        self.num_classes = len(classes)

    def reset(self):
        """
        Reset internal state for new epoch
        """
        self.results_list = defaultdict(list)

    def run_online_evaluation(self,
                              pred_classes: Sequence[np.ndarray],
                              pred_scores: Sequence[np.ndarray],
                              gt_classes: Sequence[np.ndarray],
                              ) -> Dict:
        """
        Run evaluation on each case (accepts a batch of case resutls
        at once).

        Args:
            pred_classes (Sequence[np.ndarray]): predicted classes from a batch
                of cases; List[[D]], D number of predictions
            pred_scores (Sequence[np.ndarray]): predicted score for each
                bounding box; List[[D]], D number of predictions
            gt_classes (Sequence[np.ndarray]): ground truth classes for each
                instance in a case; List[[G]], G number of ground truth

        Returns:
            Dict: empty dict
        
        Notes:
            This caches the max predicted probability per class per element
            and the unique classes present per element.
        """
        case_classes = [np.unique(gtc) for gtc in gt_classes]
        case_scores = []
        for case_instance_scores, case_instance_classes in zip(pred_scores, pred_classes):
            _scores = np.zeros(self.num_classes)
            for instance_score, instance_class in zip(case_instance_scores, case_instance_classes):
                if _scores[int(instance_class)] < instance_score:
                    _scores[int(instance_class)] = instance_score
            case_scores.append(_scores)

        self.results_list["case_classes"].extend(case_classes)
        self.results_list["case_scores"].extend(case_scores)
        return  {}

    def finish_online_evaluation(self) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """
        Compute final scores and curves of metrics

        Returns:
            Dict: results of scalar metrics
            Dict: results of curve metrics
        """
        # aggregate cases
        gt_classes = self.aggregate_classes()
        pred_scores, pred_classes = self.aggregate_prdictions()

        # compute metrics
        curve_results = {}
        for key, metric in self.score_metrics_curve.items():
            curve_results[key] = metric(gt_classes, pred_scores)
        for key, metric in self.class_metrics_curve.items():
            curve_results[key] = metric(gt_classes, pred_classes)

        scalar_results = {}
        for key, metric in self.score_metrics_scalar.items():
            try:
                scalar_results[key] = metric(gt_classes, pred_scores)
            except (ValueError, RuntimeError) as e:
                logger.warning(f"Metric {key} exited with error {e}; writing nan to result")
                scalar_results[key] = np.nan
        for key, metric in self.class_metrics_scalar.items():
            try:
                scalar_results[key] = metric(gt_classes, pred_classes)
            except (ValueError, RuntimeError) as e:
                logger.warning(f"Metric {key} exited with error {e}; writing nan to result")
                scalar_results[key] = np.nan
        return scalar_results, curve_results

    def aggregate_classes(self) -> np.ndarray:
        """
        Aggregate classes of each instance in a case to one case class
        
        Returns:
            np.ndarray: class per case [N], where N is the number of cases
        """
        if self.target_class is not None:
            gt_classes = np.asarray(
                [int(self.target_class in cc) for cc in self.results_list["case_classes"]])
        else:
            gt_classes = np.asarray(
                [1 if len(cc) > 0 else 0 for cc in self.results_list["case_classes"]])
        return gt_classes

    def aggregate_prdictions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aggreagte prediction scores per class to case scores with target class

        Returns:
            np.ndarray: predicted scores
            np.ndarray: predicted classes
        """
        _pred_scores = np.stack(self.results_list["case_scores"], axis=0) # N, num_classes
        
        if self.target_class is not None:
            pred_scores = _pred_scores[:, self.target_class] # N
            # pred_classes = (np.argmax(_pred_scores, axis=1) == self.target_class).astype(np.int32) # N
            # This is not always the correct choice, depending on the final
            # nonlinearity of the network (sigmoid vs. softmax)
            pred_classes = (pred_scores > 0.5).astype(np.int32) # N
        else:
            pred_scores = _pred_scores.max(axis=1) # N
            pred_classes = (pred_scores > 0.5).astype(np.int32) # N
        return pred_scores, pred_classes


class CaseEvaluator(_CaseEvaluator):
    @classmethod
    def create(cls,
               classes: Sequence[str],
               target_class: int = None
               ):
        """
        Evaluation on patient level

        Args:
            classes: classes present in dataset
            target_class: if multiple classes are given, define
                a target class to evaluate in an target_class vs rest setting.
                Defaults to None.
        
        Returns:
            CaseEvaluator: evaluator
        """
        # if len(classes) > 2 and target_class is None:
        #     f1_fn = partial(f1_score, average="macro")
        #     prec_fn = partial(precision_score, average="macro")
        #     rec_fn = partial(recall_score, average="macro")
        # else:
        f1_fn = f1_score
        prec_fn = precision_score
        rec_fn = recall_score

        score_metrics_scalar = {"auc_case": roc_auc_score, "ap_case": average_precision_score}
        class_metrics_scalar = {"f1_case": f1_fn, "prec_case": prec_fn,
                                "rec_case": rec_fn, "acc_case": accuracy_score}
        score_metrics_curve = {}
        class_metrics_curve = {"cfm_case": confusion_matrix}
        return cls(classes=classes,
                   score_metrics_scalar=score_metrics_scalar,
                   class_metrics_scalar=class_metrics_scalar,
                   score_metrics_curve=score_metrics_curve,
                   class_metrics_curve=class_metrics_curve,
                   target_class=target_class,
                   )
