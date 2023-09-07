# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
#
# Parts of this code are from cocoapi licensed under
# SPDX-FileCopyrightText: 2014, Piotr Dollar and Tsung-Yi Lin
# SPDX-License-Identifier: BSD-2-Clause-Views

import time
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
from loguru import logger

from nndet.eval import DetectionMetric


class CocoAPMetric(DetectionMetric):
    def __init__(
        self,
        classes: Sequence[str],
        iou_list: Sequence[float] = (0.1, 0.5, 0.75),
        iou_range: Sequence[float] = (0.1, 0.5, 0.05),
        verbose: bool = True,
    ):
        """
        Computes AP metric similar to COCO implementation. In contrast
        to the original COCO implementation this class does *not* limit
        the predictions per class per image! The number of predictions
        is already limited before the matching to reduce compute beforehand
        and move it to one place.

        Metrics computed:
        - mAP over the IoU range specified by `iou_range`
        - AP values at IoU thresholds specified by `iou_list`

        Args:
            classes: name of each class (index needs to correspond to
                predicted class indices!)
            iou_list: specific thresholds where ap is evaluated and saved
            iou_range: (start, stop, step) for mAP iou thresholds
            verbose: log time needed for evaluation

        Warning:
            In contrast to the original COCO implementation this class does
            *not* limit the predictions per class per image! The number of
            predictions is already limited before the matching to reduce
            compute beforehand and move it to one place.
        """
        self.verbose = verbose
        self.classes = classes

        iou_list = np.array(iou_list)
        _iou_range = np.linspace(
            iou_range[0],
            iou_range[1],
            int(np.round((iou_range[1] - iou_range[0]) / iou_range[2])) + 1,
            endpoint=True,
        )
        self.iou_thresholds = np.union1d(iou_list, _iou_range)
        self.iou_range = iou_range

        # get indices of iou values of ious range and ious list for later evaluation
        self.iou_list_idx = np.nonzero(iou_list[:, np.newaxis] == self.iou_thresholds[np.newaxis])[1]
        self.iou_range_idx = np.nonzero(_iou_range[:, np.newaxis] == self.iou_thresholds[np.newaxis])[1]

        assert (self.iou_thresholds[self.iou_list_idx] == iou_list).all()
        assert (self.iou_thresholds[self.iou_range_idx] == _iou_range).all()

        self.recall_thresholds = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True)

    def get_iou_thresholds(self) -> Sequence[float]:
        """
        Return IoU thresholds needed for this metric in an numpy array

        Returns:
            Sequence[float]: IoU thresholds [M], M is the number of thresholds
        """
        return self.iou_thresholds

    def compute(
        self,
        results_list: List[Dict[int, Dict[str, np.ndarray]]],
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """
        Compute AP metric similar to COCO implementation (no limit on
        detections per image in this class)

        Args:
            results_list: list with result s per image (in list) per cateory
                (dict). Inner Dict contains multiple results obtained
                by `AbstractEvalMatching`.

                ``dtMatches`` np.ndarray
                    matched detections [T, D], where T = number of thresholds,
                    D = number of detections

                ``gtMatches`` np.ndarray
                    matched ground truth boxes [T, G], where T = number of
                    thresholds, G = number of ground truth

                ``dtScores`` np.ndarray
                    prediction scores [D] detection scores

                ``gtIgnore`` np.ndarray
                    ground truth boxes which should be ignored [G] indicate
                    whether ground truth should be ignored

                ``dtIgnore`` np.ndarray
                    detections which should be ignored [T, D], indicate which
                    detections should be ignored

        Returns:
            Dict[str, float]: dictionary with AP metrics
            Dict[str, np.ndarray]: None
        """
        if self.verbose:
            logger.info("Start COCO metric computation...")
            tic = time.time()

        dataset_statistics = self.compute_statistics(results_list=results_list)
        if self.verbose:
            toc = time.time()
            logger.info(f"Statistics for COCO metrics finished (t={(toc - tic):0.2f}s).")

        results = {}
        results.update(self.compute_ap(dataset_statistics))

        if self.verbose:
            toc = time.time()
            logger.info(f"COCO metrics computed in t={(toc - tic):0.2f}s.")
        return results, None

    def compute_ap(self, dataset_statistics: dict) -> dict:
        """
        Compute AP metrics

        Args:
            dataset_statistics: computed statistics over dataset

                ``counts``: (int, int, int)
                    Number of thresholds, Number recall thresholds,
                    Number of classes, Number of max detection thresholds

                ``recall``: np.ndarray
                    Computed recall values
                    [num_iou_th, num_classes]

                ``precision``: np.ndarray
                    Precision values at specified recall thresholds
                    [num_iou_th, num_recall_th, num_classes]

                ``scores``: np.ndarray
                    Scores corresponding to specified recall thresholds
                    [num_iou_th, num_recall_th, num_classes]
        """
        results = {}
        if self.iou_range:  # mAP
            key = f"mAP_IoU_{self.iou_range[0]:.2f}_{self.iou_range[1]:.2f}_{self.iou_range[2]:.2f}"
            results[key] = self.select_ap(
                dataset_statistics,
                iou_idx=self.iou_range_idx,
            )

            for cls_idx, cls_str in enumerate(self.classes):  # per class results
                key = f"{cls_str}_" f"mAP_IoU_{self.iou_range[0]:.2f}_{self.iou_range[1]:.2f}_{self.iou_range[2]:.2f}"
                results[key] = self.select_ap(
                    dataset_statistics,
                    iou_idx=self.iou_range_idx,
                    cls_idx=cls_idx,
                )

        for idx in self.iou_list_idx:  # AP@IoU
            key = f"AP_IoU_{self.iou_thresholds[idx]:.2f}"
            results[key] = self.select_ap(dataset_statistics, iou_idx=[idx])

            for cls_idx, cls_str in enumerate(self.classes):  # per class results
                key = f"{cls_str}_AP_IoU_{self.iou_thresholds[idx]:.2f}"
                results[key] = self.select_ap(
                    dataset_statistics,
                    iou_idx=[idx],
                    cls_idx=cls_idx,
                )
        return results

    @staticmethod
    def select_ap(
        dataset_statistics: dict,
        iou_idx: Union[int, List[int]] = None,
        cls_idx: Union[int, Sequence[int]] = None,
    ) -> np.ndarray:
        """
        Compute average precision

        Args:
            dataset_statistics: computed statistics over dataset

                ``counts``: (int, int, int)
                    Number of thresholds, Number recall thresholds,
                    Number of classes, Number of max detection thresholds

                ``recall``: np.ndarray
                    Computed recall values
                    [num_iou_th, num_classes]

                ``precision``: np.ndarray
                    Precision values at specified recall thresholds
                    [num_iou_th, num_recall_th, num_classes]

                ``scores``: np.ndarray
                    Scores corresponding to specified recall thresholds
                    [num_iou_th, num_recall_th, num_classes]

            iou_idx: index of IoU values to select for evaluation
                (if None, all values are used)
            cls_idx: class indices to select, if None all classes
                will be selected

        Returns:
            np.ndarray: AP value
        """

        prec = dataset_statistics["precision"]
        if iou_idx is not None:
            prec = prec[iou_idx]
        if cls_idx is not None:
            prec = prec[..., cls_idx]
        return np.mean(prec)

    def compute_statistics(
        self, results_list: List[Dict[int, Dict[str, np.ndarray]]]
    ) -> Dict[str, Union[np.ndarray, List]]:
        """
        Compute statistics needed for metric computation
        Adapted from `https://github.com/cocodataset/cocoapi/blob/
        master/PythonAPI/pycocotools/cocoeval.py`

        Args:
            results_list: list with result s per image (in list) per cateory
                (dict). Inner Dict contains multiple results obtained
                by `AbstractEvalMatching`.

                ``dtMatches`` np.ndarray
                    matched detections [T, D], where T = number of thresholds,
                    D = number of detections

                ``gtMatches`` np.ndarray
                    matched ground truth boxes [T, G], where T = number of
                    thresholds, G = number of ground truth

                ``dtScores`` np.ndarray
                    prediction scores [D] detection scores

                ``gtIgnore`` np.ndarray
                    ground truth boxes which should be ignored [G] indicate
                    whether ground truth should be ignored

                ``dtIgnore`` np.ndarray
                    detections which should be ignored [T, D], indicate which
                    detections should be ignored

        Returns:
            dict: computed statistics over dataset
                ``counts`` List[int]
                    Number of thresholds, Number of recall thresholds,
                    Number of classes

                ``recall`` np.ndarray
                    Computed recall values
                    [num_iou_th, num_classes]

                ``precision`` np.ndarray
                    Precision values at specified recall thresholds
                    [num_iou_th, num_recall_th, num_classes]

                ``scores`` np.ndarray
                    Scores corresponding to specified recall thresholds
                    [num_iou_th, num_recall_th, num_classes]
        """
        num_iou_th = len(self.iou_thresholds)
        num_recall_th = len(self.recall_thresholds)
        num_classes = len(self.classes)

        # -1 for the precision of absent categories
        precision = -np.ones((num_iou_th, num_recall_th, num_classes))
        recall = -np.ones((num_iou_th, num_classes))
        scores = -np.ones((num_iou_th, num_recall_th, num_classes))

        for cls_idx, cls_i in enumerate(self.classes):  # for each class
            results = [r[cls_idx] for r in results_list if cls_idx in r]

            if len(results) == 0:
                logger.error(f"No results found for coco metric for class {cls_i} can not compute AP")
                continue

            dt_scores = np.concatenate([r["dtScores"] for r in results])
            # different sorting method generates slightly different results.
            # mergesort is used to be consistent as Matlab implementation.
            inds = np.argsort(-dt_scores, kind="mergesort")
            dt_scores_sorted = dt_scores[inds]

            # r['dtMatches'] [T, R], where R = sum(all detections)
            dt_matches = np.concatenate([r["dtMatches"] for r in results], axis=1)[:, inds]
            dt_ignores = np.concatenate([r["dtIgnore"] for r in results], axis=1)[:, inds]

            # case_ids = []
            # for r in results:
            #     case_ids.extend([r['case_id']] * min(len(r['dtMatches'][0]), maxDet))
            # case_ids_sorted = [case_ids[i] for i in inds]

            self.check_number_of_iou(dt_matches, dt_ignores)
            gt_ignore = np.concatenate([r["gtIgnore"] for r in results])
            num_gt = np.count_nonzero(gt_ignore == 0)  # number of ground truth boxes (non ignored)
            if num_gt == 0:
                logger.error(f"No gt found for coco metric for class {cls_i} can not compute AP")
                continue

            # ignore cases need to be handled differently for tp and fp
            tps = np.logical_and(dt_matches, np.logical_not(dt_ignores))
            fps = np.logical_and(np.logical_not(dt_matches), np.logical_not(dt_ignores))

            tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float32)
            fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float32)

            for th_ind, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):  # for each threshold th_ind
                tp, fp = np.array(tp), np.array(fp)
                r, p, s = compute_stats_single_threshold(tp, fp, dt_scores_sorted, self.recall_thresholds, num_gt)
                recall[th_ind, cls_idx] = r
                precision[th_ind, :, cls_idx] = p
                # corresponding score thresholds for recall steps
                scores[th_ind, :, cls_idx] = s

        return {
            "counts": [
                num_iou_th,
                num_recall_th,
                num_classes,
            ],  # [3]
            "recall": recall,  # [num_iou_th, num_classes]
            "precision": precision,  # [num_iou_th, num_recall_th, num_classes]
            "scores": scores,  # [num_iou_th, num_recall_th, num_classes]
        }


def compute_stats_single_threshold(
    tp: np.ndarray,
    fp: np.ndarray,
    dt_scores_sorted: np.ndarray,
    recall_thresholds: Sequence[float],
    num_gt: int,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute recall value, precision curve and scores thresholds
    Adapted from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

    Args:
        tp (np.ndarray): cumsum over true positives [R], R is the number of detections
        fp (np.ndarray): cumsum over false positives [R], R is the number of detections
        dt_scores_sorted (np.ndarray): sorted (descending) scores [R], R is the number of detections
        recall_thresholds (Sequence[float]): recall thresholds which should be evaluated
        num_gt (int): number of ground truth bounding boxes (excluding boxes which are ignored)

    Returns:
        float: overall recall for given IoU value
        np.ndarray: precision values at defined recall values
            [RTH], where RTH is the number of recall thresholds
        np.ndarray: prediction scores corresponding to recall values
            [RTH], where RTH is the number of recall thresholds
    """
    num_recall_th = len(recall_thresholds)

    rc = tp / num_gt
    # np.spacing(1) is the smallest representable epsilon with float
    pr = tp / (fp + tp + np.spacing(1))

    if len(tp):
        recall = rc[-1]
    else:
        # no prediction
        recall = 0

    # array where precision values nearest to given recall th are saved
    precision = np.zeros((num_recall_th,))
    # save scores for corresponding recall value in here
    th_scores = np.zeros((num_recall_th,))
    # numpy is slow without cython optimization for accessing elements
    # use python array gets significant speed improvement
    pr = pr.tolist()
    precision = precision.tolist()

    # smooth precision curve (create box shape)
    for i in range(len(tp) - 1, 0, -1):
        if pr[i] > pr[i - 1]:
            pr[i - 1] = pr[i]

    # get indices to nearest given recall threshold (nn interpolation!)
    inds = np.searchsorted(rc, recall_thresholds, side="left")
    try:
        for save_idx, array_index in enumerate(inds):
            precision[save_idx] = pr[array_index]
            th_scores[save_idx] = dt_scores_sorted[array_index]
    except BaseException:
        pass

    return recall, np.array(precision), np.array(th_scores)
