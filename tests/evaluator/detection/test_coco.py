import json
import math
from pathlib import Path

import numpy as np
import pytest
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pytest_mock import MockerFixture

from nndet.core import ops_np
from nndet.evaluator.det import BoxEvaluator
from nndet.evaluator.detection.coco import COCOMetric, compute_stats_single_threshold


@pytest.fixture
def metric():
    return COCOMetric(
        classes=["benign", "malignant"],
        iou_list=(0.1, 0.3),
        iou_range=(0.1, 0.2, 0.1),
        max_detection=(1, 10),
    )


@pytest.fixture
def prediction_path():
    return Path("predictions_coco_no_crowd_fixed_area.json").resolve().as_posix()


@pytest.fixture
def annotation_path():
    return Path("annotations_coco_no_crowd_fixed_area.json").resolve().as_posix()


@pytest.fixture
def evaluate_with_pycoco(prediction_path, annotation_path):
    cocoGt = COCO(annotation_path)
    cocoDt = cocoGt.loadRes(prediction_path)

    imgIds = sorted(cocoGt.getImgIds())
    # running evaluation
    annType = "bbox"
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    evaluation = cocoEval.evaluate()
    accumulation = cocoEval.accumulate()
    summary = cocoEval.summarize()
    return cocoEval


@pytest.fixture
def load_annotation_and_convert_to_nndet_format(prediction_path, annotation_path):
    with open(prediction_path) as f:
        fake_detections = json.load(f)
    with open(annotation_path) as g:
        gt_coco = json.load(g)
    annotations = gt_coco["annotations"]
    # Create Category map
    category_map = {}
    for i, category_dict in enumerate(gt_coco["categories"]):
        category_map[category_dict["id"]] = i

    # Need: pred_boxes, classes, scores, gt_boxes, gt_classes, gt_ignore
    # Group by image and predict
    detections_by_image = {}
    for detection in fake_detections:
        id = detection["image_id"]
        if id not in detections_by_image:
            detections_by_image[id] = {"score": [], "box": [], "class": []}
        detections_by_image[id]["score"].append(detection["score"])
        box = detection["bbox"]
        out_box = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]])
        detections_by_image[id]["box"].append(out_box)
        detections_by_image[id]["class"].append(category_map[detection["category_id"]])
    annotations_by_image = {}
    for annotation in annotations:
        id = annotation["image_id"]
        if id not in annotations_by_image:
            annotations_by_image[id] = {"box": [], "class": [], "gt_ignore": []}
        box = annotation["bbox"]
        out_box = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]])
        annotations_by_image[id]["box"].append(out_box)
        annotations_by_image[id]["class"].append(category_map[annotation["category_id"]])
        annotations_by_image[id]["gt_ignore"].append(annotation["iscrowd"])
    return detections_by_image, annotations_by_image


@pytest.fixture
def evaluator_coco(annotation_path):
    with open(annotation_path) as g:
        gt_coco = json.load(g)
    classes = [cat["name"] for cat in gt_coco["categories"]]
    coco = COCOMetric(
        classes, iou_list=(0.5, 0.75), iou_range=(0.5, 0.95, 0.05), max_detection=(1, 10, 100), verbose=True
    )
    ranges = {"small": (0**2, 32**2), "medium": (32**2, 96**2), "large": (96**2, 1e5**2)}
    evaluator = BoxEvaluator(
        [coco], iou_fn=ops_np.box_iou_np, box_criterion=ops_np.box_area_np, criterion_ranges=ranges
    )
    return evaluator


class TestCOCOMetric:
    def test_get_iou_thresholds(self):
        metric = COCOMetric(
            classes=["benign", "malignant"],
            iou_list=(0.1, 0.2, 0.3),
            iou_range=(0.1, 0.2, 0.05),
            max_detection=(1, 5, 100),
        )
        assert np.isclose(metric.get_iou_thresholds(), [0.1, 0.15, 0.2, 0.3]).all()

    def test_compute(self, mocker: MockerFixture, metric):
        mocker.patch("nndet.evaluator.detection.coco.COCOMetric.select_ap", return_value=1)
        mocker.patch(
            "nndet.evaluator.detection.coco.COCOMetric.compute_statistics",
            return_value={"stats": 0},
        )

        score, curve = metric([0, 1, 2])
        assert curve is None
        assert score["mAP_IoU_0.10_0.20_0.10_MaxDet_10"] == 1
        assert score["AP_IoU_0.10_MaxDet_10"] == 1
        assert score["AP_IoU_0.30_MaxDet_10"] == 1

    def test_select_ap(self, metric):
        stats = {"precision": np.array([[0.0, 0.5, 1.0], [1.0, 1.0, 1.0], [1.0, 0.5, 3.0]])[:, :, None, None]}
        ap0 = metric.select_ap(stats, [0])
        ap = metric.select_ap(stats)
        assert math.isclose(ap0, 0.5)
        assert math.isclose(ap, 1.0)

    def test_compute_statistics(self, mocker: MockerFixture, metric):
        metric.iou_thresholds = np.array([[0.1]])
        metric.recall_thresholds = np.array([0.1, 0.2])
        results_list = []
        results_list += [
            {
                0: {
                    "dtMatches": np.array([[0, 0]]),
                    "dtIgnore": np.array([[0, 0]]),
                    "dtScores": np.array([0, 0]),
                    "gtIgnore": np.array([0]),
                },
                1: {
                    "dtMatches": np.array([[0, 0]]),
                    "dtIgnore": np.array([[0, 0]]),
                    "dtScores": np.array([0, 0]),
                    "gtIgnore": np.array([0]),
                },
            }
        ] * 3
        mocker.patch(
            "nndet.evaluator.detection.coco.compute_stats_single_threshold",
            return_value=(1, [2, 3], [4, 5]),
        )

        stats = metric.compute_statistics(results_list)
        assert np.isclose(stats["counts"], [1, 2, 2, 2]).all()
        assert np.isclose(stats["recall"], [[[1.0, 1.0], [1.0, 1.0]]]).all()
        assert np.isclose(stats["precision"], [[[[2.0, 2.0], [2.0, 2.0]], [[3.0, 3.0], [3.0, 3.0]]]]).all()
        assert np.isclose(stats["scores"], [[[[4.0, 4.0], [4.0, 4.0]], [[5.0, 5.0], [5.0, 5.0]]]]).all()

    def test_compute_stats_single_threshold(self):
        tp = np.array([1, 2, 3, 4, 5, 6])
        fp = np.array([0, 1, 1, 2, 3, 3])
        dt_scores_sorted = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        recall_thresholds = np.array([0.5, 0.7, 0.9])
        num_gt = 2
        rc, prec, ths = compute_stats_single_threshold(tp, fp, dt_scores_sorted, recall_thresholds, num_gt)

        assert math.isclose(rc, 3.0)  # used different num_gt so number are nice
        assert np.isclose(prec, [1.0, 0.75, 0.75]).all()
        assert np.isclose(ths, [0.9, 0.8, 0.8]).all()

    def test_compute_ap_whole_dataset(
        self, evaluate_with_pycoco, load_annotation_and_convert_to_nndet_format, evaluator_coco
    ):

        coco_eval = evaluate_with_pycoco
        detections_by_image, annotations_by_image = load_annotation_and_convert_to_nndet_format

        for id, detection in detections_by_image.items():
            annotation = annotations_by_image[id]
            evaluator_coco.run_online_evaluation(
                [np.array(detection["box"])],
                [np.array(detection["class"])],
                [np.array(detection["score"])],
                [np.array(annotation["box"])],
                [np.array(annotation["class"])],
                [np.array(annotation["gt_ignore"])],
                case_id=id,
            )

        score, _ = evaluator_coco.finish_online_evaluation()
        coco_scores = coco_eval.stats
        assert math.isclose(coco_scores[0], score["mAP_IoU_0.50_0.95_0.05_MaxDet_100"])
        assert math.isclose(coco_scores[1], score["AP_IoU_0.50_MaxDet_100"])
        assert math.isclose(coco_scores[2], score["AP_IoU_0.75_MaxDet_100"])
        assert math.isclose(coco_scores[3], score["mAP_small_IoU_0.50_0.95_0.05_MaxDet_100"])
        assert math.isclose(coco_scores[4], score["mAP_medium_IoU_0.50_0.95_0.05_MaxDet_100"])
        assert math.isclose(coco_scores[5], score["mAP_large_IoU_0.50_0.95_0.05_MaxDet_100"])
