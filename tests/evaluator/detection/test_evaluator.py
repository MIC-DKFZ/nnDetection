import json

import numpy as np
import pytest
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pytest_mock import MockerFixture

import nndet.core.ops_np as ops_np
from nndet.evaluator.det import BoxEvaluator, DetectionEvaluator
from nndet.evaluator.detection.coco import COCOMetric


class DummyMetric:
    def __init__(self, ious=(0.1, 0.2)):
        self.ious = ious

    def get_iou_thresholds(self):
        return self.ious


@pytest.fixture
def evaluator():
    return DetectionEvaluator([DummyMetric()], iou_fn=ops_np.box_iou_np)


class TestDetectionEvaluator:
    def test_init(self):
        self.evaluator = DetectionEvaluator(
            [
                DummyMetric((0.1, 0.2)),
                DummyMetric((0.3, 0.4)),
            ],
            iou_fn=ops_np.box_iou_np,
        )
        assert all([a == b for a, b in zip(self.evaluator.iou_thresholds, [0.1, 0.2, 0.3, 0.4])])
        assert all([a == b for a, b in zip(self.evaluator.iou_mapping, [[0, 1], [2, 3]])])
        assert "" in self.evaluator.criterion_ranges.keys()
        assert self.evaluator.criterion_ranges[""][0] == np.NINF
        assert self.evaluator.criterion_ranges[""][1] == np.inf

    def test_run_online_evaluation(self, mocker: MockerFixture, evaluator):
        _pred_boxes = np.array([[0]])[None]
        _pred_classes = np.array([[1]])[None]
        _pred_scores = np.array([[2]])[None]
        _gt_boxes = np.array([[3]])[None]
        _gt_classes = np.array([[4]])[None]
        # Use pred and gt class here
        mock_matches = {
            1: {"dtMatches": np.array([[1, 1]]), "dtIgnore": np.array([[0, 0]])},
            4: {"dtMatches": np.array([[1, 1]]), "dtIgnore": np.array([[0, 0]])},
        }
        evaluator.match_fn = mocker.MagicMock(return_value=[mock_matches])
        evaluator.box_criterion = mocker.MagicMock(return_value=[0])
        res = evaluator.run_online_evaluation(
            _pred_boxes,
            _pred_classes,
            _pred_scores,
            _gt_classes,
            _gt_classes,
        )

        assert not res
        assert all(a == b for a, b in zip([""], evaluator.results_dict.keys()))
        assert len(evaluator.results_dict[""]) == 1
        assert all(mock_matches[key] == value for key, value in evaluator.results_dict[""][0].items())

    def test_find_dt_ignores(self, mocker: MockerFixture, evaluator):
        _pred_boxes = np.array([[0]])[None]
        _pred_classes = np.array([[1]])[None]
        _pred_scores = np.array([[2]])[None]
        _gt_boxes = np.array([[3]])[None]
        _gt_classes = np.array([[1]])[None]
        _gt_ignore = np.array([[0]])[None]
        # Use pred and gt class here (1), has to be unmatched so dtMatch 0
        mock_matches = {1: {"dtMatches": np.array([[0]]), "dtIgnore": np.array([[0]])}}
        # Match should be ignored as criterion returns 2 but bounds are (0, 1)
        evaluator.box_criterion = mocker.MagicMock(return_value=[2])
        evaluator.criterion_ranges[""] = (0, 1)
        # List[Dict[class, Dict]]
        matches_with_ignores = evaluator.find_dt_ignores(
            results_key="",
            matches=[mock_matches],
            iou_thresholds=[1],
            pred_boxes=_pred_boxes,
            pred_classes=_pred_classes,
            pred_scores=_pred_scores,
            gt_boxes=_gt_boxes,
            gt_classes=_gt_classes,
            gt_ignore=_gt_ignore,
        )

        assert len(matches_with_ignores) == 1
        res = matches_with_ignores[0]
        assert all(res[c]["dtIgnore"][i] == 1 for c in res.keys() for i in range(len(res[c]["dtIgnore"])))

    def test_finish_online_evaluation(self, mocker: MockerFixture, evaluator):
        evaluator.iou_filter = mocker.Mock(return_value=0)
        metric0 = mocker.Mock(return_value=({"score0": 0}, {"curve0": 1}))
        metric1 = mocker.Mock(return_value=({"score1": 2}, {"curve1": 3}))

        evaluator.metrics = [metric0, metric1]
        evaluator.results_dict = {"": [None, None]}
        evaluator.iou_mapping = [[0], [1]]
        metric_scores, metric_curves = evaluator.finish_online_evaluation()

        assert metric_curves == {"curve0": 1, "curve1": 3, "criterion": (np.NINF, np.inf)}
        assert metric_scores == {"score0": 0, "score1": 2}
        metric0.assert_called_with([0, 0], tag="")
        metric1.assert_called_with([0, 0], tag="")

    def test_iou_filter(self, evaluator):
        image_dict = {
            0: {"dtMatches": np.array([0, 1, 2, 3])},
            1: {"dtMatches": np.array([2, 3, 0, 1])},
        }
        res = evaluator.iou_filter(image_dict, iou_idx=[0, 1], filter_keys=["dtMatches"])
        assert np.isclose(res[0]["dtMatches"], [0, 1]).all()
        assert np.isclose(res[1]["dtMatches"], [2, 3]).all()

    def test_compute_ap_whole_dataset(self):
        detection_file = "new_results_no_crowd_fixed_area.json"
        annotation_file = "new_ann_no_crowd_fixed_area.json"
        coco_eval = evaluate_with_pycoco(detection_file, annotation_file)
        with open(detection_file) as f:
            fake_detections = json.load(f)
        with open(annotation_file) as g:
            gt_coco = json.load(g)

        detections_by_image, annotations_by_image = convert_to_nndet_format(fake_detections, gt_coco)
        # Create Evaluator
        classes = [cat["name"] for cat in gt_coco["categories"]]
        coco = COCOMetric(
            classes, iou_list=(0.5, 0.75), iou_range=(0.5, 0.95, 0.05), max_detection=(1, 10, 100), verbose=True
        )
        ranges = {"small": (0**2, 32**2), "medium": (32**2, 96**2), "large": (96**2, 1e5**2)}
        evaluator = BoxEvaluator(
            [coco], iou_fn=ops_np.box_iou_np, box_criterion=ops_np.box_area_np, criterion_ranges=ranges
        )

        for id, detection in detections_by_image.items():
            annotation = annotations_by_image[id]
            evaluator.run_online_evaluation(
                [np.array(detection["box"])],
                [np.array(detection["class"])],
                [np.array(detection["score"])],
                [np.array(annotation["box"])],
                [np.array(annotation["class"])],
                [np.array(annotation["gt_ignore"])],
                case_id=id,
            )

        score, _ = evaluator.finish_online_evaluation()
        imp_scores = {key: np.round(value, 3) for key, value in score.items() if key[:3] == "mAP" or key[:2] == "AP"}
        coco_scores = np.round(coco_eval.stats, 3)
        # Fails for rounding to 4 even for the first one
        assert coco_scores[0] == imp_scores["mAP_IoU_0.50_0.95_0.05_MaxDet_100"]
        assert coco_scores[1] == imp_scores["AP_IoU_0.50_MaxDet_100"]
        assert coco_scores[2] == imp_scores["AP_IoU_0.75_MaxDet_100"]
        assert coco_scores[3] == imp_scores["mAP_small_IoU_0.50_0.95_0.05_MaxDet_100"]
        assert coco_scores[4] == imp_scores["mAP_medium_IoU_0.50_0.95_0.05_MaxDet_100"]
        assert coco_scores[5] == imp_scores["mAP_large_IoU_0.50_0.95_0.05_MaxDet_100"]
        assert score is not None
        assert imp_scores is not None


def convert_to_nndet_format(fake_detections, gt_coco):
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


def evaluate_with_pycoco(detection_file, annotation_file):
    cocoGt = COCO(annotation_file)
    cocoDt = cocoGt.loadRes(detection_file)

    imgIds = sorted(cocoGt.getImgIds())
    # running evaluation
    annType = "bbox"
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    evaluation = cocoEval.evaluate()
    accumulation = cocoEval.accumulate()
    summary = cocoEval.summarize()
    return cocoEval
