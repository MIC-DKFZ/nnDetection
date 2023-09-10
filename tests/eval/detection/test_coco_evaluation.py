import io
import json
import math
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pycocotools
import pytest
import requests
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from nndet.core import ops_np
from nndet.eval.det import BoxEvaluator, CocoAPMetric
from nndet.eval.matching import EvalMatchingPerElementGreedyScoreNP


def filter_dataset(predictions: Dict, annotations: Dict) -> Tuple[Dict, Dict]:
    imageIds = []
    for result in predictions:
        imageIds.append(result["image_id"])
    resultIds = np.unique(np.array(imageIds))
    new_anns = []
    new_ids = []
    drop_imgs = []
    for an in annotations["annotations"]:
        img_id = an["image_id"]
        if an["iscrowd"]:
            if img_id in resultIds:
                drop_imgs.append(img_id)
            continue
        if img_id in resultIds:
            an.pop("segmentation")
            an["area"] = an["bbox"][2] * an["bbox"][3]
            new_anns.append(an)
            new_ids.append(img_id)

    # Filter out the is crowd images
    for i, ann in enumerate(new_anns):
        if ann["image_id"] in drop_imgs:
            new_anns.pop(i)
    for i, res in enumerate(predictions):
        if res["image_id"] in drop_imgs:
            predictions.pop(i)

    new_images = []
    for image in annotations["images"]:
        if image["id"] in resultIds:
            new_images.append(image)

    # create new dict
    new_an_dict = deepcopy(annotations)
    new_an_dict["annotations"] = new_anns
    new_an_dict["images"] = new_images

    return predictions, new_an_dict


def convert_to_nndet_format(predictions_in: Dict, annotations_in: Dict) -> Tuple[Dict, Dict]:
    annotations = annotations_in["annotations"]
    # Create Category map
    category_map = {}
    for i, category_dict in enumerate(annotations_in["categories"]):
        category_map[category_dict["id"]] = i

    # Need: pred_boxes, classes, scores, gt_boxes, gt_classes, gt_ignore
    # Group by image and predict
    detections_by_image = {}
    for detection in predictions_in:
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
def download_data():
    cache_dir = Path(__file__).parent / "coco_cache_tests"
    annotation_path = cache_dir / "temp_annotations.json"
    prediction_path = cache_dir / "temp_predictions.json"
    if not (annotation_path.is_file() and prediction_path.is_file()):
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Download data and load into arrays
        r = requests.get(
            "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
            stream=True,
        )
        z = zipfile.ZipFile(io.BytesIO(r.content))
        with z.open("annotations/instances_val2014.json") as annotation_zip:
            annotation_dict = json.load(annotation_zip)
        predictions = requests.get(
            "https://raw.githubusercontent.com/cocodataset/cocoapi/master/results"
            "/instances_val2014_fakebbox100_results.json"
        ).json()

        # Filter out iscrowd instances and fix the wrong area entries
        filtered_predictions, filtered_annotations = filter_dataset(predictions, annotation_dict)

        with open(annotation_path, "w") as f:
            json.dump(filtered_annotations, f)
        with open(prediction_path, "w") as g:
            json.dump(filtered_predictions, g)
    else:
        with open(annotation_path, "r") as f:
            filtered_annotations = json.load(f)
        with open(prediction_path, "r") as g:
            filtered_predictions = json.load(g)

    # patch np.float
    pycocotools.cocoeval.np.float = float

    # COCO Eval
    cocoGt = COCO(str(annotation_path))
    cocoDt = cocoGt.loadRes(str(prediction_path))
    imgIds = sorted(cocoGt.getImgIds())
    # running evaluation
    annType = "bbox"
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    evaluation = cocoEval.evaluate()
    accumulation = cocoEval.accumulate()
    summary = cocoEval.summarize()

    # # After evaluation with pycoco, delete temp files
    # os.remove(annotation_path)
    # os.remove(prediction_path)

    # Convert to nndet format
    detections_by_image, annotations_by_image = convert_to_nndet_format(filtered_predictions, filtered_annotations)

    # Create COCO Metric
    classes = [cat["name"] for cat in filtered_annotations["categories"]]
    coco = CocoAPMetric(classes, iou_list=(0.5, 0.75), iou_range=(0.5, 0.95, 0.05), verbose=True)
    ranges = {
        "small": (0**2, 32**2),
        "medium": (32**2, 96**2),
        "large": (96**2, 1e5**2),
    }
    matching = EvalMatchingPerElementGreedyScoreNP(
        iou_fn=ops_np.box_iou_np,
        max_detections=100,
    )
    evaluator = BoxEvaluator(
        metrics=[coco],
        matching=matching,
        criterion=ops_np.box_area_np,
        criterion_ranges=ranges,
    )
    return cocoEval, detections_by_image, annotations_by_image, evaluator


class TestEvaluatorwithCOCOMetric:
    def test_compute_ap_whole_dataset(
        self,
        download_data,
    ):
        (
            coco_eval,
            detections_by_image,
            annotations_by_image,
            evaluator_coco,
        ) = download_data

        for id, detection in detections_by_image.items():
            annotation = annotations_by_image[id]
            evaluator_coco.run_online_evaluation(
                [np.array(detection["box"])],
                [np.array(detection["class"])],
                [np.array(detection["score"])],
                [np.array(annotation["box"])],
                [np.array(annotation["class"])],
                [np.array(annotation["gt_ignore"])],
                case_ids=[id],
            )

        score, _ = evaluator_coco.finish_online_evaluation()
        coco_scores = coco_eval.stats
        assert math.isclose(coco_scores[0], score["mAP_IoU_0.50_0.95_0.05"])
        assert math.isclose(coco_scores[1], score["AP_IoU_0.50"])
        assert math.isclose(coco_scores[2], score["AP_IoU_0.75"])
        assert math.isclose(coco_scores[3], score["mAP_small_IoU_0.50_0.95_0.05"])
        assert math.isclose(coco_scores[4], score["mAP_medium_IoU_0.50_0.95_0.05"])
        assert math.isclose(coco_scores[5], score["mAP_large_IoU_0.50_0.95_0.05"])
