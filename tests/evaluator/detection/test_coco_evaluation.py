import io
import json
import math
import tempfile
import zipfile
from copy import deepcopy

import numpy as np
import pytest
import requests
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from nndet.core import ops_np
from nndet.evaluator.det import BoxEvaluator
from nndet.evaluator.detection import COCOMetric


def filter_dataset(predictions, annotations):
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


@pytest.fixture
def download_data():

    r = requests.get("http://images.cocodataset.org/annotations/annotations_trainval2014.zip", stream=True)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    with z.open("annotations/instances_val2014.json") as annotation_zip:
        annotation_dict = json.load(annotation_zip)
    predictions = requests.get(
        "https://raw.githubusercontent.com/cocodataset/cocoapi/master/results"
        "/instances_val2014_fakebbox100_results.json"
    ).json()

    filtered_predictions, filtered_annotations = filter_dataset(predictions, annotation_dict)
    annotation_path = "temp_annotations.json"
    prediction_path = "temp_predictions.json"
    with open(annotation_path, "w") as f:
        json.dump(filtered_annotations, f)
    with open(prediction_path, "w") as g:
        json.dump(filtered_predictions, g)

    # COCO Eval
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

    # Convert to nndet format
    annotations = filtered_annotations["annotations"]
    # Create Category map
    category_map = {}
    for i, category_dict in enumerate(filtered_annotations["categories"]):
        category_map[category_dict["id"]] = i

    # Need: pred_boxes, classes, scores, gt_boxes, gt_classes, gt_ignore
    # Group by image and predict
    detections_by_image = {}
    for detection in filtered_predictions:
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

    # Create COCO Metric
    classes = [cat["name"] for cat in filtered_annotations["categories"]]
    coco = COCOMetric(
        classes, iou_list=(0.5, 0.75), iou_range=(0.5, 0.95, 0.05), max_detection=(1, 10, 100), verbose=True
    )
    ranges = {"small": (0**2, 32**2), "medium": (32**2, 96**2), "large": (96**2, 1e5**2)}
    evaluator = BoxEvaluator(
        [coco], iou_fn=ops_np.box_iou_np, box_criterion=ops_np.box_area_np, criterion_ranges=ranges
    )

    return cocoEval, detections_by_image, annotations_by_image, evaluator


class TestEvaluatorwithCOCOMetric:
    def test_compute_ap_whole_dataset(
        self,
        download_data,
    ):
        coco_eval, detections_by_image, annotations_by_image, evaluator_coco = download_data

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
