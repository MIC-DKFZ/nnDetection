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

"""
This is prototype code ... Use at your own risk
This was initially part of a notebook but I needed to move it into
this scriptish functions to run it in my default pipeline
"""

import pickle
from itertools import product
from pathlib import Path
from typing import Sequence, Optional, Tuple
from collections import defaultdict
from loguru import logger

import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
from sklearn.metrics import confusion_matrix
from torch import Tensor
import SimpleITK as sitk

from nndet.core.boxes import box_iou_np, box_size_np
from nndet.io.load import load_pickle, save_json
from nndet.utils.info import maybe_verbose_iterable, experimental, deprecate


def collect_overview(prediction_dir: Path, gt_dir: Path,
                     iou: float, score: float,
                     max_num_fp_per_image: int = 5,
                     top_n: int = 10,
                     ):
    results = defaultdict(dict)
    
    for f in prediction_dir.glob("*_boxes.pkl"):
        case_id = f.stem.rsplit('_', 1)[0]

        gt_data = np.load(str(gt_dir / f"{case_id}_boxes_gt.npz"), allow_pickle=True)
        gt_boxes = gt_data["boxes"]
        gt_classes = gt_data["classes"]
        gt_ignore = [np.zeros(gt_boxes_img.shape[0]).reshape(-1, 1) for gt_boxes_img in [gt_boxes]]

        case_result = load_pickle(f)
        pred_boxes = case_result["pred_boxes"]
        pred_scores = case_result["pred_scores"]
        pred_labels = case_result["pred_labels"]
        keep = pred_scores > score
        pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]

        # if "properties" in case_data:
        #     results[case_id]["orig_spacing"] = case_data["properties"]["original_spacing"]
        #     results[case_id]["crop_shape"] = [c[1] for c in case_data["properties"]["crop_bbox"]]
        # else:
        #     results[case_id]["orig_spacing"] = None
        #     results[case_id]["crop_shape"] = None
        results[case_id]["num_gt"] = len(gt_classes)

        # computation stats here
        if gt_boxes.size == 0:
            idx = np.argsort(pred_scores)[::-1][:5]
            results[case_id]["fp_score"] = pred_scores[idx]
            results[case_id]["fp_label"] = pred_labels[idx]
            results[case_id]["fp_true_label"] = (np.ones(len(pred_labels)) * -1)
            results[case_id]["fp_type"] = ["fp_iou"] * len(pred_labels)
            results[case_id]["num_fn"] = 0
        elif pred_boxes.size == 0:
            results[case_id]["num_fn"] = len(gt_classes)
            results[case_id]["fn_boxes"] = gt_boxes
        else:
            match_quality_matrix = box_iou_np(gt_boxes, pred_boxes)
            matched_idxs = np.argmax(match_quality_matrix, axis=0)
            matched_vals = np.max(match_quality_matrix, axis=0)
            matched_idxs[matched_vals < iou] = -1

            matched_gt_boxes_per_image = gt_boxes[matched_idxs.clip(min=0)]
            target_labels = gt_classes[matched_idxs.clip(min=0)]
            target_labels[matched_idxs == -1] = -1

            # True positive analysis
            tp_keep = target_labels == pred_labels
            tp_boxes, tp_scores, tp_labels = pred_boxes[tp_keep], pred_scores[tp_keep], pred_labels[tp_keep]

            keep_high = tp_scores > 0.5
            tp_high_boxes, tp_high_scores, tp_high_labels = tp_boxes[keep_high], tp_scores[keep_high], tp_labels[
                keep_high]
            keep_low = tp_scores < 0.5
            tp_low_boxes, tp_low_scores, tp_low_labels = tp_boxes[keep_low], tp_scores[keep_low], tp_labels[keep_low]

            high_idx = np.argsort(tp_high_scores)[::-1][:3]
            low_idx = np.argsort(tp_low_scores)[:3]
            results[case_id]["iou_tp"] = int(tp_keep.sum())
            results[case_id]["tp_high_boxes"] = tp_high_boxes[high_idx]
            results[case_id]["tp_high_score"] = tp_high_scores[high_idx]
            results[case_id]["tp_high_label"] = tp_high_labels[high_idx]
            results[case_id]["tp_iou"] = matched_vals[tp_keep]

            if tp_low_boxes.size > 0:
                results[case_id]["tp_low_boxes"] = tp_low_boxes[low_idx]
                results[case_id]["tp_low_score"] = tp_low_scores[low_idx]
                results[case_id]["tp_low_label"] = tp_low_labels[low_idx]

            # False Positive Analysis
            fp_keep = (pred_labels != target_labels) * (pred_labels != -1)
            fp_boxes, fp_scores, fp_labels, fp_target_labels = pred_boxes[fp_keep], pred_scores[fp_keep], pred_labels[
                fp_keep], target_labels[fp_keep]
            idx = np.argsort(fp_scores)[::-1][:max_num_fp_per_image]
            # results[case_id]["fp_box"] = fp_boxes[idx]
            results[case_id]["fp_score"] = fp_scores[idx]
            results[case_id]["fp_label"] = fp_labels[idx]
            results[case_id]["fp_true_label"] = fp_target_labels[idx]
            results[case_id]["fp_type"] = ["fp_iou" if tl == -1 else "fp_cls" for tl in fp_target_labels]

            # Misc
            unmatched_gt = (match_quality_matrix.max(axis=1) < iou)
            false_negatives = unmatched_gt.sum()
            results[case_id]["fn_boxes"] = gt_boxes[unmatched_gt]
            results[case_id]["num_fn"] = false_negatives

    df = pd.DataFrame.from_dict(results, orient='index')
    df = df.sort_index()

    analysis_ids = {}
    if "fp_score" in list(df.columns):
        tmp = df["fp_score"].apply(lambda x: np.max(x) if np.any(x) else 0).nlargest(top_n)
        analysis_ids["top_scoring_fp"] = tmp.index.values.tolist()
        tmp = df["fp_score"].apply(
            lambda x: len(x) if isinstance(x, Sequence) or isinstance(x, np.ndarray) else 0).nlargest(top_n)
        analysis_ids["top_num_fp"] = tmp.index.values.tolist()
    if "fp_score" in list(df.columns):
        tmp = df["num_fn"].nlargest(top_n)
        analysis_ids["top_num_fn"] = tmp.index.values.tolist()
    return df, analysis_ids


def collect_score_iou(prediction_dir: Path, gt_dir: Path, iou: float, score: float):
    all_pred = []
    all_target = []

    all_pred_ious = []
    all_pred_scores = []

    for f in prediction_dir.glob("*_boxes.pkl"):
        case_id = f.stem.rsplit('_', 1)[0]

        gt_data = np.load(str(gt_dir / f"{case_id}_boxes_gt.npz"), allow_pickle=True)
        gt_boxes = gt_data["boxes"]
        gt_classes = gt_data["classes"]
        gt_ignore = [np.zeros(gt_boxes_img.shape[0]).reshape(-1, 1) for gt_boxes_img in [gt_boxes]]

        case_result = load_pickle(f)
        pred_boxes = case_result["pred_boxes"]
        pred_scores = case_result["pred_scores"]
        pred_labels = case_result["pred_labels"]

        keep = pred_scores > score
        pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]

        # computation starts here
        if gt_boxes.size == 0:
            all_pred.append(pred_labels)
            all_target.append(np.ones(len(pred_labels)) * -1)

            all_pred_ious.append(np.zeros(len(pred_labels)))
            all_pred_scores.append(pred_scores)
        elif pred_boxes.size == 0:
            all_pred.append(np.ones(len(gt_classes)) * -1)
            all_target.append(gt_classes)
        else:
            match_quality_matrix = box_iou_np(gt_boxes, pred_boxes)

            matched_idxs = np.argmax(match_quality_matrix, axis=0)
            matched_vals = np.max(match_quality_matrix, axis=0)
            matched_idxs[matched_vals < iou] = -1

            matched_gt_boxes_per_image = gt_boxes[matched_idxs.clip(min=0)]
            target_labels = gt_classes[matched_idxs.clip(min=0)]
            target_labels[matched_idxs == -1] = -1

            all_pred.append(pred_labels)
            all_target.append(target_labels)

            all_pred_ious.append(matched_vals)
            all_pred_scores.append(pred_scores)

            false_negatives = (match_quality_matrix.max(axis=1) < iou).sum()
            if false_negatives > 0:  # false negatives
                all_pred.append(np.ones(false_negatives) * -1)
                all_target.append(np.zeros(false_negatives))

    return all_pred, all_target, all_pred_ious, all_pred_scores


def plot_confusion_matrix(all_pred, all_target, iou: float, score:float):
    if len(all_pred) > 0 and len(all_target) > 0:
        cm = confusion_matrix(np.concatenate(all_target), np.concatenate(all_pred))
        plt.figure()
        ax = sns.heatmap(cm, annot=True, cbar=False)
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Ground Truth")
        ax.set_title(f"Confusion Matrix IoU {iou} and Score Threshold {score}")
        return ax
    else:
        return None


def plot_joint_iou_score(all_pred_ious, all_pred_scores):
    if isinstance(all_pred_ious, Sequence):
        if len(all_pred_ious) == 0:
            return None
        all_pred_ious = np.concatenate(all_pred_ious)
    if isinstance(all_pred_scores, Sequence):
        if len(all_pred_scores) == 0:
            return None
        all_pred_scores = np.concatenate(all_pred_scores)
    plt.figure()
    f = sns.jointplot(x=all_pred_ious, y=all_pred_scores,
                      xlim=(-0.01, 1.01), ylim=(-0.01, 1.01), marginal_kws={"bins": 10},
                      kind='reg', scatter=True,
                      )
    plt.plot([0, 1], [0, 1], 'g')
    f.set_axis_labels("IoU", "Predicted Score")
    f.ax_joint.axvline(x=0.1, c='r')
    f.ax_joint.axvline(x=0.5, c='r')
    f.fig.subplots_adjust(top=0.9)
    f.fig.suptitle('Class independent predicted score over IoU plot', fontsize=16)
    return f


def collect_boxes(prediction_dir: Path, gt_dir: Path, iou:float, score: float):
    all_pred = []
    all_target = []
    all_boxes = []

    i = 0
    for f in prediction_dir.glob("*_boxes.pkl"):
        case_id = f.stem.rsplit('_', 1)[0]

        gt_data = np.load(str(gt_dir / f"{case_id}_boxes_gt.npz"), allow_pickle=True)
        gt_boxes = gt_data["boxes"]
        gt_classes = gt_data["classes"]
        gt_ignore = [np.zeros(gt_boxes_img.shape[0]).reshape(-1, 1) for gt_boxes_img in [gt_boxes]]
        
        case_result = load_pickle(f)
        pred_boxes = case_result["pred_boxes"]
        pred_scores = case_result["pred_scores"]
        pred_labels = case_result["pred_labels"]

        keep = pred_scores > score
        pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]

        # computation starts here
        if gt_boxes.size == 0:
            all_pred.append(pred_labels)
            all_target.append(np.ones(len(pred_labels)) * -1)
            all_boxes.append(pred_boxes)
        elif pred_boxes.size == 0:
            all_pred.append(np.ones(len(gt_classes)) * -1)
            all_target.append(gt_classes)
            all_boxes.append(gt_boxes)
        else:
            match_quality_matrix = box_iou_np(gt_boxes, pred_boxes)

            matched_idxs = np.argmax(match_quality_matrix, axis=0)
            matched_vals = np.max(match_quality_matrix, axis=0)
            matched_idxs[matched_vals < iou] = -1

            matched_gt_boxes_per_image = gt_boxes[matched_idxs.clip(min=0)]
            target_labels = gt_classes[matched_idxs.clip(min=0)]
            target_labels[matched_idxs == -1] = -1

            all_pred.append(pred_labels)
            all_target.append(target_labels)
            all_boxes.append(pred_boxes)

            unmatched_gt = (match_quality_matrix.max(axis=1) < iou)
            false_negatives = unmatched_gt.sum()
            if false_negatives > 0:  # false negatives
                all_pred.append(np.ones(false_negatives) * -1)
                all_target.append(np.zeros(false_negatives))
                all_boxes.append(gt_boxes[np.nonzero(unmatched_gt)[0]])
    return all_pred, all_target, all_boxes


def plot_sizes(all_pred, all_target, all_boxes, iou, score):
    if len(all_pred) == 0 or len(all_target) == 0:
        return None, None
    _all_pred = np.concatenate(all_pred)
    _all_target = np.concatenate(all_target)
    _all_boxes = np.concatenate([ab for ab in all_boxes if ab.size > 0])

    dists = box_size_np(_all_boxes)
    tp_mask = _all_pred == _all_target
    fp_mask = (_all_pred != _all_target) * (_all_pred != -1)
    fn_mask = (_all_pred != _all_target) * (_all_pred == -1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(dists[tp_mask, 0], dists[tp_mask, 1], dists[tp_mask, 2], c='g', marker='o', label="tp")
    ax.scatter(dists[fp_mask, 0], dists[fp_mask, 1], dists[fp_mask, 2], c='r', marker='x', label="fp")
    ax.scatter(dists[fn_mask, 0], dists[fn_mask, 1], dists[fn_mask, 2], c='b', marker='^', label="fn")
    ax.set_title(
        f"IoU {iou} and Score Threshold {score}: tp {sum(tp_mask)} fp {sum(fp_mask)} fn {sum(fn_mask)}")
    ax.set_xlabel('bounding box size axis 0')
    ax.set_ylabel('bounding box size axis 1')
    ax.set_zlabel('bounding box size axis 2')
    ax.legend()
    return fig, ax


def plot_sizes_bar(all_pred, all_target, all_boxes, iou, score,
                   max_bin: Optional[int] = None ):
    if len(all_pred) == 0 or len(all_target) == 0:
        return None, None
    _all_pred = np.concatenate(all_pred)
    _all_target = np.concatenate(all_target)
    _all_boxes = np.concatenate([ab for ab in all_boxes if ab.size > 0])

    dists = box_size_np(_all_boxes)
    tp_mask = _all_pred == _all_target
    fp_mask = (_all_pred != _all_target) * (_all_pred != -1)
    fn_mask = (_all_pred != _all_target) * (_all_pred == -1)

    fig = plt.figure()
    # ax = fig.add_subplot(111)
    data = {
        "tp": dists[tp_mask, 0] + dists[tp_mask, 1] + dists[tp_mask, 2],
        "fp": dists[fp_mask, 0] + dists[fp_mask, 1] + dists[fp_mask, 2],
        "fn": dists[fn_mask, 0] + dists[fn_mask, 1] + dists[fn_mask, 2],
    }
    # plt.hist(x=[data["tp"], data["fp"], data["fn"]],
    #          bins=100, label=["tp", "fp", "fn"], stacked=False,
    #          histtype="step",
    #          )
    # ax = plt.gca()
    kwargs = {}
    if max_bin is not None:
        kwargs["binrange"] = [0, max_bin]
    
    ax = sns.histplot(data=data, bins=100, element="step",
                      palette={"tp": "g", "fp": "r", "fn": "b"},
                      legend=True, fill=False, **kwargs
                      )

    ax.set_title(
        f"IoU {iou} and Score Threshold {score}: tp {sum(tp_mask)} fp {sum(fp_mask)} fn {sum(fn_mask)}")
    ax.set_xlabel("box width + height ( + depth)")
    ax.set_ylabel("Count")
    return fig, ax


@experimental
def run_analysis_suite(prediction_dir: Path, gt_dir: Path, save_dir: Path):
    for iou, score in maybe_verbose_iterable(list(product([0.1, 0.5], [0.1, 0.5]))):
        _save_dir = save_dir / f"iou_{iou}_score_{score}"
        _save_dir.mkdir(parents=True, exist_ok=True)

        found_predictions = list(prediction_dir.glob("*_boxes.pkl"))
        logger.info(f"Found {len(found_predictions)} predictions for analysis")

        df, analysis_ids = collect_overview(prediction_dir, gt_dir,
                                            iou=iou, score=score,
                                            max_num_fp_per_image=5,
                                            top_n=10,
                                            )
        df.to_json(_save_dir / "analysis.json", indent=4, orient='index')
        df.to_csv(_save_dir / "analysis.csv")
        save_json(analysis_ids, _save_dir / "analysis_ids.json")

        all_pred, all_target, all_pred_ious, all_pred_scores = collect_score_iou(
            prediction_dir, gt_dir, iou=iou, score=score)
        confusion_ax = plot_confusion_matrix(all_pred, all_target, iou=iou, score=score)
        plt.savefig(_save_dir / "confusion_matrix.png")
        plt.close()

        iou_score_ax = plot_joint_iou_score(all_pred_ious, all_pred_scores)
        plt.savefig(_save_dir / "joint_iou_score.png")
        plt.close()

        all_pred, all_target, all_boxes = collect_boxes(
            prediction_dir, gt_dir, iou=iou, score=score)
        sizes_fig, sizes_ax = plot_sizes(all_pred, all_target, all_boxes, iou=iou, score=score)
        plt.savefig(_save_dir / "sizes.png")
        with open(str(_save_dir / 'sizes.pkl'), "wb") as fp:
            pickle.dump(sizes_fig, fp, protocol=4)
        plt.close()

        sizes_fig, sizes_ax = plot_sizes_bar(all_pred, all_target, all_boxes, iou=iou, score=score)
        plt.savefig(_save_dir / "sizes_bar.png")
        with open(str(_save_dir / 'sizes_bar.pkl'), "wb") as fp:
            pickle.dump(sizes_fig, fp, protocol=4)
        plt.close()
        
        sizes_fig, sizes_ax = plot_sizes_bar(all_pred, all_target, all_boxes,
                                             iou=iou, score=score, max_bin=100)
        plt.savefig(_save_dir / "sizes_bar_100.png")
        with open(str(_save_dir / 'sizes_bar_100.pkl'), "wb") as fp:
            pickle.dump(sizes_fig, fp, protocol=4)
        plt.close()


@deprecate(deprecate="v0.1", remove="v0.2")
def convert_box_to_nii_meta(pred_boxes: Tensor,
                            pred_scores: Tensor,
                            pred_labels: Tensor,
                            props: dict,
                            ) -> Tuple[sitk.Image, dict]:
        instance_mask = np.zeros(props["original_size_of_raw_data"], dtype=np.uint8)

        for instance_id, pbox in enumerate(pred_boxes, start=1):
            mask_slicing = [slice(int(pbox[0]), int(pbox[2])),
                            slice(int(pbox[1]), int(pbox[3])),
                            ]
            if instance_mask.ndim == 3:
                mask_slicing.append(slice(int(pbox[4]), int(pbox[5])))
            instance_mask[tuple(mask_slicing)] = instance_id
        logger.info(f"Created instance mask with {instance_mask.max()} instances.")

        instance_mask_itk = sitk.GetImageFromArray(instance_mask)
        instance_mask_itk.SetOrigin(props["itk_origin"])
        instance_mask_itk.SetDirection(props["itk_direction"])
        instance_mask_itk.SetSpacing(props["itk_spacing"])
        
        prediction_meta = {idx: {"score": float(score), "label": int(label)}
            for idx, (score, label) in enumerate(zip(pred_scores, pred_labels), start=1)}
        return instance_mask_itk, prediction_meta
