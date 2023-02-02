import math

import numpy as np
import pytest

try:
    import monai
except ImportError:
    monai = None

from nndet.evaluator.registry import BoxEvaluator


@pytest.fixture
def example():
    np.random.seed(0)

    predictions = []
    targets = []

    n_gt_all = 0
    n_pred_all = 0
    n_img = 10000

    for idx in range(n_img):
        if idx % 20 == 0:
            # add empty gt image + empty pred
            predictions.append(
                {
                    "pred_boxes": np.array([[]]).reshape(-1, 6),
                    "pred_labels": np.array([]),
                    "pred_scores": np.array([]),
                }
            )
            targets.append(
                {
                    "target_boxes": np.array([[]]).reshape(-1, 6),
                    "target_classes": np.array([]),
                }
            )
        elif idx % 19 == 0:
            # add empty pred image
            predictions.append(
                {
                    "pred_boxes": np.array([[]]).reshape(-1, 6),
                    "pred_labels": np.array([]),
                    "pred_scores": np.array([]),
                }
            )
            n_gt = int(np.random.randint(low=1, high=5, size=1))
            n_gt_all += n_gt

            targets.append(
                {
                    "target_boxes": np.arange(n_gt * 6).reshape(-1, 6),
                    "target_classes": np.random.randint(3, size=n_gt),
                }
            )
        elif idx % 18 == 0:
            # add empty gt image
            n_pred = int(np.random.randint(low=1, high=5, size=1))
            n_pred_all += n_pred
            predictions.append(
                {
                    "pred_boxes": np.arange(n_pred * 6).reshape(-1, 6),
                    "pred_labels": np.random.randint(3, size=n_pred),
                    "pred_scores": np.random.rand(n_pred),
                }
            )
            targets.append(
                {
                    "target_boxes": np.array([[]]).reshape(-1, 6),
                    "target_classes": np.array([]),
                }
            )
        else:
            n_tp = int(np.random.randint(low=0, high=8, size=1))
            n_fp = int(np.random.randint(low=0, high=16, size=1))
            n_fn = int(np.random.randint(low=0, high=2, size=1))

            n_pred = n_tp + n_fp
            n_gt = n_tp + n_fn
            n_pred_all += n_pred
            n_gt_all += n_gt

            # first indices tp
            # second indices fp
            # third indices fn

            offset_preds = np.array([0] * n_tp * 6 + [n_tp] * n_fp * 6).reshape(-1, 6)
            offset_gt = np.array([0] * n_tp * 6 + [n_tp + n_fp] * n_fn * 6).reshape(-1, 6)

            predictions.append(
                {
                    "pred_boxes": np.arange(n_pred * 6).reshape(-1, 6) + offset_preds,
                    "pred_labels": np.random.randint(3, size=n_pred),
                    "pred_scores": np.random.rand(n_pred),
                }
            )
            targets.append(
                {
                    "target_boxes": np.arange(n_gt * 6).reshape(-1, 6) + offset_gt,
                    "target_classes": np.random.randint(3, size=n_gt),
                }
            )
    return predictions, targets, n_img, n_pred_all, n_gt_all


def test_evaluator(snapshot, example):
    example_preds, example_gt, n_img, n_pred_all, n_gt_all = example
    assert len(example_preds) == len(example_gt)

    evaluator = BoxEvaluator.create(
        classes=["class0", "class1", "class2"],
        fast=False,
        verbose=True,
        save_dir=None,
    )

    for idx, (p, t) in enumerate(zip(example_preds, example_gt)):
        evaluator.run_online_evaluation(
            pred_boxes=[p["pred_boxes"]],
            pred_classes=[p["pred_labels"]],
            pred_scores=[p["pred_scores"]],
            gt_boxes=[t["target_boxes"]],
            gt_classes=[t["target_classes"]],
            gt_ignore=None,
            case_id=[f"case{idx}"],
        )

    res = evaluator.finish_online_evaluation()

    assert res[1]["FROC_num_images"] == n_img
    assert res[1]["FROC_num_gt"] == n_gt_all
    assert res[1]["class0_FROC_num_images"] == res[1]["class1_FROC_num_images"]
    assert res[1]["class1_FROC_num_images"] == res[1]["class2_FROC_num_images"]
    assert res[1]["class2_FROC_num_images"] == n_img
    assert res[1]["class0_FROC_num_gt"] + res[1]["class1_FROC_num_gt"] + res[1]["class2_FROC_num_gt"] == n_gt_all
    assert res == snapshot


@pytest.mark.skipif(monai is None, reason="Monai is not available")
def test_froc_against_monai(example):
    from monai.metrics.froc import compute_froc_curve_data, compute_froc_score

    example_preds, example_gt, n_img, n_pred_all, n_gt_all = example
    assert len(example_preds) == len(example_gt)
    evaluator = BoxEvaluator.create(
        classes=["class0", "class1", "class2"],
        fast=False,
        verbose=True,
        save_dir=None,
    )

    for idx, (p, t) in enumerate(zip(example_preds, example_gt)):
        evaluator.run_online_evaluation(
            pred_boxes=[p["pred_boxes"]],
            pred_classes=[p["pred_labels"]],
            pred_scores=[p["pred_scores"]],
            gt_boxes=[t["target_boxes"]],
            gt_classes=[t["target_classes"]],
            gt_ignore=None,
            case_id=[f"case{idx}"],
        )

    res = evaluator.finish_online_evaluation()
    froc_score_nndet = res[0]["FROC_score_IoU_0.10"]

    results_list = evaluator.results_dict[""]

    results = [_r for r in results_list for _r in r.values()]
    tp_props = np.concatenate([r["dtScores"][r["dtMatches"][0] == 1] for r in results])
    fp_props = np.concatenate([r["dtScores"][r["dtMatches"][0] == 0] for r in results])

    fppi, sens = compute_froc_curve_data(fp_probs=fp_props, tp_probs=tp_props, num_targets=n_gt_all, num_images=n_img)
    froc_score_monai = compute_froc_score(fppi, sens, eval_thresholds=(0.125, 0.25, 0.5, 1, 2, 4, 8))

    assert res[1]["FROC_num_images"] == n_img
    assert res[1]["FROC_num_gt"] == n_gt_all
    assert res[1]["class0_FROC_num_images"] == res[1]["class1_FROC_num_images"]
    assert res[1]["class1_FROC_num_images"] == res[1]["class2_FROC_num_images"]
    assert res[1]["class2_FROC_num_images"] == n_img
    assert res[1]["class0_FROC_num_gt"] + res[1]["class1_FROC_num_gt"] + res[1]["class2_FROC_num_gt"] == n_gt_all

    assert math.isclose(float(froc_score_nndet), float(froc_score_monai))
