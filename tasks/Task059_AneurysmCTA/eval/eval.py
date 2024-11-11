import argparse
import logging
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from sklearn.metrics._ranking import _binary_clf_curve
from tabulate import tabulate
from tqdm import tqdm

from nndet.io import load_pickle
import nndet.core.ops_np as ops_np
from nndet.utils.check import env_guard
from loguru import logger

# nndet uses -1 (lower boundary) and +1 (upper boundary) for the sizes
# ref implementation only uses +1 at the upper boundary
SIZE_OFFSET = 1


################################################################################
# DIRECT COPY OF EVAL CODE https://github.com/alceballosa/deform-aneurysm-detection
# ADDED ONE LINE TO PRINT THE MEAN AND CHANGED some prints to logger.info
# MAKE SURE TO CHECK THE LICENSE IN THE FOLDER!!
# DATE: 2024-11-06

DISEASE = "aneurysm"
np.set_printoptions(linewidth=310)

class FROCEvaluator:
    def __init__(
        self,
        label_file,
        # pred_file,
        preds,
        *,
        logger=None,
        iou_thr=0.4,
        out_dir=None,
        max_fppi=None,
        n_workers=8,
        n_bootstraps=10000,
        ci=0.95,
        n_fppi=10000,
        fppi_thrs=[
            0.125,
            0.25,
            0.5,
            1.0,
        ],
        seed=0,
        out_bs=500,
        save_curves=False,
        min_fppi=1e-4,
        fp_scale="linear",
        meta_data=None,
        exp_name=None,
        use_world_xyz=True,
        mode="val",
    ):
        assert fp_scale in ["linear", "log"]
        self._iou_thr = iou_thr
        out_dir = Path(out_dir)
        self._out_dir = out_dir
        self._mode = mode
        if out_dir is not None:
            os.makedirs(out_dir / "figures", exist_ok=True)
            os.makedirs(out_dir / "bt_cache", exist_ok=True)
            os.makedirs(out_dir / "curves", exist_ok=True)
        self._max_fppi = max_fppi
        self._min_fppi = min_fppi
        self.use_world_xyz = use_world_xyz
        self._fp_scale = fp_scale
        self._n_fppi = n_fppi
        self._fppi_thrs = np.array(fppi_thrs)
        self._metric_names = [f"Se@FPpI={x:.3f}" for x in fppi_thrs]
        self._n_workers = n_workers
        self._n_bootstraps = n_bootstraps
        self._ci = ci
        self._seed = seed
        self._out_bs = out_bs
        self._save_curves = save_curves
        self._logger = logging.getLogger(__name__) if logger is None else logger
        self._exp_name = exp_name
        # self._gts, self._categories, self._images = self.parse_gt_json(label_file)
        self._gts, self._categories, self._images = self.parse_gt_csv(
            label_file, meta=meta_data
        )
        # self._dts = self.parse_dt_json(preds, self._categories)
        self._dts = self.parse_dt_csv(preds, self._categories, meta=meta_data)

        # compute totol pos per category
        n_pos_per_cat = {}
        for cat in self._categories.values():
            n_pos = sum([len(x["box"]) for x in self._gts[cat].values()])
            n_pos_per_cat[cat] = n_pos
        #print(n_pos_per_cat)
        self._n_pos_per_cat = n_pos_per_cat
        if mode == "val":
            self._images = [f"Ts{i:0>4}.nii.gz" for i in range(1, 153)]
        elif mode == "train":
            self._images = [f"Tr{i:0>4}.nii.gz" for i in range(1, 1187)]
        elif mode == "ext":
            self._images = [f"ExtA{i:0>4}.nii.gz" for i in range(1, 72)] + [
                f"ExtB{i:0>4}.nii.gz" for i in range(1, 68)
            ]
        elif mode == "priv" or mode == "hospital":
            self._images = [f"CA_{i:0>5}_0000.nii.gz" for i in range(0, 38)]

    def evaluate(self):
        # compute iou

        ious = {}  # category -> img_id -> iou matrix
        for category in self._categories.values():
            per_cat_ious = {}
            for img_id in self._images:

                pred = self._dts[category].get(img_id, {"box": []})["box"]
                gt = self._gts[category].get(img_id, {"box": []})["box"]

                iou = None
                if len(pred) and len(gt):
                    iou = self._pairwise_iou(gt, pred)
                per_cat_ious[img_id] = iou
            ious[category] = per_cat_ious

        # matching per category per image detection
        # image_id -> category -> {
        #    'scores': list of score,
        #    'gts': list of corresponding gt
        # }
        match_result = defaultdict(dict)

        col_detections = "model", "seriesuid", "detected"

        list_detections = []

        for img_id in self._images:
            for category in self._categories.values():
                p_boxes = self._dts[category].get(img_id, {"box": []})["box"]
                p_scores = self._dts[category].get(img_id, {"score": []})["score"]
                gts = self._gts[category].get(img_id, [])
                dict_match, un_matched_gt = self._match(
                    p_boxes, p_scores, gts, ious[category][img_id], self._iou_thr
                )
                dict_match["un_matched_gt"] = un_matched_gt

                match_result[img_id][category] = dict_match
                for i in range(len(un_matched_gt) - 1):
                    if un_matched_gt[i] == 1:
                        list_detections.append((self._exp_name, img_id, False))
                    else:
                        list_detections.append((self._exp_name, img_id, True))

        df_detections = pd.DataFrame(list_detections, columns=col_detections)
        df_detections.to_csv(
            os.path.join(self._out_dir, "model_detections.csv"), index=False
        )
        print(self._out_dir)
        self._match_results = match_result

        # self._compute_froc(save_fig=True)

    def get_bootstrap_data(self):
        match_results = self._match_results
        match_list, n_pos = [], []
        categories = list(self._categories.values())
        gts = self._gts
        for img_id in self._images:
            match_list.append(match_results[img_id])
            pos = {c: len(gts[c].get(img_id, {"box": []})["box"]) for c in categories}
            n_pos.append(pos)

        return match_list, n_pos, categories

    def run_compute_froc(self, save_fig=False):
        fppi_thrs = self._fppi_thrs
        cats = []
        results = []
        n_imgs = len(self._images)
        print("computing froc ...")
        for category in tqdm(self._categories.values()):
            cats.append(category)
            gts, preds = [], []
            for match in self._match_results.values():
                gts.append(match[category]["gts"])
                preds.append(match[category]["scores"])
            gts = np.concatenate(gts)
            preds = np.concatenate(preds)

            recalls, FPpI, _ = compute_froc(
                preds, gts, self._n_pos_per_cat[category], n_imgs
            )
            results.append(np.interp(fppi_thrs, FPpI, recalls))
            print(np.interp(fppi_thrs, FPpI, _))
            if save_fig:
                self._save_fig(recalls, FPpI, category)

            if self._save_curves:
                cat_file_name = category.replace("/", "_").replace(" ", "_").lower()
                torch.save(
                    {"recall": recalls, "fppi": FPpI},
                    os.path.join(self._out_dir, "curves", f"curve_{cat_file_name}.pth"),
                )
        self._derive_results(cats, results)

    def _derive_results(self, classes, results):
        def f2str(x):
            if x < 0.9995:
                return f"{x:.3f}"[1:]
            return "1.00"

        def str2f(x):
            if x[0] == "1":
                return 1.0
            return float(f"0{x}")

        metric_names = self._metric_names
        results_table = []
        for cat, k_results in zip(classes, results):
            row = [cat] + [f2str(float(x)) for x in k_results]
            results_table.append(row)

        headers = ["Finding"] + metric_names
        df = pd.DataFrame(results_table, columns=headers)
        means = {}
        for metric in metric_names:
            means[metric] = f2str(df[metric].apply(lambda x: str2f(x[:4])).mean())
        means[headers[0]] = "Mean"
        df.loc["mean"] = means
        df.to_csv(os.path.join(self._out_dir, "froc.csv"), index=False, columns=headers)

        logger.info(f"Found results: {results}")
        logger.info(f"MEAN: {results[0].mean()}")

        table = tabulate(
            results_table + [[means[x] for x in headers]],
            tablefmt="pipe",
            floatfmt=".3f",
            headers=headers,
            numalign="left",
        )
        # self._logger.info(f"Per-finding bbox FROC at iou {self._iou_thr} \n" + table)
        logger.info((f"Per-finding bbox FROC at iou {self._iou_thr} \n" + table))

    def _derive_bt_results(self, classes, m_results, ub_results, lb_results):
        """
        results: shape (K,M) K: num classes, M: num metrics
        """

        def f2str(x):
            if x < 0.9995:
                return f"{x:.3f}"[1:]
            return "1.00"

        def str2f(x):
            if x[0] == "1":
                return 1.0
            return float(f"0{x}")

        metric_names = self._metric_names
        results_table = []
        for cat, k_means, k_lbs, k_ubs in zip(
            classes, m_results, lb_results, ub_results
        ):
            row = [cat]
            for mean, lb, ub in zip(k_means, k_lbs, k_ubs):
                # print(mean, lb, ub)
                row.append(
                    f"{f2str(float(mean))}({f2str(float(lb))}--{f2str(float(ub))})"
                )
            results_table.append(row)

        headers = ["Finding"] + metric_names
        df = pd.DataFrame(results_table, columns=headers)
        # means = {}
        # for metric in metric_names:
        #     means[metric] = f2str(df[metric].apply(lambda x: str2f(x[:4])).mean())
        # means[headers[0]] = "Mean"
        # df.loc['mean'] = means
        df.to_csv(
            os.path.join(self._out_dir, "froc_bt.csv"), index=False, columns=headers
        )

        table = tabulate(
            # results_table + [[means[x] for x in headers]],
            results_table,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=headers,
            numalign="left",
        )
        

    def _save_fig(self, recalls, FPpI, category, *, rec_ub=None, rec_lb=None):
        assert self._out_dir is not None
        plt.close()
        path = os.path.join(
            self._out_dir, "figures", f"{category.replace('/', '_')}_froc.png"
        )

        path_recalls = os.path.join(
            self._out_dir, "figures", f"{category.replace('/', '_')}_recalls.npy"
        )
        np.save(path_recalls, recalls)
        path_FPpI = os.path.join(
            self._out_dir, "figures", f"{category.replace('/', '_')}_FPpI.npy"
        )
        np.save(path_FPpI, FPpI)
        plt.figure(figsize=(5, 5))
        plt.title(f"{category} FROC")
        plt.plot(FPpI, recalls)
        plt.xlabel("Average number of false positive per scan")
        plt.ylabel("Recall")
        if self._fp_scale == "log":
            plt.xscale("log")
        if rec_lb is not None:
            path = path.replace("froc.", "froc_bootstrap.")
            plt.plot(
                FPpI,
                rec_ub,
                "r--",
                FPpI,
                rec_lb,
                "r--",
            )
        xmax = FPpI.max()
        max_fppi = self._max_fppi
        if max_fppi:
            if isinstance(max_fppi, (float, int)):
                xmax = float(max_fppi)
            else:
                xmax = float(max_fppi[category])
        if self._fp_scale == "log":
            xmin = self._min_fppi
        else:
            xmin = -0.1
        plt.xlim(xmin, xmax)
        plt.ylim(xmin, 1.01)
        plt.grid(linestyle="--", which="both")
        plt.savefig(path)

    def _match(self, p_boxes, p_scores, gts, ious, iou_thr):
        """
        assess whether each pred is pos or neg
        args
            threshold: iou threshold to match

        return
        """
        assert len(p_boxes) == len(p_scores)
        # no_pred_score = -0.01
        if not (len(gts) or len(p_scores)):
            return {
                "scores": np.array([]),
                "gts": np.array([]),
                "all_un_matched_gt": np.array([]),
            }, []

        if not len(gts):
            scores = p_scores.numpy()
            gts = np.zeros_like(scores)
            return {"scores": scores, "gts": gts, "all_un_matched_gt": np.array([])}, []

        if not len(p_scores):
            # gts = np.ones(size=(len(gts),))
            # scores = no_pred_score * gts
            # return {"scores": scores, "gts": gts}
            return {
                "scores": np.array([]),
                "gts": np.array([]),
                "all_un_matched_gt": np.array([]),
            }, []

        un_matched_gt = torch.ones(size=(ious.size(0) + 1,))

        _, sorted_ids = torch.sort(p_scores, descending=True)
        all_un_matched_gts = []
        r_scores, r_gts = [], []
        for i in sorted_ids:
            i_iou = ious[:, i]
            matched_gt_id = -1
            best_iou = -1
            for gt_id, iou in enumerate(i_iou):
                if iou >= iou_thr and un_matched_gt[gt_id] > 0 and iou > best_iou:
                    matched_gt_id = gt_id
                    best_iou = iou

            # update un_matched_gt
            un_matched_gt[matched_gt_id] = 0
            all_un_matched_gts.append(
                [float(p_scores[i]), un_matched_gt.numpy().copy()]
            )

            # update results
            gt = 1.0 if matched_gt_id > -1 else 0.0
            r_gts.append(gt)
            r_scores.append(float(p_scores[i]))

        # add false negative (if any)
        # for not_detected in un_matched_gt[:-1]:
        #     if not_detected:
        #         r_scores.append(no_pred_score)
        #         r_gts.append(1.)

        return (
            {
                "scores": np.array(r_scores),
                "gts": np.array(r_gts),
                "all_un_matched_gt": all_un_matched_gts,
            },
            un_matched_gt.numpy(),
        )

    def _pairwise_iou(self, box_list1, box_list2):
        """
        compute pairwise 3d iou

        args:
            box_list1: shape (N,4)
            box_list2: shape (M,4)
        return:
            iou shape (N,M)

        assume box is non empty
        """
        # compute intersection
        width_height = torch.min(
            box_list1[:, None, 3:], box_list2[None, :, 3:]
        ) - torch.max(box_list1[:, None, :3], box_list2[None, :, :3])
        width_height.clamp_(min=0.0)
        intersection = width_height.prod(dim=2)  # (N,M)

        # compute area
        area1 = (box_list1[:, 3:] - box_list1[:, :3]).prod(dim=1)
        area2 = (box_list2[:, 3:] - box_list2[:, :3]).prod(dim=1)
        if self._mode not in ["hospital"]:
            return intersection / (area1[:, None] + area2[None, :] - intersection)
        else:
            # print("Using iom")
            # print(intersection.shape, area1.shape, area2.shape)
            # return intersection / (area1[:, None] + area2[None, :] - intersection)
            return intersection / (np.minimum(area1[:, None], area2[None, :]))

    def parse_gt_csv(self, path, meta=None):
        """
        json file follows coco ground truth format
        return
            -- dict(disease-> image_id -> {"box": tensor,}),
            -- dict(catid -> category)
            -- list of img_id
        """
        results = {}
        data = pd.read_csv(path)

        id2disease = {1: DISEASE}
        all_imgs = []

        # self._logger.info(f"got {len(all_imgs)} gt images")

        #print("parsing ground truth ...")
        for seriesuid, rows in data.groupby("seriesuid"):
            all_imgs.append(seriesuid)
            box = np.array(rows[["coordX", "coordY", "coordZ", "w", "h", "d"]])
            if self._mode == "hospital":
                sides = box[:, 3:]

                minimum_side = np.argmin(sides, axis=1)
                # get the second largest side value  using argsort
                second_largest_side = np.argsort(box[:, 3:], axis=1)[:, 1]

                sides[:, minimum_side] = sides[:, second_largest_side]
                box[:, 3:] = sides
                # for i in range(len(box)):
                #    box[i,3:] = box[i,3:].mean()

            if meta is not None and self._mode not in ["ext",  "hospital"]:
                origin = np.array(meta[seriesuid]["origin"])
                spacing = np.array(meta[seriesuid]["spacing"])
                # only convert box xyz if needed
                box[:, :3] = box[:, :3] * spacing + origin

            box = xyzwhd2xyzxyz(torch.tensor(box))

            results[seriesuid] = {"box": box}

        return {DISEASE: results}, id2disease, all_imgs

    def parse_dt_csv(self, preds, id2disease, meta=None):
        """
        args: preds: prediction_df
        return dict(disease-> image_id -> {"box": tensor, "score": tensor})
        """
        results = {}

        def fix_bad_origin(pred, spacing, origin):
            pred_part = pred * spacing + origin * np.array([1, -1, 1])
            return pred_part * np.array([1, -1, 1])

        #print("parsing predictions ...")
        for seriesuid, rows in preds.groupby("seriesuid"):
            box_data = np.array(
                rows[["coordX", "coordY", "coordZ", "w", "h", "d", "probability"]]
            )

            # convert box in pixel coordinate to world coordinates

            if meta is not None:

                origin = np.array(meta[seriesuid]["origin"])
                spacing = np.array(meta[seriesuid]["spacing"])
                # only convert box xyz if needed

                if self._mode == "hospital":
                    box_data[:, :3] = fix_bad_origin(box_data[:, :3], spacing, origin)
                else:
                    box_data[:, :3] = box_data[:, :3] * spacing + origin
                box_data[:, 3:6] *= spacing
            results[seriesuid] = {"box": box_data[:, :6], "score": box_data[:, -1]}

        # convert to tensor and sort box
        for image in results.values():
            score, sorted_id = torch.sort(torch.tensor(image["score"]), descending=True)
            image["score"] = score

            box = torch.tensor(image["box"])[sorted_id]
            image["box"] = xyzwhd2xyzxyz(box)

        return {DISEASE: results}


def xyzwhd2xyzxyz(boxes):
    res = torch.zeros_like(boxes)
    res[:, :3] = boxes[:, :3] - boxes[:, 3:] / 2
    res[:, 3:] = boxes[:, :3] + boxes[:, 3:] / 2
    return res


def compute_froc(preds, gts, n_pos, n_imgs, *, outputs=None):
    """
    compute froc and return froc curve

    args
        -- preds: np.array of scores
        -- gts: np.array of gt (0. or 1.)

    return (array of recalls, array of FP per img)
    """
    # n_gt_pos = gts.sum()
    # sorted_ids = np.argsort(preds, kind="mergesort")[::-1]
    # preds = preds[sorted_ids]
    fps, tps, thrs = _binary_clf_curve(gts, preds)
    # print(fps, "\n", tps)
    # listas
    if outputs:
        assert outputs[-4:] == ".pth"
        torch.save(
            {"fps": fps, "tps": tps, "thrs": thrs, "n_pos": n_pos, "n_imgs": n_imgs},
            outputs,
        )

    recalls = tps / n_pos
    FPpI = fps / n_imgs
    return recalls.astype(np.float32), FPpI.astype(np.float32), thrs.astype(np.float32)

################################################################################


@env_guard
def main():
    """
    This is a nnDetection adaption layer for the provided evaluation script.
    nnDetection uses a different box format than the original code so we need
    to account for that here. The FROC scores are otherwise identical 
    to the FROC computed by nnDetection.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        help="full name of experiment to sweep e.g. RetinaUNetV0_D3V001_3d",
    )
    parser.add_argument(
        "--external",
        help="Evaluate external test set instead of internal test set",
        action="store_true",
    )
    args = parser.parse_args()
    model = args.model
    external = args.external

    det_models = Path(os.getenv("det_models"))
    det_data = Path(os.getenv("det_data"))
    model_base = det_models / "Task059_ANeurysmCTA" / model / "consolidated"

    prefix = "external" if external else "internal"
    prediction_dir_name = "test_predictions_internal" if not external else "test_predictions_external"
    gt_dir_name = "labelsTs_internal" if not external else "labelsTs_external"

    prediction_dir = model_base / prediction_dir_name
    gt_dir = det_data / "Task059_ANeurysmCTA" / "preprocessed" / gt_dir_name

    assert prediction_dir.is_dir(), f"Prediction path {prediction_dir} does not exist"
    assert gt_dir.is_dir(), f"GT path {gt_dir} does not exist"

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(model_base / "eval_official.log", level="DEBUG")

    pred_csv_path = model_base / f"preds_{prefix}.csv"
    gt_csv_path = model_base / f"gt_{prefix}.csv"
    out_dir: Path = model_base / f"results_prov_script_{prefix}"
    out_dir.mkdir(exist_ok=True)
    exp = "model"

    _convert(
        prediction_dir=prediction_dir,
        gt_dir=gt_dir,
        pred_csv_path=pred_csv_path,
        gt_csv_path=gt_csv_path,
        external=external,
    )
    _run(
        label_file=gt_csv_path,
        pred_file=pred_csv_path,
        out_dir=out_dir,
        exp=exp,
        external=external,
    )


def _convert(
    prediction_dir: Path,
    gt_dir: Path,
    pred_csv_path: Path,
    gt_csv_path: Path,
    external: bool,
):
    prediction_files = list(prediction_dir.glob("*.pkl"))
    prediction_files.sort()

    case_ids = [f.stem.rsplit("_", 1)[0] for f in prediction_files]
    logger.info(f"Found {len(case_ids)} cases")
    logger.info(f"Cases: {case_ids}")

    if external:
        assert len(case_ids) == 138, f"Expected 138 cases, got {len(case_ids)}"
    else:
        assert len(case_ids) == 152, f"Expected 152 cases, got {len(case_ids)}"

    gt_converted_data = []
    pred_converted_data = []

    for cid in case_ids:
        logger.info(f"Converting case {cid}")
        gt_boxes = np.load(gt_dir / f"{cid}_boxes_gt.npz")["boxes"]
        preds = load_pickle(prediction_dir / f"{cid}_boxes.pkl")
        pred_boxes = preds["pred_boxes"]
        pred_scores = preds["pred_scores"]

        # include offset due to conversion
        # adapt lower boundary of the boxes to adjust to ref implementation
        # ref implementation likely evaluates in preprocessed space -> we don't do that
        # labels change after preprocessing making evaluation dependent 
        # on resampling procedure 
        gt_boxes[:, 0] += SIZE_OFFSET
        gt_boxes[:, 1] += SIZE_OFFSET
        gt_boxes[:, 4] += SIZE_OFFSET
        pred_boxes[:, 0] += SIZE_OFFSET
        pred_boxes[:, 1] += SIZE_OFFSET
        pred_boxes[:, 4] += SIZE_OFFSET

        # ground truth handling
        num_gt = len(gt_boxes) if gt_boxes.size > 0 else 0
        gt_centers = ops_np.box_center_np(gt_boxes)
        gt_sizes = ops_np.box_size_np(gt_boxes)
        
        for gt_idx in range(num_gt):
            gt_converted_data.append(
                {
                    "seriesuid": f"{cid}.nii.gz",
                    "coordX": (gt_centers[gt_idx][0]),
                    "coordY": (gt_centers[gt_idx][1]),
                    "coordZ": (gt_centers[gt_idx][2]),
                    "w": (gt_sizes[gt_idx][0]),
                    "h": (gt_sizes[gt_idx][1]),
                    "d": (gt_sizes[gt_idx][2]),
                }
            )

        # prediction handling
        num_pred = len(pred_boxes) if pred_boxes.size > 0 else 0
        pred_centers = ops_np.box_center_np(pred_boxes)
        pred_sizes = ops_np.box_size_np(pred_boxes)

        for pred_idx in range(num_pred):
            pred_converted_data.append(
                {
                    "seriesuid": f"{cid}.nii.gz",
                    "coordX": float(pred_centers[pred_idx][0]),
                    "coordY": float(pred_centers[pred_idx][1]),
                    "coordZ": float(pred_centers[pred_idx][2]),
                    "w": float(pred_sizes[pred_idx][0]),
                    "h": float(pred_sizes[pred_idx][1]),
                    "d": float(pred_sizes[pred_idx][2]),
                    "probability": float(pred_scores[pred_idx]),
                }
            )

    df_gt = pd.DataFrame(gt_converted_data)
    df_preds = pd.DataFrame(pred_converted_data)
    df_gt.to_csv(gt_csv_path, index=False)
    df_preds.to_csv(pred_csv_path, index=False)


def _run(
    label_file: Path,
    pred_file: Path,
    out_dir: Path,
    exp: str,
    external: bool,
    ):
    iou_thr = 0.3
    min_fppi = 1 / 16
    max_fppi = 16
    fppi_thrs = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    n_bootstraps = 10000
    n_workers = 1
    fp_scale = "log"
    meta = None # optionally include meta data with origin and spacing -> adds and rescales preds and images
    mode = "val" if not external else "ext"

    df_preds = pd.read_csv(pred_file)

    evaluator = FROCEvaluator(
        label_file=label_file,
        preds=df_preds,
        iou_thr=iou_thr,
        out_dir=out_dir,
        max_fppi=max_fppi,
        fppi_thrs=fppi_thrs,
        min_fppi=min_fppi,
        n_bootstraps=n_bootstraps,
        n_workers=n_workers,
        fp_scale=fp_scale,
        meta_data=meta,
        use_world_xyz=False, # seems to be unused in the code
        exp_name=exp + f"_{mode}",
        mode=mode,
    )
    evaluator.evaluate()
    evaluator.run_compute_froc(save_fig=True)


if __name__ == "__main__":
    main()
