import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from loguru import logger

import nndet.core.ops_np as ops_np
from nndet.io.load import load_pickle
from nndet.utils.info import maybe_verbose_iterable

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Name of model")
    parser.add_argument("fold", type=int, help="Fold of model, -1 for consolidated")
    args = parser.parse_args()
    model = args.model
    fold = args.fold

    task_dir = Path(os.getenv("det_models")) / "Task053_PN9"
    model_dir = task_dir / model
    assert model_dir.is_dir(), f"Model directory {model_dir} does not exist"

    fold = f"fold{fold}" if fold >= 0 else "consolidated"

    prediction_dir = model_dir / fold / "test_predictions"
    assert prediction_dir.is_dir(), f"Prediction directory {prediction_dir} does not exist"

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    log_file = model_dir / "prepare_eval_cpm.log"

    prediction_cache = defaultdict(list)
    prediction_paths = sorted([p for p in prediction_dir.iterdir() if p.is_file() and p.name.endswith("_boxes.pkl")])
    logger.info(f"Found {len(prediction_paths)} predictions for evaluation")
    for prediction_path in maybe_verbose_iterable(prediction_paths):
        seriusuid = prediction_path.stem.rsplit("_", 1)[0].replace("_", ".")
        predictions = load_pickle(prediction_path)

        boxes = predictions["pred_boxes"]
        probs = predictions["pred_scores"]
        centers = ops_np.box_center_np(boxes)
        assert predictions["restore"]

        for box, center, prob in zip(boxes, centers, probs):
            # PN9 annotations are 1 indexed, nnDet annotations are 0 indexed
            position_image = (float(center[2]) + 1, float(center[1]) + 1, float(center[0]) + 1)  # x,y,z
            position_world = position_image  # 1mm spacing, no origin offset due to np format

            x_min = float(box[4]) + 1
            x_max = float(box[5]) - 1
            y_min = float(box[1]) + 1
            y_max = float(box[3]) - 1
            z_min = float(box[0]) + 1
            z_max = float(box[2]) - 1
            diameter_mm = max(x_max - x_min, y_max - y_min, z_max - z_min)

            prediction_cache["pid"].append(seriusuid)
            prediction_cache["center_x"].append(float(position_world[0]))
            prediction_cache["center_y"].append(float(position_world[1]))
            prediction_cache["center_z"].append(float(position_world[2]))
            prediction_cache["probability"].append(float(prob))
            prediction_cache["xmin"].append(x_min)
            prediction_cache["xmax"].append(x_max)
            prediction_cache["ymin"].append(y_min)
            prediction_cache["ymax"].append(y_max)
            prediction_cache["zmin"].append(z_min)
            prediction_cache["zmax"].append(z_max)
            prediction_cache["diameter"].append(diameter_mm)

    df = pd.DataFrame(prediction_cache)
    df.to_csv(model_dir / f"{model}.csv")
