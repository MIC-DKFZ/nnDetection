import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
from loguru import logger
from tqdm import tqdm

from nndet.io.itk import load_sitk
from nndet.io.load import load_pickle
from nndet.core.boxes.ops_np import box_center_np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help="Name of model")
    args = parser.parse_args()
    model = args.model

    task_dir = Path(os.getenv("det_models")) / "Task016_Luna"
    model_dir = task_dir / model
    assert model_dir.is_dir()

    raw_splitted_images = Path(os.getenv("det_data")) / "Task016_Luna" / "raw_splitted" / "imagesTr"

    prediction_dir = model_dir / "consolidated" / "val_predictions"
    assert prediction_dir.is_dir()

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    log_file = model_dir / "prepare_eval_cpm.log"

    prediction_cache = defaultdict(list)
    prediction_paths = sorted([p for p in prediction_dir.iterdir() if p.is_file() and p.name.endswith("_boxes.pkl")])
    logger.info(f"Found {len(prediction_paths)} predictions for evaluation")
    for prediction_path in tqdm(prediction_paths):
        seriusuid = prediction_path.stem.rsplit("_", 1)[0].replace('_', ".")
        predictions = load_pickle(prediction_path)

        data_path = raw_splitted_images / f"{prediction_path.stem.rsplit('_', 1)[0]}_0000.nii.gz"
        image_itk = load_sitk(data_path)

        boxes = predictions["pred_boxes"]
        probs = predictions["pred_scores"]
        centers = box_center_np(boxes)
        assert predictions["restore"]

        for center, prob in zip(centers, probs):
            position_image = (float(center[2]), float(center[1]), float(center[0]))
            position_world = image_itk.TransformContinuousIndexToPhysicalPoint(position_image)

            prediction_cache["seriesuid"].append(seriusuid)
            prediction_cache["coordX"].append(float(position_world[0]))
            prediction_cache["coordY"].append(float(position_world[1]))
            prediction_cache["coordZ"].append(float(position_world[2]))
            prediction_cache["probability"].append(float(prob))

    df = pd.DataFrame(prediction_cache)
    df.to_csv(model_dir / f"{model}.csv")
