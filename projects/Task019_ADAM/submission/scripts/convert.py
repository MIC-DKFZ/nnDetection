import argparse

from pathlib import Path

from nndet.io import load_pickle
from nndet.core.boxes.ops_np import box_center_np

THRESHOLD = 0.5

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    args = parser.parse_args()

    source = args.source

    predictions = load_pickle(source / "case_boxes.pkl")
    boxes = predictions["pred_boxes"]
    scores = predictions["pred_scores"]
    keep = scores > THRESHOLD

    boxes = boxes[keep]
    if boxes.size > 0:
        centers = box_center_np(boxes)
    else:
        centers = []

    with open(source / "result.txt", "a") as f:
        if len(centers) > 0:
            for c in centers[:-1]:
                f.write(f"{round(float(c[2]))}, {round(float(c[1]))}, {round(float(c[0]))}\n")
            c = centers[-1]
            f.write(f"{round(float(c[2]))}, {round(float(c[1]))}, {round(float(c[0]))}")
