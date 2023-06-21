import argparse
import os
import sys
import traceback
from collections import defaultdict
from itertools import repeat
from multiprocessing.pool import Pool

import pandas as pd
import SimpleITK as sitk
from pathlib import Path

from loguru import logger

from nndet.io.itk import create_circle_mask_itk
from nndet.io.load import save_pickle, save_json, save_yaml, load_json
from nndet.utils.check import env_guard


def create_masks(source: Path, target: Path, df: pd.DataFrame, num_processes: int):
    files = []
    split = {}
    for i in range(10):
        subset_dir = source / f"subset{i}"
        if not subset_dir.is_dir():
            logger.error(f"{subset_dir} is not s valid subset directory!")
            continue

        tmp = list((subset_dir.glob('*.mhd')))
        files.extend(tmp)
        for t in tmp:
            split[t.stem.replace('.', '_')] = i
    save_json(split, target.parent.parent / "splits.json")

    centers = []
    rads = []
    for f in files:
        c = []
        r = []
        try:
            series_df = df.loc[[f.name.rsplit('.', 1)[0]]]
        except KeyError:
            pass
        else:
            for _, row in series_df.iterrows():
                c.append((float(row['coordX']), float(row['coordY']), float(row['coordZ'])))
                r.append(float(row['diameter_mm']) / 2)
        centers.append(c)
        rads.append(r)

    assert len(files) == len(centers) == len(rads)
    with Pool(processes=num_processes) as p:
        p.starmap(_create_mask, zip(files, repeat(target), centers, rads))
    # for t in zip(files, repeat(target), centers, rads):
    #     _create_mask(*t)


def _create_mask(source, target, centers, rads):
    try:
        logger.info(f"Processing {source.stem}")
        data = sitk.ReadImage(str(source))
        mask = create_circle_mask_itk(data, centers, rads, ndim=3)
        sitk.WriteImage(mask, str(target / f"{source.stem.replace('.', '_')}.nii.gz"))
        save_json({"instances": {str(k + 1): 0 for k in range(len(centers))}},
                  target / f"{source.stem.replace('.', '_')}.json")
    except Exception as e:
        logger.error(f"Case {source.stem} failed with {e} and {traceback.format_exc()}")


def create_splits(source, target):
    files = []
    for p in source.glob('subset*'):
        path = Path(p)
        if not p.is_dir():
            continue
        _files = [str(i).rsplit('.', 1)[0] for i in path.iterdir() if i.suffix == ".mhd"]
        files.append(_files)
    splits = []
    for i in range(len(files)):
        train_ids = list(range(len(files)))
        test = files[i]
        train_ids.pop(i)
        val = files[(i + 1) % len(files)]
        train_ids.pop((i + 1) % len(files))
        assert len(train_ids) == len(files) - 2
        train = [tr for tri in train_ids for tr in files[tri]]
        splits.append({"train": train, "val": val, "test": test})
    save_pickle(splits, target)


def convert_data(source: Path, target: Path, num_processes: int):
    for subset_dir in source.glob('subset*'):
        subset_dir = Path(subset_dir)
        if not subset_dir.is_dir():
            continue

        with Pool(processes=num_processes) as p:
            p.starmap(_convert_data, zip(subset_dir.glob('*.mhd'), repeat(target)))


def _convert_data(f, target):
    logger.info(f"Converting {f}")
    try:
        data = sitk.ReadImage(str(f))
        sitk.WriteImage(data, str(target / f"{f.stem.replace('.', '_')}_0000.nii.gz"))
    except Exception as e:
        logger.error(f"Case {f} failed with {e} and {traceback.format_exc()}")


@env_guard
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_processes', type=int, default=4, required=False,
                        help="Number of processes to use for preparation.")
    args = parser.parse_args()
    num_processes = args.num_processes

    det_data_dir = Path(os.getenv('det_data'))
    task_data_dir = det_data_dir / "Task016_Luna"
    source_data_dir = task_data_dir / "raw"

    if not source_data_dir.is_dir():
        raise RuntimeError(f"{source_data_dir} should contain the raw data but does not exist.")
    for i in range(10):
        if not (p := source_data_dir / f"subset{i}"):
            raise ValueError(f"Expected {p} to contain Luna data")
    if not (p := source_data_dir / "annotations.csv").is_file():
        raise ValueError(f"Exptected {p} to exist.")

    target_data_dir = task_data_dir / "raw_splitted" / "imagesTr"
    target_data_dir.mkdir(exist_ok=True, parents=True)
    target_label_dir = task_data_dir / "raw_splitted" / "labelsTr"
    target_label_dir.mkdir(exist_ok=True, parents=True)
    target_preprocessed_dir = task_data_dir / "preprocessed"
    target_preprocessed_dir.mkdir(exist_ok=True)

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(task_data_dir / "prepare.log", level="DEBUG")

    meta = {
        "name": "Luna",
        "task": "Task016_Luna",

        "target_class": None,
        "test_labels": False,
        
        "labels": {
            "0": "lesion",
        },
        "modalities": {
            "0": "CT",
        },
        "dim": 3,
    }
    save_json(meta, task_data_dir / "dataset.json")

    # prepare data and labels
    csv = source_data_dir / "annotations.csv"
    convert_data(source_data_dir, target_data_dir, num_processes=num_processes)

    df = pd.read_csv(csv, index_col='seriesuid')
    create_masks(source_data_dir, target_label_dir, df, num_processes=num_processes)

    # generate split
    logger.info("Generating luna splits... ")
    saved_original_splits = load_json(task_data_dir / "splits.json")
    logger.info(f"Found {len(list(saved_original_splits.keys()))} ids in splits.json")
    original_fold_ids = defaultdict(list)
    for cid, fid in saved_original_splits.items():
        original_fold_ids[fid].append(cid)

    splits = []
    for test_fold in range(10):
        all_folds = list(range(10))
        all_folds.pop(test_fold)

        train_ids = []
        for af in all_folds:
            train_ids.extend(original_fold_ids[af])
        splits.append({
            "train": train_ids,
            "val": original_fold_ids[test_fold],
        })
    save_pickle(splits, target_preprocessed_dir / "splits_final.pkl")
    save_json(splits, target_preprocessed_dir / "splits_final.json")

if __name__ == '__main__':
    main()
