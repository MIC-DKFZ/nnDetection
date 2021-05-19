import sys
import os
from itertools import repeat
from multiprocessing.pool import Pool

import pandas as pd
import numpy as np
import numpy.testing as npt
import SimpleITK as sitk
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from pathlib import Path

from nndet.io.load import save_json, load_json
from nndet.io.paths import subfiles
from nndet.utils.check import env_guard
from nndet.utils.info import maybe_verbose_iterable


def prepare_case(case_dir: Path, target_dir: Path, df: pd.DataFrame):
    target_data_dir = target_dir / "imagesTr"
    target_label_dir = target_dir / "labelsTr"

    case_id = str(case_dir).split('/')[-1]
    logger.info(f"Processing case {case_id}")
    df = df[df.PatientID == case_id]

    # process data
    img = sitk.ReadImage(str(case_dir / f"{case_id}_ct_scan.nrrd"))
    sitk.WriteImage(img, str(target_data_dir / f"{case_id}.nii.gz"))
    img_arr = sitk.GetArrayFromImage(img)

    # process mask
    final_rois = np.zeros_like(img_arr, dtype=np.uint8)
    mal_labels = {}
    roi_ids = set([ii.split('.')[0].split('_')[-1]
                   for ii in os.listdir(case_dir) if '.nii.gz' in ii])

    rix = 1
    for rid in roi_ids:
        roi_id_paths = [ii for ii in os.listdir(case_dir) if '{}.nii'.format(rid) in ii]
        nodule_ids = [ii.split('_')[2].lstrip("0") for ii in roi_id_paths]
        rater_labels = [df[df.NoduleID == int(ii)].Malignancy.values[0] for ii in nodule_ids]
        rater_labels.extend([0] * (4-len(rater_labels)))
        mal_label = np.mean([ii for ii in rater_labels if ii > -1])

        roi_rater_list = []
        for rp in roi_id_paths:
            roi = sitk.ReadImage(str(case_dir / rp))
            roi_arr = sitk.GetArrayFromImage(roi).astype(np.uint8)
            assert roi_arr.shape == img_arr.shape, [
                roi_arr.shape, img_arr.shape, case_id, roi.GetSpacing()]
            for ix in range(len(img_arr.shape)):
                npt.assert_almost_equal(roi.GetSpacing()[ix], img.GetSpacing()[ix])
            roi_rater_list.append(roi_arr)

        roi_rater_list.extend([np.zeros_like(roi_rater_list[-1])]*(4-len(roi_id_paths)))
        roi_raters = np.array(roi_rater_list)
        roi_raters = np.mean(roi_raters, axis=0)
        roi_raters[roi_raters < 0.5] = 0
        if np.sum(roi_raters) > 0:
            mal_labels[rix] = mal_label
            final_rois[roi_raters >= 0.5] = rix
            rix += 1
        else:
            # indicate rois suppressed by majority voting of raters
            logger.warning(f'suppressed roi! {roi_id_paths}')

    mask_itk = sitk.GetImageFromArray(final_rois)
    sitk.WriteImage(mask_itk, str(target_label_dir / f"{case_id}.nii.gz"))
    instance_classes = {key: int(item >= 3) for key, item in mal_labels}
    save_json({"instances": instance_classes, "scores": mal_labels},
              target_label_dir / f"{case_id}")


def reformat_labels(target: Path):
    for p in subfiles(target, identifier="*json", join=True):
        label = load_json(Path(p))
        mal_labels = label["scores"]
        instance_classes = {key: int(item >= 3) for key, item in mal_labels.items()}
        save_json({"instances": instance_classes, "scores": mal_labels}, Path(p))


def delete_without_label(target: Path):
    for p in subfiles(target, identifier="*.npz", join=True):
        _p = str(p).rsplit('.', 1)[0] + '.pkl'
        if not os.path.isfile(_p):
            os.remove(p)


def check_data_load(target: Path):
    for p in tqdm(subfiles(target, identifier="*.npy", join=True)):
        try:
            data = np.load(p)
        except Exception as e:
            print(f"Failed to load: {p} with {e}")


@env_guard
def main():
    det_data_dir = Path(os.getenv('det_data'))
    task_data_dir = det_data_dir / "Task012_LIDC"
    source_data_dir = task_data_dir / "raw"
    
    if not (p := source_data_dir / "data_nrrd").is_dir():
        raise ValueError(f"Expted {p} to contain LIDC data")
    if not (p := source_data_dir / 'characteristics.csv').is_file():
        raise ValueError(f"Expted {p} to contain exist")

    target_dir = task_data_dir / "raw_splitted"
    target_data_dir = task_data_dir / "raw_splitted" / "imagesTr"
    target_data_dir.mkdir(exist_ok=True, parents=True)
    target_label_dir = task_data_dir / "raw_splitted" / "labelsTr"
    target_label_dir.mkdir(exist_ok=True, parents=True)

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(task_data_dir / "prepare.log", level="DEBUG")

    data_dir = source_data_dir / "data_nrrd"
    case_dirs = [x for x in data_dir.iterdir() if x.is_dir()]
    df = pd.read_csv(source_data_dir / 'characteristics.csv', sep=';')

    for cd in maybe_verbose_iterable(case_dirs):
        prepare_case(cd, target_dir, df)

    # TODO download custom split file


if __name__ == '__main__':
    main()
