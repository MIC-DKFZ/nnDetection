import os
import sys
from pathlib import Path

import SimpleITK as sitk
from loguru import logger

from nndet.io import save_json
from nndet.io.itk import load_sitk
from nndet.utils.check import env_guard
from nndet.utils.info import maybe_verbose_iterable


@env_guard
def main():
    task_name = "Task051_VALDO_Microbleeds"
    data_dir = Path(os.getenv("det_data")) / task_name

    raw_data_dir = data_dir / "raw" / "Task2"

    splitted_data_dir = data_dir / "raw_splitted"
    images_tr_dir = splitted_data_dir / "imagesTr"
    images_tr_dir.mkdir(exist_ok=True, parents=True)
    labels_tr_dir = splitted_data_dir / "labelsTr"
    labels_tr_dir.mkdir(exist_ok=True, parents=True)

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(splitted_data_dir / "prepare.log", level="DEBUG")

    # create dataset.json
    meta = {
        "dim": 3,
        "task": task_name,
        "target_class": None,
        "test_labels": True,
        "labels": {
            "0": "CMB",
        },
        "modalities": {"0": "T1", "1": "T2", "2": "T2S"},
        "annotation_style": "seg",
        # needed to run connected components for instances
        "seg2det_stuff": [],
        "seg2det_things": [1],
        "min_size": 0,
        "min_vol": 0,
    }
    save_json(meta, data_dir / "dataset.json")

    case_ids = sorted([p.stem for p in raw_data_dir.iterdir() if p.is_dir() and p.name.startswith("sub-")])
    logger.info(f"Found {len(case_ids)} cases")

    assert len(case_ids) == 72, f"Missing cases, only found {len(case_ids)} cases, expected 72 cases."

    # convert data
    for case_id in maybe_verbose_iterable(case_ids):

        logger.info(f"Processing case {case_id}")
        case_dir = raw_data_dir / case_id
        files = [_p for _p in case_dir.iterdir() if _p.is_file() and not _p.stem.startswith(".")]
        assert len(files) in [3, 4]
        file_types = [_n.name.rsplit("_", 1)[1].split(".", 1)[0] for _n in files]

        t1_itk = load_sitk(files[file_types.index("T1")])
        sitk.WriteImage(t1_itk, str(images_tr_dir / f"case{case_id}_0000.nii.gz"))

        t2_itk = load_sitk(files[file_types.index("T2")])
        sitk.WriteImage(t2_itk, str(images_tr_dir / f"case{case_id}_0001.nii.gz"))

        t2s_itk = load_sitk(files[file_types.index("T2S")])
        sitk.WriteImage(t2s_itk, str(images_tr_dir / f"case{case_id}_0002.nii.gz"))

        if "CMB" in file_types:
            label = load_sitk(files[file_types.index("CMB")])
            sitk.WriteImage(label, str(labels_tr_dir / f"case{case_id}.nii.gz"))
        else:
            # create empty label
            label = sitk.Image(*t1_itk.GetSize(), sitk.sitkUInt8)
            label.CopyInformation(t1_itk)
            sitk.WriteImage(label, str(labels_tr_dir / f"case{case_id}.nii.gz"))


if __name__ == "__main__":
    main()
