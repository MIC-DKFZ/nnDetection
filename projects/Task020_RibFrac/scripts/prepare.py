import os
import shutil
from pathlib import Path

import pandas as pd

from nndet.io import save_json
from nndet.utils.check import env_guard
from nndet.utils.info import maybe_verbose_iterable


def create(
    image_source: Path,
    label_source: Path,
    image_target_dir: Path,
    label_target_dir: Path,
    df: pd.DataFrame,
    fg_only: bool = False,
    ):
    image_target_dir.mkdir(parents=True, exist_ok=True)
    label_target_dir.mkdir(parents=True, exist_ok=True)

    case_id = image_source.stem.rsplit('-', 1)[0]
    case_id_check = label_source.stem.rsplit('-', 1)[0]
    assert case_id == case_id_check, f"case ids not matching, found image {case_id} and label {case_id_check}"

    df_case = df.loc[df['public_id'] == case_id]
    instances = {}
    for row in df_case.itertuples():
        _cls = int(row.label_code)
        if _cls == 0:   # background has label code 0 and lab id 0
            continue

        if fg_only:
            _cls = 1
        elif _cls == -1:
            _cls = 5

        instances[str(row.label_id)] = _cls - 1  # class range from 0 - 4 // if fg only 0
        assert 0 < _cls < 6, f"Something strange happened {_cls}"
    save_json({"instances": instances}, label_target_dir / f"{case_id}.json")

    shutil.copy2(image_source, image_target_dir / f"{case_id}_0000.nii.gz")
    shutil.copy2(label_source, label_target_dir / f"{case_id}.nii.gz")


@env_guard
def main():
    det_data_dir = Path(os.getenv('det_data'))
    task_data_dir = det_data_dir / "Task020_RibFrac"
    source_data_dir = task_data_dir / "raw"

    if not source_data_dir.is_dir():
        raise RuntimeError(f"{source_data_dir} should contain the raw data but does not exist.")
    if not (p := source_data_dir / "imagesTr").is_dir():
        raise ValueError(f"Expected data to be located at {p}")
    if not (p := source_data_dir / "labelsTr").is_dir():
        raise ValueError(f"Expected labels to be located at {p}")
    if not (p := source_data_dir / "ribfrac-train-info-1.csv").is_file():
        raise ValueError(f"Expected {p} to exist.")
    if not (p := source_data_dir / "ribfrac-train-info-2.csv").is_file():
        raise ValueError(f"Expected {p} to exist.")
    if not (p := source_data_dir / "ribfrac-val-info.csv").is_file():
        raise ValueError(f"Expected {p} to exist.")

    target_data_dir = task_data_dir / "raw_splitted" / "imagesTr"
    target_data_dir.mkdir(exist_ok=True, parents=True)
    target_label_dir = task_data_dir / "raw_splitted" / "labelsTr"
    target_label_dir.mkdir(exist_ok=True, parents=True)

    csv_fies = [source_data_dir / "ribfrac-train-info-1.csv",
                source_data_dir / "ribfrac-train-info-2.csv",
                source_data_dir / "ribfrac-val-info.csv"]
    df = pd.concat([pd.read_csv(f) for f in csv_fies])

    image_paths = list((source_data_dir / "imagesTr").glob("*.nii.gz"))
    image_paths.sort()
    label_paths = list((source_data_dir / "labelsTr").glob("*.nii.gz"))
    label_paths.sort()
    
    print(f"Found {len(image_paths)} data files and {len(label_paths)} label files.")
    assert len(image_paths) == len(label_paths)

    meta = {
        "name": "RibFracFG",
        "task": "Task020FG_RibFrac",
        "target_class": None,
        "test_labels": False,
        "labels": {"0": "fracture"}, # since we are running FG vs BG this is not completely correct
        "modalities": {"0": "CT"},
        "dim": 3,
    }
    save_json(meta, task_data_dir / "dataset.json")

    for ip, lp in maybe_verbose_iterable(list(zip(image_paths, label_paths))):
        create(image_source=ip,
               label_source=lp,
               image_target_dir=target_data_dir,
               label_target_dir=target_label_dir,
               df=df,
               fg_only=True,
               )


if __name__ == '__main__':
    main()
