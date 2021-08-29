import os
import sys
import traceback
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import SimpleITK as sitk
from nndet.io.prepare import create_test_split
from loguru import logger

from nndet.utils.check import env_guard
from nndet.io import save_json
from nndet.io.itk import load_sitk, load_sitk_as_array
from nndet.utils.info import maybe_verbose_iterable


def load_dicom_series_sitk(p):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(p))
    reader.SetFileNames(dicom_names)
    return reader.Execute()


def prepare_case(case_id,
                 data_dirs,
                 ktrans_dirs,
                 t2_masks,
                 df_labels,
                 df_masks,
                 data_target,
                 label_target,
                 ):
    try:
        logger.info(f"Preparing {case_id}")

        tmp_dir = data_dirs / case_id
        _dirs = [f for f in tmp_dir.iterdir() if f.is_dir()]
        assert len(_dirs) == 1
        data_dir = tmp_dir / _dirs[0]

        df_mask_case = df_masks[df_masks['T2'].str.contains(case_id)]
        assert len(df_mask_case) == 1

        t2_mask_file = df_mask_case.iloc[0]["T2"]
        assert f"{case_id}" in t2_mask_file
        t2_series_id = int(t2_mask_file.rsplit(".", 2)[0].rsplit('_', 1)[1])

        adc_mask_file = df_mask_case.iloc[0]["ADC"]
        assert f"{case_id}" in adc_mask_file
        if case_id == "ProstateX-0025":
            # case 0025 has a 7a inside the table
            adc_series_id = 7
            assert adc_mask_file.endswith("7a.nii.gz")
        elif case_id == "ProstateX-0113":
            # even though the table shows 9 as the series
            # ID we use 10 because 9 is not an ADC file?
            adc_series_id = int(adc_mask_file.rsplit(".", 2)[0].rsplit('_', 1)[1])
            assert adc_series_id == 9
            adc_series_id = 10
        else:
            adc_series_id = int(adc_mask_file.rsplit(".", 2)[0].rsplit('_', 1)[1])

        # T2
        t2_dir = [f for f in data_dir.glob("*t2*") if f.name.startswith(f"{t2_series_id}.")]
        assert len(t2_dir) == 1
        t2_data_itk = load_dicom_series_sitk(t2_dir[0])

        # ADC
        adc_dir = [f for f in data_dir.glob("*ADC*") if f.name.startswith(f"{adc_series_id}.")]
        assert len(adc_dir) == 1
        adc_data_itk = load_dicom_series_sitk(adc_dir[0])

        # PD-W
        pdw_dir = sorted(data_dir.glob("* PD *"))[-1]
        pdw_data_itk = load_dicom_series_sitk(pdw_dir)

        # k-trans
        ktrans_dir = ktrans_dirs / case_id
        ktrans_data_itk = load_sitk(ktrans_dir / f"{case_id}-Ktrans.mhd")

        # resample data to t2 (only early fusion is currently supported)
        resampler = sitk.ResampleImageFilter()  # default linear
        resampler.SetReferenceImage(t2_data_itk)
        adc_data_itk_res = resampler.Execute(adc_data_itk)
        pdw_data_itk_res = resampler.Execute(pdw_data_itk)
        ktrans_data_itk_res = resampler.Execute(ktrans_data_itk)

        # prepare mask
        mask_paths = list(t2_masks.glob(f"{case_id}*"))
        fids = [int([l for l in mp.name.split("-") if "Finding" in l][0][7:]) for mp in mask_paths]
        mask_itk = load_sitk(str(mask_paths[0]))
        mask = sitk.GetArrayFromImage(mask_itk)
        mask[mask > 0] = 1

        for idx, mp in enumerate(mask_paths[1:], start=2):
            _mask = load_sitk_as_array(str(mp))[0]
            mask[_mask > 0] = idx

        mask_final = sitk.GetImageFromArray(mask)
        mask_final.SetOrigin(t2_data_itk.GetOrigin())
        mask_final.SetDirection(t2_data_itk.GetDirection())
        mask_final.SetSpacing(t2_data_itk.GetSpacing())

        df_case = df_labels.loc[df_labels['ProxID'] == case_id]
        instances = {}
        for row in df_case.itertuples():
            if row.fid in fids:
                instances[fids.index(int(row.fid)) + 1] = int(row.ClinSig)
            else:
                logger.info(f"Found removed fid {row.fid} in {case_id}")

        # save
        sitk.WriteImage(t2_data_itk, str(data_target / f"{case_id}_0000.nii.gz"))
        sitk.WriteImage(adc_data_itk_res, str(data_target / f"{case_id}_0001.nii.gz"))
        sitk.WriteImage(pdw_data_itk_res, str(data_target / f"{case_id}_0002.nii.gz"))
        sitk.WriteImage(ktrans_data_itk_res, str(data_target / f"{case_id}_0003.nii.gz"))
        sitk.WriteImage(mask_final, str(label_target / f"{case_id}.nii.gz"))
        save_json({"instances": instances}, label_target / f"{case_id}.json")
    except Exception as e:
        logger.error(f"Case {case_id} failed with {e} and {traceback.format_exc()}")


@env_guard
def main():
    """
    Does not use the KTrans Sequence of ProstateX
    This script only uses the provided T2 masks
    """
    det_data_dir = Path(os.getenv('det_data'))
    task_data_dir = det_data_dir / "Task021_ProstateX"

    # setup raw paths
    source_data_dir = task_data_dir / "raw"
    if not source_data_dir.is_dir():
        raise RuntimeError(f"{source_data_dir} should contain the raw data but does not exist.")

    source_data = source_data_dir / "PROSTATEx"
    source_masks = source_data_dir / "rcuocolo-PROSTATEx_masks-e344452"
    source_ktrans = source_data_dir / "ktrains"
    csv_labels = source_data_dir / "ProstateX-TrainingLesionInformationv2" / "ProstateX-Findings-Train.csv"
    csv_masks = source_data_dir / "rcuocolo-PROSTATEx_masks-e344452" / "Files" / "Image_list.csv"

    data_target = task_data_dir / "raw_splitted" / "imagesTr"
    data_target.mkdir(parents=True, exist_ok=True)
    label_target = task_data_dir / "raw_splitted" / "labelsTr"
    label_target.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stdout, format="{level} {message}", level="INFO")
    logger.add(data_target.parent.parent / "prepare.log", level="DEBUG")

    base_masks = source_masks / "Files" / "Masks"
    t2_masks = base_masks / "T2"

    df_labels = pd.read_csv(csv_labels)
    df_masks = pd.read_csv(csv_masks)
    case_ids = [f.stem.split("-", 2)[:2] for f in t2_masks.glob("*nii.gz")]
    case_ids = list(set([f"{c[0]}-{c[1]}" for c in case_ids]))
    logger.info(f"Found {len(case_ids)} cases")

    # save meta
    logger.info("Saving dataset info")
    dataset_info = {
        "name": "ProstateX",
        "task": "Task021_ProstateX",

        "target_class": None,
        "test_labels": False,

        "labels": {
            "0": "clinically_significant",
            "1": "clinically_insignificant",
        },
        "modalities": {
            "0": "T2",
            "1": "ADC",
            "2": "PD-W",
            "3": "Ktrans"
        },
        "dim": 3,
        "info": "Ground Truth: T2 Masks; \n"
                "Modalities: T2, ADC, PD-W, Ktrans \n;"
                "Classes: clinically significant = 1, insignificant = 0 \n"
                "Keep: ProstateX-0025 '10-28-2011-MR prostaat kanker detectie WDSmc MCAPRODETW-19047'\n"
                "Masks\n"
                "https://github.com/rcuocolo/PROSTATEx_masks\n"
                "Github hash: e3444521e70cd5e8d405f4e9a6bc08312df8afe7"
    }
    save_json(dataset_info, task_data_dir / "dataset.json")

    # prepare labels and data
    for cid in maybe_verbose_iterable(case_ids):
        prepare_case(cid,
                     data_dirs=source_data,
                     ktrans_dirs=source_ktrans,
                     t2_masks=t2_masks,
                     df_labels=df_labels,
                     df_masks=df_masks,
                     data_target=data_target,
                     label_target=label_target,
                     )

    # with Pool(processes=6) as p:
    #     p.starmap(prepare_case, zip(case_ids,
    #                                 repeat(source_data),
    #                                 repeat(source_ktrans),
    #                                 repeat(t2_masks),
    #                                 repeat(df_labels),
    #                                 repeat(df_masks),
    #                                 repeat(data_target),
    #                                 repeat(label_target),
    #                                 ))

    # create test split
    create_test_split(task_data_dir / "raw_splitted",
                      num_modalities=len(dataset_info["modalities"]),
                      test_size=0.3,
                      random_state=0,
                      shuffle=True,
                      )


if __name__ == '__main__':
    main()
