import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pylidc as pl
import SimpleITK as sitk


def read_dcm_sitk(path_to_dcm: str) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path_to_dcm)
    reader.SetFileNames(dicom_names)
    img_itk = reader.Execute()
    return img_itk


def write_all_annotations_case():
    parser = argparse.ArgumentParser(
        description=("Write all annotation files for a single case in the LIDC-IDRI dataset.")
    )
    parser.add_argument("case_identifier", type=int, help="Case identifier")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument("--img_idx", type=int, help="Index of image", default=1, required=False)
    args = parser.parse_args()
    case_identifier = args.case_identifier
    output_dir = args.output_dir
    img_idx = args.img_idx

    output_dir.mkdir(parents=True, exist_ok=True)

    lidc_id = f"LIDC-IDRI-{case_identifier:04d}"
    scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == lidc_id)
    curr_img_idx = 1
    for scan_idx, scan in enumerate(scans):
        # iterate images
        if img_idx > 1:
            if curr_img_idx < img_idx:
                curr_img_idx += 1
                continue
        img_itk = read_dcm_sitk(scan.get_path_to_dicom_files())
        itk_shape = img_itk.GetSize()

        annotations = scan.annotations
        for i, ann in enumerate(annotations):
            mask_np = np.zeros(itk_shape, dtype=np.int32)
            mask_np[ann.bbox()][ann.boolean_mask()] = 1
            mask_np = mask_np.transpose(2, 0, 1)  # z, y, x

            mask_itk = sitk.GetImageFromArray(mask_np)
            mask_itk.CopyInformation(img_itk)
            sitk.WriteImage(
                mask_itk,
                str(
                    output_dir
                    / f"{lidc_id}_{curr_img_idx}_scanidx{scan_idx:02d}_ann{ann.id:03d}_scanid{ann.scan_id:04d}.nii.gz"
                ),
            )


if __name__ == "__main__":
    write_all_annotations_case()
