"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import shutil
import json
from itertools import repeat
from multiprocessing import Pool

import SimpleITK as sitk
import numpy as np
from loguru import logger
from pathlib import Path
from typing import Sequence, Union

from nndet.io.itk import load_sitk_as_array, load_sitk
from nndet.io.load import save_json, load_json
from nndet.io.paths import get_case_ids_from_dir
from nndet.io.transforms.instances import instances_to_segmentation_np

Pathlike = Union[str, Path]


class Exporter:
    """
    Helper to export datasets to nnunet
    """

    def __init__(self,
                 data_info: dict,
                 tr_image_dir: Pathlike,
                 label_dir: Pathlike,
                 target_dir: Pathlike,
                 ts_image_dir: Pathlike = None,
                 export_stuff: bool = False,
                 processes: int = 6,
                 ):
        """
        Args:
            data_info: dataset information. See :method:`export_dataset_info`.
                Required keys: `modality`, `labels`
            tr_image_dir: training data dir
            label_dir: label data dir
            target_dir: target directory
            ts_image_dir: test data dir
            export_stuff: export stuff segmentations
        """
        self.data_info = data_info
        self.tr_image_dir = Path(tr_image_dir)
        self.label_dir = Path(label_dir)
        self.target_dir = Path(target_dir)
        self.export_stuff = export_stuff
        self.processes = processes
        if ts_image_dir is not None:
            self.ts_image_dir = Path(ts_image_dir)
        else:
            self.ts_image_dir = None

    def export(self):
        """
        Export entire dataset
        """
        self.export_images()
        self.export_labels()
        self.export_dataset_info()

    def export_images(self):
        """
        Export images
        """
        # data can be copied directly
        for img_dir in [self.tr_image_dir, self.ts_image_dir]:
            if img_dir is None:
                continue
            image_target_dir = self.target_dir / img_dir.stem
            logger.info(f"Copy data from {img_dir} to {image_target_dir}")
            shutil.copytree(img_dir, image_target_dir)

    def export_labels(self):
        """
        Export labels
        """
        case_ids = get_case_ids_from_dir(self.label_dir, remove_modality=False, pattern="*.json")
        label_target_dir = self.target_dir / self.label_dir.stem
        label_target_dir.mkdir(exist_ok=True, parents=True)
        num_classes = len(self.data_info.get("labels", {}))
        if num_classes == 0:
            logger.warning(f"Did not find any fg classes.")

        logger.info(f"Found {len(case_ids)} to process.")
        logger.info(f"Export stuff: {self.export_stuff}")
        if self.processes == 0:
            logger.info("Using for loop to export labels")
            for cid in case_ids:
                self._export_label(cid, num_classes, label_target_dir)
        else:
            logger.info(f"Using pool with {self.processes} processes to export labels")
            with Pool(processes=self.processes) as p:
                p.starmap(self._export_label, zip(
                    case_ids, repeat(num_classes), repeat(label_target_dir)))
        assert len(get_case_ids_from_dir(
            label_target_dir, remove_modality=False, pattern="*.nii.gz")) == len(case_ids)

    def _export_label(self, cid: str, num_classes: int, target_dir: Path):
        logger.info(f"Processing {cid}")
        meta = load_json(self.label_dir / f"{cid}.json")
        instance_seg_itk = sitk.ReadImage(str(self.label_dir / f"{cid}.nii.gz"))
        instance_seg = sitk.GetArrayFromImage(instance_seg_itk)

        if np.any(np.isnan(instance_seg)):
            logger.error(f"FOUND NAN IN {cid} LABEL")
        
        # instance classes start form 0 which is background in nnUNet
        seg = instances_to_segmentation_np(instance_seg,
                                           meta["instances"],
                                           add_background=True,
                                           )
        if num_classes > 0:
            assert seg.max() <= num_classes, "Wrong class id, something went wrong."
        if instance_seg.max() > 0:
            assert seg.max() > 0, "Instance got lost, something went wrong"
        assert np.all((instance_seg > 0) == (seg > 0)), "Something wrong with foreground"
        assert np.all((instance_seg == 0) == (seg == 0)), "Something wrong with background"

        if self.export_stuff:
            # map stuff classes to: max(labels) + stuff_cls
            stuff_seg = load_sitk_as_array(self.label_dir / f"{cid}_stuff.nii.gz")[0]
            for i in range(1, stuff_seg.max() + 1):
                seg[stuff_seg == i] = num_classes + i

        seg_itk = sitk.GetImageFromArray(seg)
        spacing = instance_seg_itk.GetSpacing()
        seg_itk.SetSpacing(spacing)
        origin = instance_seg_itk.GetOrigin()
        seg_itk.SetOrigin(origin)
        direction = instance_seg_itk.GetDirection()
        seg_itk.SetDirection(direction)        
        sitk.WriteImage(seg_itk, str(target_dir / f"{cid}.nii.gz"))

    def export_dataset_info(self):
        """
        Export dataset settings (dataset.json for nnunet)
        """
        self.target_dir.mkdir(exist_ok=True, parents=True)
        dataset_info = {}
        dataset_info["name"] = self.data_info.get("name", "unknown")
        dataset_info["description"] = self.data_info.get("description", "unknown")
        dataset_info["reference"] = self.data_info.get("reference", "unknown")
        dataset_info["licence"] = self.data_info.get("licence", "unknown")
        dataset_info["release"] = self.data_info.get("release", "unknown")
        min_size = self.data_info.get("min_size", 0)
        min_vol = self.data_info.get("min_vol", 0)
        dataset_info["prep_info"] = f"min size: {min_size} ; min vol {min_vol}"

        dataset_info["tensorImageSize"] = f"{self.data_info.get('dim', 3)}D"
        # dataset_info["tensorImageSize"] = self.data_info.get("tensorImageSize", "4D")
        dataset_info["modality"] = self.data_info.get("modalities", {})
        if not dataset_info["modality"]:
            logger.error("Did not find any modalities for dataset")

        # +1 for seg classes because of background
        dataset_info["labels"] = {"0": "background"}
        instance_classes = self.data_info.get("labels", {})
        if not instance_classes:
            logger.error("Did not find any labels of dataset")
        for _id, _class in instance_classes.items():
            seg_id = int(_id) + 1
            dataset_info["labels"][str(seg_id)] = _class

        if self.export_stuff:
            stuff_classes = self.data_info.get("labels_stuff", {})
            num_instance_classes = len(instance_classes)
            # copy stuff classes into nnuent dataset.json
            stuff_classes = {
                str(int(key) + num_instance_classes): item
                for key, item in stuff_classes.items() if int(key) > 0
            }
            dataset_info["labels_stuff"] = stuff_classes
            dataset_info["labels"].update(stuff_classes)

        _case_ids = get_case_ids_from_dir(self.label_dir, remove_modality=False)
        case_ids_tr = get_case_ids_from_dir(self.tr_image_dir, remove_modality=True)
        assert len(set(_case_ids).union(case_ids_tr)) == len(_case_ids), "All training  images need a label"
        dataset_info["numTraining"] = len(case_ids_tr) 

        dataset_info["training"] = [
            {"image": f"./imagesTr/{cid}.nii.gz", "label": f"./labelsTr/{cid}.nii.gz"}
            for cid in case_ids_tr]

        if self.ts_image_dir is not None:
            case_ids_ts = get_case_ids_from_dir(self.ts_image_dir, remove_modality=True)
            dataset_info["numTest"] = len(case_ids_ts)
            dataset_info["test"] = [f"./imagesTs/{cid}.nii.gz" for cid in case_ids_ts]
        else:
            dataset_info["numTest"] = 0
            dataset_info["test"] = []
        save_json(dataset_info, self.target_dir / "dataset.json")
