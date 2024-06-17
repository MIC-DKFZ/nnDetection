# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import copy
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
from loguru import logger

from nndet.io.load import save_txt
from nndet.io.utils import load_dataset_id
from nndet.utils.config import load_splits_from_task

# from sklearn.model_selection import KFold


class BaseModule(pl.LightningDataModule):
    def __init__(
        self,
        plan: dict,
        io_cfg: dict,
        augment_cfg: dict,
        data_dir: os.PathLike,
        fold: int = 0,
        log_aug: bool = False,
        **kwargs,
    ):
        """
        Baseclass for nnDetection data nodules.
        Overwrite `setup` to customize the bahvior.
        The splits are created iniside the init because we

        Args:
            plan: plan file
            io_cfg: Input/Output configuration

                ``"splits"`` str, optional
                    provide alternative splits file

                ``"oversample_foreground_percent"`` float, optional
                    ratio of foreground and background inside of batches,
                    defaults to 0.33

            augment_cfg: provide settings for augmentation
            data_dir: path to preprocessed data dir. Needs to follow:
                `.../preprocessed/[data_identifier]/imagesTr`
            fold: current fold
        """
        super().__init__(**kwargs)
        self.plan = plan
        self.io_cfg = io_cfg
        self.augment_cfg = augment_cfg
        self.data_dir = Path(data_dir)
        self.label_dir = self.get_label_dir(self.data_dir)
        self.fold = fold
        self.log_aug = log_aug

        self.preprocessed_dir = self.data_dir.parent.parent
        self.splits_file = self.io_cfg.get("splits", "splits_final")

        self.dataset_tr = {}
        self.dataset_val = {}
        self.dataset = load_dataset_id(self.data_dir, self.plan['preprocessed_data_format'], self.label_dir)
        self.do_split()

    @staticmethod
    def get_label_dir(data_dir: os.PathLike) -> Path:
        _data_dir = Path(data_dir)
        data_name = _data_dir.name

        if data_name == "imagesTr":
            label_name = "labelsTr"
            assert (_data_dir.parent / label_name).is_dir()
        elif data_name == "imagesTs":
            label_name = "labelsTs"
        else:
            raise RuntimeWarning(f"Wasn't able to retrieve label dir from {_data_dir}.")
        return _data_dir.parent / label_name

    @property
    def splits_file(self) -> str:
        return self._splits_file

    @splits_file.setter
    def splits_file(self, f: str) -> None:
        if f != "splits_final":
            logger.warning(f"Found splits_file overwrite: {f}")

        if f.endswith("pkl"):
            self._splits_file = f[:-4]
        else:
            self._splits_file = f

    @property
    def patch_size(self) -> np.ndarray:
        """
        Get patch size which can be (optionally) overwritten in the
        io config
        """
        if "patch_size" in self.io_cfg:
            logger.error("Patch Size Overwrite Found: this is not supported anymore. Simply edit the plan file.")
            raise NotImplementedError("Patch size overwrite not supported anymore")
        else:
            return np.array(self.plan["patch_size"]).astype(np.int32)

    @property
    def batch_size(self) -> int:
        """
        Get batch size which can be (optionally) overwritten in the
        io config
        """
        if "batch_size" in self.io_cfg:
            logger.error("Batch Size Overwrite Found: this is not supported anymore. Simply edit the plan file.")
            raise NotImplementedError("Batch size overwrite not supported anymore")
        else:
            return self.plan["batch_size"]

    def log_augmentation(self, pipeline: Any) -> None:
        """
        Log augmentation pipeline into file and logger
        """
        pipeline_str = f"+++ Augmentation Pipeline +++ \n\n{str(pipeline)}"
        Path("./augmentation.txt").unlink(missing_ok=True)
        save_txt(pipeline_str, "./augmentation")

        if self.log_aug:
            logger.info(pipeline_str)

    def do_split(self) -> None:
        """
        Load a datasplit.
        If not split is found, a new split is created.
        Results are saved into :attr:`dataset_tr` and :attr:`dataset_val`
        """
        logger.info(f"Using splits {self.splits_file} with fold {self.fold}")
        splits = load_splits_from_task(self.splits_file, self.preprocessed_dir.parent.name)

        if self.fold is None:
            raise RuntimeError("Not supported anymore, remove this on own risk")
            logger.warning("USING SAME TRAIN AND VAL SET")
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            tr_keys = splits[self.fold]["train"]
            val_keys = splits[self.fold]["val"]

        tr_keys.sort()
        val_keys.sort()
        _dataset = copy.deepcopy(self.dataset)

        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = _dataset.pop(i)

        self.dataset_val = OrderedDict()
        for j in val_keys:
            self.dataset_val[j] = _dataset.pop(j)
        if len(_dataset) > 0:
            logger.error(
                "IMPORTANT: Found data samples which are not present "
                f"in split file and will be ignored: {list(_dataset.keys())}"
            )
