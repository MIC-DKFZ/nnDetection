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
from sklearn.model_selection import KFold

from nndet.io.load import load_pickle, save_pickle, save_txt
from nndet.io.utils import load_dataset_id


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

                ``"patch_size"`` Sequence[int], optional
                    overwrite patch size

                ``"batch_size"`` int, optional
                    overwrite patch size

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
        self.fold = fold
        self.log_aug = log_aug

        self.preprocessed_dir = self.data_dir.parent.parent
        self.splits_file = self.io_cfg.get("splits", "splits_final")

        self.dataset_tr = {}
        self.dataset_val = {}
        self.dataset = load_dataset_id(self.data_dir)
        self.do_split()

    @property
    def splits_file(self) -> str:
        return self._splits_file

    @splits_file.setter
    def splits_file(self, f: str) -> None:
        if f != "splits_final":
            logger.warning(f"Found splits_file overwrite: {f}")

        if f.endswith("pkl"):
            self._splits_file = f
        else:
            self._splits_file = f + ".pkl"

    @property
    def patch_size(self) -> np.ndarray:
        """
        Get patch size which can be (optionally) overwritten in the
        io config
        """
        if "patch_size" in self.io_cfg:
            ps = self.io_cfg["patch_size"]
            logger.warning(f"Patch Size Overwrite Found: running patch size {ps}")
            return np.array(ps).astype(np.int32)
        else:
            return np.array(self.plan["patch_size"]).astype(np.int32)

    @property
    def batch_size(self) -> int:
        """
        Get batch size which can be (optionally) overwritten in the
        io config
        """
        if "batch_size" in self.io_cfg:
            bs = self.io_cfg["batch_size"]
            logger.warning(f"Batch Size Overwrite Found: running batch size {bs}")
            return bs
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
        splits_file = self.preprocessed_dir / self.splits_file

        if not splits_file.is_file():
            self.create_new_split(splits_file)
        logger.info(f"Using splits {splits_file} with fold {self.fold}")
        splits = load_pickle(splits_file)

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
                f"in split file and will be ignored: {_dataset}"
            )

    def create_new_split(self, splits_file: Path) -> None:
        """
        Create a new 5 fold split with a fixed seed

        Args:
            splits_file: path where splits file should be saved
        """
        logger.info("Creating new split...")
        splits = []
        all_keys_sorted = np.sort(list(self.dataset.keys()))

        kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
        for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):

            train_keys = np.array(all_keys_sorted)[train_idx]
            test_keys = np.array(all_keys_sorted)[test_idx]

            splits.append(OrderedDict())
            splits[-1]["train"] = train_keys
            splits[-1]["val"] = test_keys
        save_pickle(splits, splits_file)
