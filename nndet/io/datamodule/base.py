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

import os
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pytorch_lightning as pl
from loguru import logger
from sklearn.model_selection import KFold

from nndet.io.utils import load_dataset_id
from nndet.io.load import load_pickle, save_pickle


class BaseModule(pl.LightningDataModule):
    def __init__(self,
                 plan: dict,
                 augment_cfg: dict,
                 data_dir: os.PathLike,
                 fold: int = 0,
                 **kwargs,
                 ):
        """
        Baseclass for nnDetection data nodules.
        Overwrite :method:`setup` to customize the bahvior.
        The splits are created iniside the init because we 

        Args:
            plan: plan file
            augment_cfg: provide settings for augmentation
                `splits_file` (str, optional): provide alternative splits file
            data_dir: path to preprocessed data dir. Needs to follow:
                `.../preprocessed/[data_identifier]/imagesTr
            fold: current fold; if None, does not create folds and uses
                whole dataset for training and validation (don't do this ...
                except you know what you are doing :P)
        """
        super().__init__(**kwargs)
        self.plan = plan
        self.augment_cfg = augment_cfg
        self.data_dir = Path(data_dir)
        self.fold = fold

        self.preprocessed_dir = self.data_dir.parent.parent
        
        if "splits" in self.augment_cfg:
            self.splits_file = self.augment_cfg["splits"]
        elif "splits_final" in self.augment_cfg:
            self.splits_file = self.augment_cfg["splits_final"]
        else:
            self.splits_file = "splits_final"

        self.dataset_tr = {}
        self.dataset_val = {}
        self.dataset = load_dataset_id(self.data_dir)
        self.do_split()

    @property
    def splits_file(self) -> str:
        return self._splits_file

    @splits_file.setter
    def splits_file(self, f: str) -> None:
        if f.endswith("pkl"):
            self._splits_file = f
        else:
            self._splits_file = f + ".pkl"

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
            logger.warning(f"USING SAME TRAIN AND VAL SET")
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            tr_keys = splits[self.fold]['train']
            val_keys = splits[self.fold]['val']

        tr_keys.sort()
        val_keys.sort()

        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]

        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

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
            splits[-1]['train'] = train_keys
            splits[-1]['val'] = test_keys
        save_pickle(splits, splits_file)
