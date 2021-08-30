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
from typing import Iterable, Optional, List, Sequence, Type

import numpy as np
from loguru import logger

from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter

from nndet.io.augmentation import AUGMENTATION_REGISTRY
from nndet.io.datamodule import DATALOADER_REGISTRY
from nndet.io.augmentation.base import AugmentationSetup
from nndet.io.datamodule.base import BaseModule


class FixedLengthMultiThreadedAugmenter(MultiThreadedAugmenter):
    def __len__(self):
        return len(self.generator)


class FixedLengthSingleThreadedAugmenter(SingleThreadedAugmenter):
    def __len__(self):
        return len(self.data_loader)


def get_augmenter(dataloader,
                  transform,
                  num_processes: int,
                  num_cached_per_queue: int = 2,
                  multiprocessing: bool = True,
                  seeds: Optional[List[int]] = None,
                  pin_memory=True,
                  **kwargs,
                  ):
    """
    Wrapper to switch between multi-threaded and single-threaded augmenter
    """
    if multiprocessing: 
        logger.info(f"Using {num_processes} num_processes "
                    f"and {num_cached_per_queue} num_cached_per_queue for augmentation.")
        loader = FixedLengthMultiThreadedAugmenter(
            data_loader=dataloader,
            transform=transform,
            num_processes=num_processes,
            num_cached_per_queue=num_cached_per_queue,
            seeds=seeds,
            pin_memory=pin_memory,
            **kwargs,
            )
    else:
        loader = FixedLengthSingleThreadedAugmenter(
            data_loader=dataloader,
            transform=transform,
            **kwargs,
            )
    return loader


class Datamodule(BaseModule):
    def __init__(self,
                 plan: dict,
                 augment_cfg: dict,
                 data_dir: os.PathLike,
                 fold: int = 0,
                 **kwargs,
                 ):
        """
        Batchgenerator based datamodule

        Args:
            augment_cfg: provide settings for augmentation
                `splits_file` (str, optional): provide alternative splits file
                `oversample_foreground_percent` (float, optional):
                    ratio of foreground and background inside of batches,
                    defaults to 0.33
                `patch_size`(Sequence[int], optional): overwrite patch size
                `batch_size`(int, optional): overwrite patch size
            plan: current plan
            preprocessed_dir: path to base preprocessed dir
            data_dir: path to preprocessed data dir
            fold: current fold; if None, does not create folds and uses
                whole dataset for training and validation (don't do this ...
                except you know what you are doing :P)
        """
        super().__init__(
            plan=plan,
            augment_cfg=augment_cfg,
            data_dir=data_dir,
            fold=fold,
            **kwargs,
        )
        self.augmentation: Optional[Type[AugmentationSetup]] = None
        self.patch_size_generator: Optional[Sequence[int]] = None

    @property
    def patch_size(self):
        """
        Get patch size which can be (optionally) overwritten in the
        augmentation config
        """
        if "patch_size" in self.augment_cfg:
            ps = self.augment_cfg["patch_size"]
            logger.warning(f"Patch Size Overwrite Found: running patch size {ps}")
            return np.array(ps).astype(np.int32)
        else:
            return np.array(self.plan['patch_size']).astype(np.int32)

    @property
    def batch_size(self):
        """
        Get batch size which can be (optionally) overwritten in the
        augmentation config
        """
        if "batch_size" in self.augment_cfg:
            bs = self.augment_cfg["batch_size"]
            logger.warning(f"Batch Size Overwrite Found: running batch size {bs}")
            return bs
        else:
            return self.plan["batch_size"]

    @property
    def dataloader(self):
        """
        Get dataloader class name
        """
        return self.augment_cfg['dataloader'].format(self.plan["network_dim"])

    @property
    def dataloader_kwargs(self):
        """
        Get dataloader kwargs which can be (optionally) overwritten in the
        augmentation config
        """
        dataloader_kwargs = self.plan.get('dataloader_kwargs', {})
        if dl_kwargs := self.augment_cfg.get("dataloader_kwargs", {}):
            logger.warning(f"Dataloader Kwargs Overwrite Found: {dl_kwargs}")
            dataloader_kwargs.update(dl_kwargs)
        return dataloader_kwargs

    def setup(self, stage: Optional[str] = None):
        """
        Process augmentation configurations and plan to determine the
        patch size, the patch size for the generator and create the
        augmentation object.
        """
        dim = len(self.patch_size)
        params = self.augment_cfg["augmentation"]
        patch_size = self.patch_size

        if dim == 2:
            logger.info("Using 2D augmentation params")
            overwrites_2d = params.get("2d_overwrites", {})
            params.update(overwrites_2d)
        elif dim == 3 and self.plan['do_dummy_2D_data_aug']:
            logger.info("Using dummy 2d augmentation params")
            params["dummy_2D"] = True
            params["elastic_deform_alpha"] = params["2d_overwrites"]["elastic_deform_alpha"]
            params["elastic_deform_sigma"] = params["2d_overwrites"]["elastic_deform_sigma"]
            params["rotation_x"] = params["2d_overwrites"]["rotation_x"]

        params["selected_seg_channels"] = [0]
        params["use_mask_for_norm"] = self.plan['use_mask_for_norm']
        params["rotation_x"] = [i / 180 * np.pi for i in params["rotation_x"]]
        params["rotation_y"] = [i / 180 * np.pi for i in params["rotation_y"]]
        params["rotation_z"] = [i / 180 * np.pi for i in params["rotation_z"]] 

        augmentation_cls = AUGMENTATION_REGISTRY[params["transforms"]]
        self.augmentation = augmentation_cls(
            patch_size=patch_size,
            params=params,
        )
        self.patch_size_generator = self.augmentation.get_patch_size_generator()

        logger.info(f"Augmentation: {params['transforms']} transforms and "
                    f"{params.get('name', 'no_name')} params ")
        logger.info(f"Loading network patch size {self.augmentation.patch_size} "
                    f"and generator patch size {self.patch_size_generator}")

    def train_dataloader(self) -> Iterable:
        """
        Create training dataloader

        Returns:
            Iterable: dataloader for training
        """
        dataloader_cls = DATALOADER_REGISTRY.get(self.dataloader)
        logger.info(f"Using training {self.dataloader} with {self.dataloader_kwargs}")

        dl_tr = dataloader_cls(
            data=self.dataset_tr,
            batch_size=self.batch_size,
            patch_size_generator=self.patch_size_generator,
            patch_size_final=self.patch_size,
            oversample_foreground_percent=self.augment_cfg[
                "oversample_foreground_percent"],
            pad_mode="constant",
            num_batches_per_epoch=self.augment_cfg[
                "num_train_batches_per_epoch"],
            **self.dataloader_kwargs,
            )

        tr_gen = get_augmenter(
            dataloader=dl_tr,
            transform=self.augmentation.get_training_transforms(),
            num_processes=min(int(self.augment_cfg.get('num_threads', 12)), 16) - 1,
            num_cached_per_queue=self.augment_cfg.get('num_cached_per_thread', 2),
            multiprocessing=self.augment_cfg.get("multiprocessing", True),
            seeds=None,
            pin_memory=True,
            )
        logger.info("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())))
        return tr_gen

    def val_dataloader(self):
        """
        Create validation dataloader

        Returns:
            Iterable: dataloader for validation
        """
        dataloader_cls = DATALOADER_REGISTRY.get(self.dataloader)
        logger.info(f"Using validation {self.dataloader} with {self.dataloader_kwargs}")

        dl_val = dataloader_cls(
            data=self.dataset_val,
            batch_size=self.batch_size,
            patch_size_generator=self.patch_size,
            patch_size_final=self.patch_size,
            oversample_foreground_percent=self.augment_cfg[
                "oversample_foreground_percent"],
            pad_mode="constant",
            num_batches_per_epoch=self.augment_cfg[
                "num_val_batches_per_epoch"],
            **self.dataloader_kwargs,
            )

        val_gen = get_augmenter(
            dataloader=dl_val,
            transform=self.augmentation.get_validation_transforms(),
            num_processes=min(int(self.augment_cfg.get('num_threads', 12)), 16) - 1,
            num_cached_per_queue=self.augment_cfg.get('num_cached_per_thread', 2),
            multiprocessing=self.augment_cfg.get("multiprocessing", True),
            seeds=None,
            pin_memory=True,
            )
        logger.info("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())))
        return val_gen
