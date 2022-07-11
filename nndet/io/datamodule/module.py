# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import os
import random
from abc import abstractstaticmethod
from typing import Dict, Iterable, List, Optional, Sequence, Type

import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)
from loguru import logger

from nndet.io.augmentation import AUGMENTATION_REGISTRY
from nndet.io.augmentation.base import AugmentationSetup
from nndet.io.datamodule import DATALOADER_REGISTRY
from nndet.io.datamodule.base import BaseModule


class FixedLengthSingleThreadedAugmenter(SingleThreadedAugmenter):
    def __len__(self):
        return len(self.data_loader)


class FixedLengthMultiThreadedAugmenter(MultiThreadedAugmenter):
    def __len__(self):
        return len(self.generator)


class TransformWrapper(torch.utils.data.Dataset):
    def __init__(self, bgloader, transform) -> None:
        super().__init__()
        self.loader = bgloader
        self.transform = transform

    def __getitem__(self, index) -> Dict:
        return self.transform(**self.loader.generate_train_batch())

    def __len__(self):
        return len(self.loader)


def seed_worker(worker_id):
    """
    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    to fix https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def skipped_collate_fn(batch):
    assert len(batch) == 1
    return batch[0]


import subprocess


# TODO: remove this! do something different
def get_allowed_n_proc_DA():
    dnt = os.getenv("det_num_threads", None)
    if dnt is None:
        hostname = subprocess.getoutput(["hostname"])
        if hostname in ["hdf19-gpu16", "hdf19-gpu17", "e230-AMDworkstation"]:
            return 16
        if hostname.startswith("hdf19-gpu") or hostname.startswith("e071-gpu"):
            return 12
        elif hostname.startswith("e230-dgx1"):
            return 10
        elif hostname.startswith("hdf18-gpu") or hostname.startswith("e132-comp"):
            return 16
        elif hostname.startswith("e230-dgx2"):
            return 6
        elif hostname.startswith("e230-dgxa100-") or hostname.startswith("lsf22-"):
            return 24  # max 32
        else:
            raise RuntimeError(
                f"Could not determine det_num_threads, env: {dnt} hostname {hostname}"
            )
    else:
        return int(dnt)


class BaseDatamodule(BaseModule):
    def __init__(
        self,
        plan: dict,
        io_cfg: dict,
        augment_cfg: dict,
        data_dir: os.PathLike,
        fold: int = 0,
        **kwargs,
    ):
        """
        Batchgenerator based datamodule

        Args:
            augment_cfg: provide settings for augmentation
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

            plan: current plan
            preprocessed_dir: path to base preprocessed dir
            data_dir: path to preprocessed data dir
            fold: current fold

        Warnings:
            `fold=None` was deperacated to prevent wrong usage, it is
            possible to enable it by uncommenting the raised error.
        """
        super().__init__(
            plan=plan,
            io_cfg=io_cfg,
            augment_cfg=augment_cfg,
            data_dir=data_dir,
            fold=fold,
            **kwargs,
        )
        self.augmentation: Optional[Type[AugmentationSetup]] = None
        self.patch_size_generator: Optional[Sequence[int]] = None

    @property
    def dataloader(self):
        """
        Get dataloader class name
        """
        return self.io_cfg["dataloader"].format(self.plan["network_dim"])

    @property
    def dataloader_kwargs(self):
        """
        Get dataloader kwargs which can be (optionally) overwritten in the
        io config
        """
        dataloader_kwargs = self.plan.get("dataloader_kwargs", {})
        if dl_kwargs := self.io_cfg.get("dataloader_kwargs", {}):
            # logger.warning(f"Dataloader Kwargs Overwrite Found: {dl_kwargs}")
            dataloader_kwargs.update(dl_kwargs)
        return dataloader_kwargs

    def setup(self, stage: Optional[str] = None):
        """
        Process augmentation configurations and plan to determine the
        patch size, the patch size for the generator and create the
        augmentation object.
        """
        dim = len(self.patch_size)
        params = self.augment_cfg
        patch_size = self.patch_size

        if dim == 2:
            logger.info("Using 2D augmentation params")
            overwrites_2d = params.get("2d_overwrites", {})
            params.update(overwrites_2d)
        elif dim == 3 and self.plan["do_dummy_2D_data_aug"]:
            logger.info("Using dummy 2d augmentation params")
            params["dummy_2D"] = True
            params["elastic_deform_alpha"] = params["2d_overwrites"][
                "elastic_deform_alpha"
            ]
            params["elastic_deform_sigma"] = params["2d_overwrites"][
                "elastic_deform_sigma"
            ]
            params["rotation_x"] = params["2d_overwrites"]["rotation_x"]

        params["selected_seg_channels"] = [0]
        params["use_mask_for_norm"] = self.plan["use_mask_for_norm"]
        params["rotation_x"] = [i / 180 * np.pi for i in params["rotation_x"]]
        params["rotation_y"] = [i / 180 * np.pi for i in params["rotation_y"]]
        params["rotation_z"] = [i / 180 * np.pi for i in params["rotation_z"]]

        augmentation_cls = AUGMENTATION_REGISTRY[params["transforms"]]
        self.augmentation = augmentation_cls(
            patch_size=patch_size,
            params=params,
        )
        self.patch_size_generator = self.augmentation.get_patch_size_generator()

        logger.info(
            f"Augmentation: {params['transforms']} transforms and "
            f"{params.get('name', 'no_name')} params "
        )
        logger.info(
            f"Loading network patch size {self.augmentation.patch_size} "
            f"and generator patch size {self.patch_size_generator}"
        )

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
            oversample_foreground_percent=self.io_cfg["oversample_foreground_percent"],
            pad_mode="constant",
            num_batches_per_epoch=self.io_cfg["num_train_batches_per_epoch"],
            **self.dataloader_kwargs,
        )
        tr_transforms = self.augmentation.get_training_transforms()
        self.log_augmentation(tr_transforms)
        tr_gen = self.get_augmenter(
            dataloader=dl_tr,
            transform=tr_transforms,
            # num_processes=min(int(self.io_cfg.get('num_threads', 12)), 16) - 1,
            num_processes=get_allowed_n_proc_DA(),
            num_cached_per_queue=self.io_cfg.get("num_cached_per_thread", 2),
            multiprocessing=self.io_cfg.get("multiprocessing", True),
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
            oversample_foreground_percent=self.io_cfg["oversample_foreground_percent"],
            pad_mode="constant",
            num_batches_per_epoch=self.io_cfg["num_val_batches_per_epoch"],
            **self.dataloader_kwargs,
        )
        val_transforms = self.augmentation.get_validation_transforms()
        val_gen = self.get_augmenter(
            dataloader=dl_val,
            transform=val_transforms,
            # num_processes=min(int(self.io_cfg.get('num_threads', 12)), 16) - 1,
            num_processes=get_allowed_n_proc_DA(),
            num_cached_per_queue=self.io_cfg.get("num_cached_per_thread", 2),
            multiprocessing=self.io_cfg.get("multiprocessing", True),
            seeds=None,
            pin_memory=True,
        )
        logger.info("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())))
        return val_gen

    @abstractstaticmethod
    def get_augmenter(
        dataloader,
        transform,
        num_processes: int,
        num_cached_per_queue: int = 2,
        multiprocessing: bool = True,
        seeds: Optional[List[int]] = None,
        pin_memory=True,
        **kwargs,
    ):
        """
        Provide an interface to wrap the dataset (Pt naming) with a
        dataloader (Pt naming)
        """
        raise NotImplementedError


class BgDatamodule(BaseDatamodule):
    @staticmethod
    def get_augmenter(
        dataloader,
        transform,
        num_processes: int,
        num_cached_per_queue: int = 2,
        multiprocessing: bool = True,
        seeds: Optional[List[int]] = None,
        pin_memory=True,
        **kwargs,
    ):
        """
        Provide an interface to wrap the dataset (Pt naming) with a
        dataloader (Pt naming)
        """
        if multiprocessing:
            logger.info(
                f"Using Batchgenerators with {num_processes} num_processes "
                f"and {num_cached_per_queue} num_cached_per_queue for augmentation."
            )
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


class PtDatamodule(BaseDatamodule):
    @staticmethod
    def get_augmenter(
        dataloader,
        transform,
        num_processes: int,
        num_cached_per_queue: int = 2,
        multiprocessing: bool = True,
        seeds: Optional[List[int]] = None,
        pin_memory=True,
        **kwargs,
    ):
        """
        Provide an interface to wrap the dataset (Pt naming) with a
        dataloader (Pt naming)
        """
        if not multiprocessing:
            num_processes = 0
            persistent_workers = False
        else:
            persistent_workers = True

        logger.info(
            f"Using PyTorch with {num_processes} num_processes "
            f"and {num_cached_per_queue} num_cached_per_queue for augmentation."
        )
        wrapped_loader = TransformWrapper(dataloader, transform=transform)

        if not torch.distributed.is_initialized():
            s = 0
        else:
            s = torch.distributed.get_rank()
            logger.info(f"Found Torch Distributed: Using local rank for seed {s}")
        g = torch.Generator()
        g.manual_seed(s)

        ptloader = torch.utils.data.DataLoader(
            dataset=wrapped_loader,
            num_workers=num_processes,
            batch_size=1,  # dataset provides batches so use batch size 1 here
            pin_memory=pin_memory,
            drop_last=False,
            worker_init_fn=seed_worker,
            prefetch_factor=num_cached_per_queue,
            collate_fn=skipped_collate_fn,
            persistent_workers=persistent_workers,
            generator=g,
            **kwargs,
        )
        return ptloader
