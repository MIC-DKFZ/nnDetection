# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import os
from abc import ABC, abstractclassmethod
from pathlib import Path
from typing import Any, Dict, Sequence, Type

from loguru import logger

from nndet.inference.ensembler.base import BaseEnsembler
from nndet.inference.helper import predict_dir
from nndet.inference.loading import get_loader_fn
from nndet.inference.predictor import Predictor
from nndet.inference.sweeper import Sweeper
from nndet.inference.transforms import Inference2D, get_tta_transforms
from nndet.ptmodule.module import LightningBaseModule


class PredictionMixin(ABC):
    @classmethod
    def requires_box_eval(cls) -> bool:
        """
        Returns:
            bool: `True` if box evaluation should be performed.
                `False` otherwise.
        """
        return False

    @classmethod
    def requires_mask_eval(cls) -> bool:
        """
        Returns:
            bool: `True` if mask evaluation should be performed.
                `False` otherwise.
        """
        return False

    @classmethod
    def requires_case_eval(cls) -> bool:
        """
        Returns:
            bool: `True` if case evaluation should be performed.
                `False` otherwise.
        """
        return False

    @classmethod
    def requires_seg_eval(cls) -> bool:
        """
        Returns:
            bool: `True` if (semantic) seg evaluation should be performed.
                `False` otherwise.
        """
        return False

    @abstractclassmethod
    def get_ensembler_cls(cls, dim: int) -> Type[BaseEnsembler]:
        """
        Returns:
            Type[BaseEnsembler]: return class of ensembler to use for this
                class
        """
        raise NotImplementedError

    @abstractclassmethod
    def get_sweeper_cls(cls) -> Type[Sweeper]:
        """
        Returns:
            Type[Sweeper]: return class of sweeper to use for this class
        """
        raise NotImplementedError

    def _get_detections_per_image(self, plan: Dict, **kwargs) -> Dict:
        """
        Get detections per image

        Args:
            plan: plan obtained from preprocessing
            kwargs: keyword arguments passed to get_predictor

        Returns:
            Dict: detections per image
        """
        raise NotImplementedError

    @classmethod
    def get_predictor(
        cls,
        plan: Dict,
        models: Sequence[LightningBaseModule],
        num_tta_transforms: int = None,
        # do_seg: bool = False,
        **kwargs,
    ) -> Predictor:
        """
        Create predictor

        Args:
            plan: plan obtained from preprocessing. Required keys:

                ``"patch_size"`` Sequence[int]
                    patch size to use for inference

                ``"batch_size"`` int
                    batch size to use for inference

                ``"network_dim"`` int
                    indicate dim of network -> 3 or 2

                ``"inference_plan"`` dict
                    parameters which were determined by sweep can be
                    provided here and will overwrite values from the
                    ensembler

            models: models to ensemble
            num_tta_transforms: number of tta transforms. If None, maximum
                number of tta transforms will be used. One of None | 0 | 4 | 8

        Returns:
            Predictor: instantiated predictor for inference
        """
        # process plan
        crop_size = plan["patch_size"]
        batch_size = plan["batch_size"]
        inference_plan = plan.get("inference_plan", {})
        logger.info(f"Found inference plan: {inference_plan} for prediction")
        if num_tta_transforms is None:
            num_tta_transforms = 8 if plan["network_dim"] == 3 else 4

        # setup
        tta_transforms, tta_inverse_transforms = get_tta_transforms(
            num_tta_transforms=num_tta_transforms,
            inverse_boxes=cls.requires_box_eval(),
            inverse_masks=cls.requires_mask_eval(),
            inverse_seg=cls.requires_seg_eval(),
        )
        logger.info(f"Using {len(tta_transforms)} tta transformations for prediction (one dummy trafo).")

        ensembler_cls = cls.get_ensembler_cls(dim=plan["network_dim"])
        _ensembler, _ensembler_key = ensembler_cls.constructor(parameters=inference_plan)
        ensembler = {_ensembler_key: _ensembler}

        predictor = Predictor(
            ensembler=ensembler,
            models=models,
            crop_size=crop_size,
            tta_transforms=tta_transforms,
            tta_inverse_transforms=tta_inverse_transforms,
            batch_size=batch_size,
            **kwargs,
        )
        if plan["network_dim"] == 2:
            predictor.pre_transform = Inference2D(["data"])
        return predictor

    def sweep(
        self,
        cfg: dict,
        save_dir: os.PathLike,
        train_data_dir: os.PathLike,
        case_ids: Sequence[str],
        run_prediction: bool = True,
        val_best: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Sweep detection parameters to find the best predictions

        Args:
            cfg: config used for training
            save_dir: save dir used for training
            train_data_dir: directory where preprocessed training/validation
                data is located
            case_ids: case identifies to prepare and predict
            run_prediction: predict cases
            val_best: use the best checkpoint (instead of the trainer_cfg's
                configured `sweep_ckpt`, default "last") to predict cases
            kwargs: keyword arguments passed to predict function

        Returns:
            Dict: inference plan. Exact parameter depend on current ensembler
                class.
        """
        logger.info(f"Running parameter sweep on {case_ids}")

        train_data_dir = Path(train_data_dir)
        preprocessed_dir = train_data_dir.parent
        processed_eval_labels = preprocessed_dir / "labelsTr"

        _save_dir = save_dir / "sweep"
        _save_dir.mkdir(parents=True, exist_ok=True)

        prediction_dir = save_dir / "sweep_predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)

        if run_prediction:
            logger.info("Predict cases with default settings...")
            if val_best:
                model_fn = get_loader_fn(mode=self.trainer_cfg.get("sweep_ckpt", "best"))
            else:
                model_fn = get_loader_fn(mode=self.trainer_cfg.get("sweep_ckpt", "last"))
            predict_dir(
                source_dir=train_data_dir,
                target_dir=prediction_dir,
                cfg=cfg,
                plan=self.plan,
                source_models=save_dir,
                num_models=1,
                num_tta_transforms=None,
                case_ids=case_ids,
                save_state=True,
                model_fn=model_fn,
                **kwargs,
            )

        logger.info("Start parameter sweep...")
        ensembler_cls = self.get_ensembler_cls(dim=self.plan["network_dim"])
        logger.info(f"Got ensembler class: {ensembler_cls.__name__} for sweep")
        sweeper_cls = self.get_sweeper_cls()
        logger.info(f"Got sweeper class: {sweeper_cls.__name__} for sweep")
        sweeper = sweeper_cls(
            classes=[item for _, item in cfg["data"]["labels"].items()],
            pred_dir=prediction_dir,
            gt_dir=processed_eval_labels,
            target_metric=self.sweep_key,
            ensembler_cls=ensembler_cls,
            save_dir=_save_dir,
        )
        inference_plan = sweeper.run_postprocessing_sweep()
        return inference_plan
