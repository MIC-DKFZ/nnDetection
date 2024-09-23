# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Hashable, Sequence, Type

from loguru import logger

from nndet.inference.ensembler.base import BaseEnsembler
from nndet.inference.ensembler.detection import (
    BoxEnsemblerSelective,
    BoxEnsemblerSelective2D,
    BoxEnsemblerSelectiveAsymNMS,
)
from nndet.inference.ensembler.segmentation import SegmentationEnsembler
from nndet.inference.predictor import Predictor
from nndet.inference.sweeper import BoxSweeper, Sweeper
from nndet.inference.transforms import Inference2D, get_tta_transforms
from nndet.ptmodule.mixins.prediction.base import PredictionMixin
from nndet.ptmodule.module import LightningBaseModule


class BoxPredictionMixin(PredictionMixin):
    @classmethod
    def requires_box_eval(cls) -> bool:
        return True

    @classmethod
    def requires_case_eval(cls) -> bool:
        return True

    @classmethod
    def get_ensembler_cls(cls, dim: int) -> Type[BaseEnsembler]:
        """
        Returns:
            Type[BaseEnsembler]: return class of ensembler to use for this
                class
        """
        if dim == 2:
            return BoxEnsemblerSelective2D
        elif dim == 3:
            return BoxEnsemblerSelective
        else:
            raise ValueError(f"Dim {dim} not supported in get_ensembler_cls.")

    @classmethod
    def get_sweeper_cls(cls) -> Type[Sweeper]:
        return BoxSweeper

    @classmethod
    def _get_ensembler_cls(cls, key: Hashable, dim: int) -> Type[BaseEnsembler]:
        """
        This is used internally to exchange the ensembler classes
        for experimentation but should not be used from outside.
        """
        if key == "boxes":
            return cls.get_ensembler_cls(dim=dim)
        elif key == "seg":
            return SegmentationEnsembler
        else:
            raise ValueError(f"Key {key} not supported in _get_ensembler_cls.")

    @classmethod
    def get_predictor(
        cls,
        plan: Dict,
        models: Sequence[LightningBaseModule],
        num_tta_transforms: int = None,
        do_seg: bool = False,
        **kwargs,
    ) -> Predictor:
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
            inverse_seg=(cls.requires_seg_eval() or do_seg),
        )
        logger.info(f"Using {len(tta_transforms)} tta transformations for prediction (one dummy trafo).")

        ensembler_cls = cls.get_ensembler_cls(dim=plan["network_dim"])
        _ensembler, _ensembler_key = ensembler_cls.constructor(parameters=inference_plan)
        ensembler = {_ensembler_key: _ensembler}

        if do_seg:
            seg_ensembler_cls = cls._get_ensembler_cls(
                key="seg",
                dim=plan["network_dim"],
            )
            seg_ensembler, seg_ensembler_key = seg_ensembler_cls.constructor()
            ensembler[seg_ensembler_key] = seg_ensembler

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


class BoxPredictionMixinV2(BoxPredictionMixin):
    @classmethod
    def _get_detections_per_image(cls, plan: dict) -> int:
        """
        Heuristic to compute the number of predictions of the model for a
        single image

        Args:
            plan: plan for model and dataset

        Returns:
            int: number of detections for model
        """
        instances_image = plan["architecture"]["instances_img"]["perc95"]
        return max(1000, 10 * instances_image)  # use a conversative topk value here

    @classmethod
    def get_predictor(
        cls,
        plan: Dict,
        models: Sequence[LightningBaseModule],
        num_tta_transforms: int = None,
        do_seg: bool = False,
        **kwargs,
    ) -> Predictor:
        # process plan
        crop_size = plan["patch_size"]
        batch_size = plan["batch_size"]
        inference_plan = plan.get("inference_plan", {})
        det_per_image = cls._get_detections_per_image(plan)
        for k in ["model_topk", "ensemble_topk", "model_detections_per_image"]:
            if k in inference_plan:
                logger.warning(f"Overwriting {k} in inference plan with value {det_per_image}.")
            inference_plan[k] = det_per_image
        logger.info(f"Found inference plan: {inference_plan} for prediction")
        if num_tta_transforms is None:
            num_tta_transforms = 8 if plan["network_dim"] == 3 else 4

        # setup
        tta_transforms, tta_inverse_transforms = get_tta_transforms(
            num_tta_transforms=num_tta_transforms,
            inverse_boxes=cls.requires_box_eval(),
            inverse_masks=cls.requires_mask_eval(),
            inverse_seg=(cls.requires_seg_eval() or do_seg),
        )
        logger.info(f"Using {len(tta_transforms)} tta transformations for prediction (one dummy trafo).")

        ensembler_cls = cls.get_ensembler_cls(dim=plan["network_dim"])
        _ensembler, _ensembler_key = ensembler_cls.constructor(parameters=inference_plan)
        ensembler = {_ensembler_key: _ensembler}

        if do_seg:
            seg_ensembler_cls = cls._get_ensembler_cls(
                key="seg",
                dim=plan["network_dim"],
            )
            seg_ensembler, seg_ensembler_key = seg_ensembler_cls.constructor()
            ensembler[seg_ensembler_key] = seg_ensembler

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


class BoxPredictionMixinV3(BoxPredictionMixin):
    @classmethod
    def get_ensembler_cls(cls, dim: int) -> Type[BaseEnsembler]:
        """
        Returns:
            Type[BaseEnsembler]: return class of ensembler to use for this
                class
        """
        if dim == 3:
            return BoxEnsemblerSelectiveAsymNMS
        else:
            raise ValueError(f"Dim {dim} not supported in get_ensembler_cls.")
