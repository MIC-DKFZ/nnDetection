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

from __future__ import annotations

import os
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, Hashable, Sequence

import numpy as np
import torch
from loguru import logger

from nndet.arch.blocks.basic import StackedConvBlock2
from nndet.arch.conv import ConvGroupRelu, ConvInstanceRelu, Generator
from nndet.arch.decoder.base import DecoderType, UFPNModular
from nndet.arch.encoder.abstract import EncoderType
from nndet.arch.encoder.modular import Encoder
from nndet.arch.heads.classifier import CEClassifier, DenseClassifierType
from nndet.arch.heads.comb import AnchorHeadType, BoxHeadHNM
from nndet.arch.heads.regressor import DenseRegressorType, L1Regressor
from nndet.arch.heads.regressor.dense_single import DenseRegressor
from nndet.arch.heads.segmenter import DiCESegmenter, SegmenterType
from nndet.core.boxes.anchors import AnchorGeneratorType, get_anchor_generator
from nndet.core.boxes.coder import BoxCoderND, CoderType
from nndet.core.boxes.matcher import IoUMatcher
from nndet.core.boxes.ops import box_iou
from nndet.core.boxes.sampler import HardNegativeSamplerBatched
from nndet.core.retina import BaseRetinaNet
from nndet.evaluator.det import BoxEvaluator
from nndet.evaluator.seg import SegmentationEvaluator
from nndet.inference.ensembler.detection import (
    BoxEnsemblerSelective,
    BoxEnsemblerSelective2D,
)
from nndet.inference.ensembler.segmentation import SegmentationEnsembler
from nndet.inference.helper import predict_dir
from nndet.inference.loading import get_loader_fn
from nndet.inference.predictor import Predictor
from nndet.inference.sweeper import BoxSweeper
from nndet.inference.transforms import Inference2D, get_tta_transforms
from nndet.io.transforms import (
    Compose,
    FindInstances,
    Instances2Boxes,
    Instances2Segmentation,
    TransferInputChannel,
)
from nndet.ptmodule.mixins.mode import BoxMixin, SemanticMixin
from nndet.ptmodule.mixins.model import SingleStageMixin
from nndet.ptmodule.module import LightningBaseModule
from nndet.training.learning_rate import LinearWarmupPolyLR
from nndet.training.optimizer import get_params_no_wd_on_norm
from nndet.utils.tensor import to_numpy


class RetinaUNetModule(LightningBaseModule, BoxMixin, SemanticMixin, SingleStageMixin):
    # FIXME
    base_conv_cls = ConvInstanceRelu
    head_conv_cls = ConvGroupRelu
    block = StackedConvBlock2
    encoder_cls = Encoder
    decoder_cls = UFPNModular
    matcher_cls = IoUMatcher
    head_cls = BoxHeadHNM
    head_classifier_cls = CEClassifier
    head_regressor_cls = L1Regressor
    head_sampler_cls = HardNegativeSamplerBatched
    segmenter_cls = DiCESegmenter

    def __init__(self, model_cfg: dict, trainer_cfg: dict, plan: dict, **kwargs):
        """
        RetinaUNet Lightning Module Skeleton

        Args:
            model_cfg: model configuration. Check :method:`from_config_plan`
                for more information
            trainer_cfg: trainer information
            plan: contains parameters which were derived from the planning
                stage
        """
        super().__init__(
            model_cfg=model_cfg,
            trainer_cfg=trainer_cfg,
            plan=plan,
        )

        _classes = [
            f"class{c}" for c in range(plan["architecture"]["classifier_classes"])
        ]
        self.box_evaluator = BoxEvaluator.create(
            classes=_classes,
            fast=True,
            save_dir=None,
        )
        self.seg_evaluator = SegmentationEvaluator.create()

        # box transformations
        trafos = [
            FindInstances(
                instance_key="target",
                save_key="present_instances",
            ),
            Instances2Boxes(
                instance_key="target",
                map_key="instance_mapping",
                box_key="boxes",
                class_key="classes",
                present_instances="present_instances",
            ),
            Instances2Segmentation(
                instance_key="target",
                map_key="instance_mapping",
                present_instances="present_instances",
            ),
        ]

        # transfer learning setup
        # TODO: might move this to base class
        data_channels = self.plan["num_modalities"]  # number of channels of source data
        network_channels = self.plan["architecture"][
            "in_channels"
        ]  # number of channels of target data
        if network_channels > data_channels:
            logger.info(
                "Detected Transfer Learning Setup with different soruce "
                "and target channels. Adding additional transformation."
            )
            trafos.append(
                TransferInputChannel(
                    out_channels=network_channels,
                    data_key="data",
                )
            )

        self.pre_trafo = Compose(trafos)
        self.eval_score_key = "mAP_IoU_0.10_0.50_0.05_MaxDet_100"

    def configure_optimizers(self):
        """
        Configure optimizer and scheduler
        Base configuration is SGD with LinearWarmup and PolyLR learning rate
        schedule
        """
        # configure optimizer
        logger.info(
            f"Running: initial_lr {self.trainer_cfg['initial_lr']} "
            f"weight_decay {self.trainer_cfg['weight_decay']} "
            f"SGD with momentum {self.trainer_cfg['sgd_momentum']} and "
            f"nesterov {self.trainer_cfg['sgd_nesterov']}"
        )
        wd_groups = get_params_no_wd_on_norm(
            self, weight_decay=self.trainer_cfg["weight_decay"]
        )
        optimizer = torch.optim.SGD(
            wd_groups,
            self.trainer_cfg["initial_lr"],
            weight_decay=self.trainer_cfg["weight_decay"],
            momentum=self.trainer_cfg["sgd_momentum"],
            nesterov=self.trainer_cfg["sgd_nesterov"],
        )

        # configure lr scheduler
        num_iterations = (
            self.train_epochs * self.trainer_cfg["num_train_batches_per_epoch"]
        )
        scheduler = LinearWarmupPolyLR(
            optimizer=optimizer,
            warm_iterations=self.trainer_cfg["warm_iterations"],
            warm_lr=self.trainer_cfg["warm_lr"],
            poly_gamma=self.trainer_cfg["poly_gamma"],
            num_iterations=num_iterations,
        )
        return [optimizer], {"scheduler": scheduler, "interval": "step"}

    @staticmethod
    def get_ensembler_cls(key: Hashable, dim: int) -> Callable:
        """
        Get ensembler classes to combine multiple predictions
        Needs to be overwritten in subclasses!
        """
        _lookup = {
            2: {
                "boxes": BoxEnsemblerSelective2D,
                "seg": SegmentationEnsembler,
            },
            3: {
                "boxes": BoxEnsemblerSelective,
                "seg": SegmentationEnsembler,
            },
        }
        if dim == 2:
            raise NotImplementedError
        return _lookup[dim][key]

    @classmethod
    def get_predictor(
        cls,
        plan: Dict,
        models: Sequence[RetinaUNetModule],
        num_tta_transforms: int = None,
        do_seg: bool = False,
        **kwargs,
    ) -> Predictor:
        # process plan
        crop_size = plan["patch_size"]
        batch_size = plan["batch_size"]
        inferene_plan = plan.get("inference_plan", {})
        logger.info(f"Found inference plan: {inferene_plan} for prediction")
        if num_tta_transforms is None:
            num_tta_transforms = 8 if plan["network_dim"] == 3 else 4

        # setup
        tta_transforms, tta_inverse_transforms = get_tta_transforms(
            num_tta_transforms, True
        )
        logger.info(
            f"Using {len(tta_transforms)} tta transformations for prediction (one dummy trafo)."
        )

        ensembler = {
            "boxes": partial(
                cls.get_ensembler_cls(key="boxes", dim=plan["network_dim"]).from_case,
                parameters=inferene_plan,
            )
        }
        if do_seg:
            ensembler["seg"] = partial(
                cls.get_ensembler_cls(key="seg", dim=plan["network_dim"]).from_case,
            )

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
            **kwargs: keyword arguments passed to predict function

        Returns:
            Dict: inference plan
                e.g. (exact params depend on ensembler class usef for prediction)
                `iou_thresh` (float): best IoU threshold
                `score_thresh (float)`: best score threshold
                `no_overlap` (bool): enable/disable class independent NMS (ciNMS)
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
                model_fn=get_loader_fn(mode=self.trainer_cfg.get("sweep_ckpt", "last")),
                **kwargs,
            )

        logger.info("Start parameter sweep...")
        ensembler_cls = self.get_ensembler_cls(
            key="boxes", dim=self.plan["network_dim"]
        )
        sweeper = BoxSweeper(
            classes=[item for _, item in cfg["data"]["labels"].items()],
            pred_dir=prediction_dir,
            gt_dir=processed_eval_labels,
            target_metric=self.eval_score_key,
            ensembler_cls=ensembler_cls,
            save_dir=_save_dir,
        )
        inference_plan = sweeper.run_postprocessing_sweep()
        return inference_plan
