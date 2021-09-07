# TODO: cleanup

import torch

from nndet.arch.blocks.basic import StackedConvBlock2
from nndet.arch.conv import ConvGroupRelu, ConvInstanceRelu
from nndet.arch.decoder.base import UFPNModular
from nndet.arch.encoder.modular import Encoder
from nndet.arch.heads.classifier.dense import CEClassifier
from nndet.arch.heads.classifier.roi import RoIClassifierTwoMLP
from nndet.arch.heads.comb.anchor_sampled import BoxHeadHNM
from nndet.arch.heads.comb.roi import RoIBoxHead
from nndet.arch.heads.masker import BCESingleMasker
from nndet.arch.heads.regressor.dense_single import GIoURegressor, L1Regressor
from nndet.arch.heads.regressor.roi_single import RoIRegressorConv
from nndet.core.boxes.matcher import ATSSMatcher, IoUMatcher
from nndet.core.boxes.sampler import (
    BalancedHardNegativeSampler,
    HardNegativeSamplerBatched,
)
from nndet.core.rcnn import RCNN
from nndet.core.retina import BaseRetinaNet
from nndet.core.rois.module import CascadeRoIModule, RoIModule
from nndet.core.rois.pooler import RoIAlignNaiveAssign
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.mixins.mode import BoxMixin
from nndet.ptmodule.mixins.model import MultiStageMixin, TwoStageMixin
from nndet.ptmodule.mixins.optimizer import SGDDefaultMixin
from nndet.ptmodule.mixins.prediction import BoxPredictionMixin
from nndet.ptmodule.module import LightningBaseModule


@MODULE_REGISTRY.register
class BoxRCNN(
    SGDDefaultMixin,  # Default SGD optimization
    LightningBaseModule,  # Detection Base
    BoxMixin,  # Boundig Box Evaluation
    TwoStageMixin,  # Single Stage Detector
    BoxPredictionMixin,  # Bounding Box Sweep
):
    # define detector cls
    detector_cls = BaseRetinaNet

    backbone_cls = Encoder  # define class for backbone
    backbone_conv_cls = ConvInstanceRelu  # conv class used for backbone
    backbone_block = StackedConvBlock2  # define central building block of backbone

    neck_cls = UFPNModular  # define class for neck
    neck_conv_cls = ConvInstanceRelu  # conv class used for neck

    head_cls = BoxHeadHNM  # define class for head
    head_conv_cls = ConvGroupRelu  # conv class used for head
    head_classifier_cls = CEClassifier  # define class for head classifier
    head_regressor_cls = L1Regressor  # define class for head regressor
    # [optional] sampler class for negative mining
    # if None: no sampler will be given to the head
    head_sampler_cls = HardNegativeSamplerBatched

    matcher_cls = ATSSMatcher  # define class to match anchors to ground truth
    segmenter_cls = None  # [optional] segmentation head as in RetinaUNet

    # Use `detector_cls` to set RPN module class
    full_detector_cls = RCNN  # Two stage detector class RCNN

    # RoI classes
    roi_module_cls = RoIModule  # RoIModule
    roi_head_cls = RoIBoxHead  # RoIBoxHead
    roi_classifier_cls = RoIClassifierTwoMLP  # RoIClassifierTwoMLP
    roi_regressor_cls = RoIRegressorConv  # RoIRegressorConv

    roi_matcher_cls = IoUMatcher  # IoUMatcher
    roi_sampler_cls = BalancedHardNegativeSampler  # BalancedHardNegativeSampler
    roi_box_pooler_cls = RoIAlignNaiveAssign  # RoIAlignNaiveAssign

    # optional mask branches
    roi_masker_cls = BCESingleMasker  # BCESingleMasker
    roi_mask_pooler_cls = RoIAlignNaiveAssign  # RoIAlignNaiveAssign

    def training_step(self, batch, batch_idx):  # TODO
        """
        Computes a single training step
        See :class:`BaseRetinaNet` for more information
        """
        with torch.no_grad():
            batch = self.pre_trafo(**batch)

        losses, _ = self.model.train_step(
            images=batch["data"],
            targets={
                "target_boxes": batch["boxes"],
                "target_classes": batch["classes"],
                # "target_seg": batch["target_seg"][:, 0],  # Remove channel dimension
                "target_masks": batch["target"][:, 0],  # Remove channel dimension
                "target_num_instances": [len(i) for i in batch["present_instances"]],
            },
            predict=False,
            batch_num=batch_idx,
        )
        loss = sum(losses.values())
        self.log_dict(
            {f"train_loss_step/{k}": i for k, i in losses.items()},
            prog_bar=True,
            logger=True,
            on_step=True,
        )
        return {"loss": loss, **{key: l.detach().item() for key, l in losses.items()}}

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            batch = self.pre_trafo(**batch)
            targets = {
                "target_boxes": batch["boxes"],
                "target_classes": batch["classes"],
                # "target_seg": batch['target'][:, 0]  # Remove channel dimension
            }
            predictions = self.model.inference_step(
                images=batch["data"],
                targets=targets,
                predict=True,
                batch_num=batch_idx,
            )

        self.evaluation_step(predictions=predictions, targets=targets)
        return {"loss": 0}


@MODULE_REGISTRY.register
class BoxCascadeRCNN(
    SGDDefaultMixin,  # Default SGD optimization
    LightningBaseModule,  # Detection Base
    BoxMixin,  # Boundig Box Evaluation
    MultiStageMixin,  # Single Stage Detector
    BoxPredictionMixin,  # Bounding Box Sweep
):
    # define detector cls
    detector_cls = BaseRetinaNet

    backbone_cls = Encoder  # define class for backbone
    backbone_conv_cls = ConvInstanceRelu  # conv class used for backbone
    backbone_block = StackedConvBlock2  # define central building block of backbone

    neck_cls = UFPNModular  # define class for neck
    neck_conv_cls = ConvInstanceRelu  # conv class used for neck

    head_cls = BoxHeadHNM  # define class for head
    head_conv_cls = ConvGroupRelu  # conv class used for head
    head_classifier_cls = CEClassifier  # define class for head classifier
    head_regressor_cls = GIoURegressor  # define class for head regressor
    # [optional] sampler class for negative mining
    # if None: no sampler will be given to the head
    head_sampler_cls = HardNegativeSamplerBatched

    matcher_cls = ATSSMatcher  # define class to match anchors to ground truth
    segmenter_cls = None  # [optional] segmentation head as in RetinaUNet

    # Use `detector_cls` to set RPN module class
    full_detector_cls = RCNN  # Two stage detector class RCNN

    # RoI classes
    roi_module_cls = CascadeRoIModule  # RoIModule
    roi_head_cls = RoIBoxHead  # RoIBoxHead
    roi_classifier_cls = RoIClassifierTwoMLP  # RoIClassifierTwoMLP
    roi_regressor_cls = RoIRegressorConv  # RoIRegressorConv

    roi_matcher_cls = IoUMatcher  # IoUMatcher
    roi_sampler_cls = BalancedHardNegativeSampler  # BalancedHardNegativeSampler
    roi_box_pooler_cls = RoIAlignNaiveAssign  # RoIAlignNaiveAssign

    # optional mask branches
    roi_masker_cls = BCESingleMasker  # BCESingleMasker
    roi_mask_pooler_cls = RoIAlignNaiveAssign  # RoIAlignNaiveAssign

    def training_step(self, batch, batch_idx):  # TODO
        """
        Computes a single training step
        See :class:`BaseRetinaNet` for more information
        """
        with torch.no_grad():
            batch = self.pre_trafo(**batch)

        losses, _ = self.model.train_step(
            images=batch["data"],
            targets={
                "target_boxes": batch["boxes"],
                "target_classes": batch["classes"],
                # "target_seg": batch["target_seg"][:, 0],  # Remove channel dimension
                "target_masks": batch["target"][:, 0],  # Remove channel dimension
                "target_num_instances": [len(i) for i in batch["present_instances"]],
            },
            predict=False,
            batch_num=batch_idx,
        )
        loss = sum(losses.values())
        self.log_dict(
            {f"train_loss_step/{k}": i for k, i in losses.items()},
            prog_bar=True,
            logger=True,
            on_step=True,
        )
        return {"loss": loss, **{key: l.detach().item() for key, l in losses.items()}}

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            batch = self.pre_trafo(**batch)
            targets = {
                "target_boxes": batch["boxes"],
                "target_classes": batch["classes"],
                # "target_seg": batch['target'][:, 0]  # Remove channel dimension
            }
            predictions = self.model.inference_step(
                images=batch["data"],
                targets=targets,
                predict=True,
                batch_num=batch_idx,
            )

        self.evaluation_step(predictions=predictions, targets=targets)
        return {"loss": 0}
