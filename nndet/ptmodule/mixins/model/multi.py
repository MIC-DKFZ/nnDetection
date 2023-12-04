# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Optional, Sequence, Type

from loguru import logger

import nndet.core.ops_torch as ops_torch
from nndet.core.abstract import AbstractDetector, AbstractOneStageDetector
from nndet.core.boxes.coder import BoxCoderND
from nndet.core.boxes.matcher import Matcher
from nndet.core.boxes.sampler import AbstractSampler
from nndet.core.post.box import BoxPostprocessing
from nndet.core.post.mask import MaskPostprocessing
from nndet.core.rois.module.base import BaseRoIModule
from nndet.core.rois.module.cascade import CascadeRoIModule
from nndet.core.rois.module.single import RoIModule
from nndet.core.rois.pooler import RoIPooler
from nndet.nn.backbone.abstract import AbstractBackbone
from nndet.nn.heads.classifier.dense import DenseClassifier
from nndet.nn.heads.classifier.roi import RoIClassifier
from nndet.nn.heads.comb.base import AnchorHead, RoIHead
from nndet.nn.heads.comb.roi import RoIBoxHead
from nndet.nn.heads.masker.roi import Masker
from nndet.nn.heads.regressor.dense import DenseRegressor
from nndet.nn.heads.regressor.roi import RoIRegressor
from nndet.nn.heads.segmenter import Segmenter
from nndet.nn.layers.wrapper import Generator
from nndet.nn.neck.abstract import AbstractNeck
from nndet.ptmodule.mixins.model.single import SingleStageMixin
from nndet.utils.typing import CONVSEQ


class RoIBuildMixin:
    # Use `detector_cls` to set RPN module class
    full_detector_cls: Type[AbstractDetector] = ...  #: Two stage detector class RCNN

    # RoI classes
    roi_conv_cls: Type[CONVSEQ] = ...  #: conv class for RoI head
    roi_module_cls: Type[BaseRoIModule] = ...  #: define class of RoI module (usually `RoIModule` or `CascadeRoIModule`)
    roi_head_cls: Type[RoIBoxHead] = ...  #: define class for RoI box head
    roi_classifier_cls: Type[RoIClassifier] = ...  #: define class for box classifier
    roi_regressor_cls: Type[RoIRegressor] = ...  #: define class for box regressor

    roi_matcher_cls: Type[Matcher] = ...  #:  define class to match proposals to ground truth
    roi_sampler_cls: Type[AbstractSampler] = ...  #: sampler class for negative mining. None = no sampling
    roi_box_pooler_cls: Type[RoIPooler] = ...  #: define pooling operation of RoIs for box branch
    roi_box_post_cls: Type[BoxPostprocessing] = ...  #: define roi box postprocessing strategy

    # optional mask branches
    roi_masker_cls: Optional[Type[Masker]] = None  #: define class of mask branch in RoI module
    roi_mask_pooler_cls: Optional[Type[RoIPooler]] = None  #: define pooling operation of RoIs for mask branch
    roi_mask_post_cls: Optional[Type[MaskPostprocessing]] = None  #: define roi mask postprocessing strategy

    @staticmethod
    def get_roi_box_size(
        plan_arch: dict,
        model_cfg: dict,
    ) -> Sequence[int]:
        """
        Retrieve RoI size for Box Pooler

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings

        Returns:
            Sequence[int]: RoI size for Box Pooler
        """
        return model_cfg["roi_pooling"]["roi_box_size"]

    @staticmethod
    def get_roi_mask_size(
        plan_arch: dict,
        model_cfg: dict,
    ) -> Sequence[int]:
        """
        Retrieve RoI size for Mask Pooler

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings

        Returns:
            Sequence[int]: RoI size for Mask Pooler
        """
        return model_cfg["roi_pooling"]["roi_mask_size"]

    @classmethod
    def _build_rpn(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        plan_anchors: dict,
        **kwargs,
    ) -> AbstractOneStageDetector:
        """
        Build RPN Detector class. This can be any SingelStageDetector
        which produces a hierarchical feature representation and
        region proposals

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings
            plan_anchors: parameters for anchors (see `AnchorGenerator`
                for more info)

                ``"stride"``
                    stride # FIXME docs

                ``"aspect_ratios"``
                    aspect ratios # FIXME docs

                ``"sizes"``
                    sized for 2d acnhors # FIXME docs

                ``"zsizes"``
                    (optional) additional z sizes for 3d # FIXME docs

        Returns:
            AbstractOneStageDetector: one stage detector
        """
        _plan_arch = copy.deepcopy(plan_arch)
        _plan_arch["classifier_classes"] = 1
        rpn = super().from_config_plan(
            model_cfg=model_cfg,
            plan_arch=_plan_arch,
            plan_anchors=plan_anchors,
            **kwargs,
        )
        return rpn

    @classmethod
    def _build_roi_classifier(
        cls,
        plan_arch: dict,
        model_cfg: dict,
    ) -> RoIClassifier:
        """
        Build RoI classifier subnetwork

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings

        Returns:
            RoIClassifier: classifier subnetwork
        """
        conv = Generator(cls.roi_conv_cls, plan_arch["dim"])
        name = cls.head_classifier_cls.__name__
        kwargs = model_cfg["roi_classifier_kwargs"]
        logger.info(f"Building:: roi classifier {name}: {kwargs}")

        roi_channel_multiplier = model_cfg["roi_channel_multiplier"]
        classifier = cls.roi_classifier_cls(
            conv=conv,
            input_size=cls.get_roi_box_size(plan_arch, model_cfg),
            in_channels=plan_arch["fpn_channels"],
            internal_channels=int(roi_channel_multiplier * plan_arch["fpn_channels"]),
            num_classes=plan_arch["classifier_classes"],
            **kwargs,
        )
        return classifier

    @classmethod
    def _build_roi_regressor(
        cls,
        plan_arch: dict,
        model_cfg: dict,
    ) -> RoIRegressor:
        """
        Build RoI regression subnetwork

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings

        Returns:
            RoIRegressor: regression subnetwork
        """
        conv = Generator(cls.roi_conv_cls, plan_arch["dim"])
        name = cls.roi_regressor_cls.__name__
        kwargs = model_cfg["roi_regressor_kwargs"]
        logger.info(f"Building:: roi regressor {name}: {kwargs}")

        roi_channel_multiplier = model_cfg["roi_channel_multiplier"]
        regressor = cls.roi_regressor_cls(
            conv=conv,
            input_size=cls.get_roi_box_size(plan_arch, model_cfg),
            in_channels=plan_arch["fpn_channels"],
            internal_channels=int(roi_channel_multiplier * plan_arch["fpn_channels"]),
            num_classes=plan_arch["classifier_classes"],
            **kwargs,
        )
        return regressor

    @classmethod
    def _build_roi_head(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        classifier: RoIClassifier,
        regressor: RoIRegressor,
        coder: BoxCoderND,
    ) -> RoIHead:
        """
        Build RoI Head which combine the classifier, regressor and
        coder module

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings
            classifier: classifier subnetwork
            regressor: regression subnetwork
            coder: en-/de-coder functionality

        Returns:
            RoIHead: combined head
        """
        name = cls.roi_head_cls.__name__
        kwargs = model_cfg["roi_head_kwargs"]

        logger.info(f"Building:: roi head {name}: {kwargs}")
        roi_head = cls.roi_head_cls(
            classifier=classifier,
            regressor=regressor,
            coder=coder,
            **kwargs,
        )
        return roi_head

    @classmethod
    def _build_roi_masker(
        cls,
        plan_arch: dict,
        model_cfg: dict,
    ) -> Masker:
        """
        Build RoI Mask subnetwork

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings

        Returns:
            Masker: mask subnetwork
        """
        if cls.roi_masker_cls is not None:
            conv = Generator(cls.roi_conv_cls, plan_arch["dim"])
            name = cls.roi_masker_cls.__name__
            kwargs = model_cfg["roi_masker_kwargs"]
            logger.info(f"Building:: roi masker {name}: {kwargs}")

            roi_mask_channel_multiplier = model_cfg["roi_mask_channel_multiplier"]
            masker = cls.roi_masker_cls(
                conv=conv,
                in_channels=plan_arch["fpn_channels"],
                internal_channels=int(roi_mask_channel_multiplier * plan_arch["fpn_channels"]),
                num_classes=plan_arch["classifier_classes"],
                **kwargs,
            )
        else:
            masker = None
        return masker

    @classmethod
    def _build_box_pooler(
        cls,
        plan_arch: dict,
        model_cfg: dict,
    ) -> RoIPooler:
        """
        Build RoI Box Pooler

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings

        Returns:
            RoIPooler: RoI Box Pooler
        """
        pooler_name = cls.roi_box_pooler_cls.__name__
        feature_output_size = cls.get_roi_box_size(plan_arch, model_cfg)
        box_feature_kwargs = model_cfg["roi_pooling"]["roi_box_feature_kwargs"]

        logger.info(
            f"Building:: box pooler {pooler_name} with output "
            f"size {feature_output_size} and feature kwargs {box_feature_kwargs}"
        )

        box_pooler = cls.roi_box_pooler_cls(
            feature_output_size=feature_output_size,
            feature_pool_kwargs=box_feature_kwargs,
        )
        return box_pooler

    @classmethod
    def _build_mask_pooler(
        cls,
        plan_arch: dict,
        model_cfg: dict,
    ) -> RoIPooler:
        """
        Build RoI Mask Pooler

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings

        Returns:
            RoIPooler: RoI Mask Pooler
        """
        if cls.roi_mask_pooler_cls is not None:
            pooler_name = cls.roi_box_pooler_cls.__name__
            mask_feature_size = cls.get_roi_mask_size(plan_arch, model_cfg)
            mask_gt_size = [m * 2 for m in mask_feature_size]  # FIXME important

            mask_feature_kwargs = model_cfg["roi_pooling"]["roi_mask_feature_kwargs"]
            mask_gt_kwargs = model_cfg["roi_pooling"]["roi_mask_gt_kwargs"]

            logger.info(
                f"Building:: mask pooler {pooler_name} with output "
                f"size {mask_feature_size} and gt size {mask_gt_size}"
                f"feature kwargs {mask_feature_kwargs} and gt kwargs {mask_gt_kwargs}"
            )

            mask_pooler = cls.roi_mask_pooler_cls(
                feature_output_size=mask_feature_size,
                mask_output_size=mask_gt_size,
                feature_pool_kwargs=mask_feature_kwargs,
                mask_pool_kwargs=mask_gt_kwargs,
            )
        else:
            mask_pooler = None
        return mask_pooler

    @classmethod
    def _build_roi_box_post(
        cls,
        plan_arch: dict,
        model_cfg: dict,
    ) -> BoxPostprocessing:
        """
        Define module to perform postprocessing of generated boxes

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings

        Returns:
            BoxPostprocessing: module to perform postprocessing of boxes
        """
        name = cls.roi_box_post_cls.__name__
        kwargs = model_cfg["roi_box_post_kwargs"]
        logger.info(f"Building:: roi box postprocessing {name}: {kwargs}")

        roi_box_post = cls.roi_box_post_cls(
            num_classes=plan_arch["classifier_classes"],
            is_class_agnostic=cls.roi_regressor_cls.is_class_agnostic(),
            **model_cfg["roi_box_post_kwargs"],
        )
        return roi_box_post

    @classmethod
    def _build_roi_mask_post(
        cls,
        plan_arch: dict,
        model_cfg: dict,
    ) -> MaskPostprocessing:
        """
        Define module to perform postprocessing of generated masks

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings

        Returns:
            MaskPostprocessing: module to perform postprocessing of masks
        """
        if cls.roi_mask_post_cls is not None:
            name = cls.roi_mask_post_cls.__name__
            kwargs = model_cfg["roi_mask_post_kwargs"]
            logger.info(f"Building:: roi mask postprocessing {name}: {kwargs}")

            roi_mask_post = cls.roi_mask_post_cls(
                num_classes=plan_arch["classifier_classes"],
                is_class_agnostic=cls.roi_masker_cls.is_class_agnostic(),
                **model_cfg["roi_mask_post_kwargs"],
            )
        else:
            roi_mask_post = None
        return roi_mask_post

    @classmethod
    def _build_roi_sampler(
        cls,
        plan_arch: dict,
        model_cfg: dict,
    ) -> AbstractSampler:
        """
        Define strategy to subsampe RoIs to compute the loss

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings

        Returns:
            AbstractSampler: sampler to perform subsampling
        """
        sampler_name = cls.roi_sampler_cls.__name__
        sampler_kwargs = model_cfg["roi_sampler_kwargs"]

        logger.info(f"Building:: roi sampler {sampler_name}: {sampler_kwargs}")
        return cls.roi_sampler_cls(**sampler_kwargs)

    @classmethod
    def _build_roi_module(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        box_head: RoIHead,
        box_pooler: RoIPooler,
        box_post: BoxPostprocessing,
        matcher: Matcher,
        sampler: AbstractSampler,
        # mask heads
        mask_head: Masker,
        mask_pooler: RoIPooler,
        mask_post: MaskPostprocessing,
    ) -> BaseRoIModule:
        """
        Build RoI handling module which performs training and inference of
        the RoI subnetworks

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings
            box_head: RoI head responsible for boxes and labels
            box_pooler: performs RoI pooling for the RoI/box head
            box_post: define postprocessing strategy for boxes
            matcher: assign labels to region proposals
            sampler: subsample region proposals to compute loss
            mask_head: subnetwork to produce masks
            mask_pooler: performs RoI pooling for the mask head
            mask_post: define postprocessing strategy for masks

        Returns:
            RoIModule: assembled RoI module
        """
        roi_module_name = cls.roi_module_cls.__name__
        roi_module_kwargs = model_cfg["roi_module_kwargs"]

        logger.info(f"Building:: roi module {roi_module_name}: {roi_module_kwargs}")

        roi_module = cls.roi_module_cls(
            box_head=box_head,
            box_pooler=box_pooler,
            box_post=box_post,
            matcher=matcher,
            sampler=sampler,
            num_classes=plan_arch["classifier_classes"],
            decoder_levels=plan_arch["decoder_levels"],
            # mask heads
            mask_head=mask_head,
            mask_pooler=mask_pooler,
            mask_post=mask_post,
            **roi_module_kwargs,
        )
        return roi_module


class TwoStageMixin(RoIBuildMixin, SingleStageMixin):
    full_detector_cls: Type[AbstractDetector] = ...  # Two stage detector class RCNN
    # Use `detector_cls` to set RPN module class
    # define RPN cls
    detector_cls: Type[AbstractOneStageDetector] = ...  #: define detector cls

    ###################
    # RPN Configuration
    ###################
    backbone_cls: Type[AbstractBackbone] = ...  #: define class for backbone
    backbone_conv_cls: Type[CONVSEQ] = ...  #: conv class used for backbone

    neck_cls: Type[AbstractNeck] = ...  #: define class for neck
    neck_conv_cls: Type[CONVSEQ] = ...  #: conv class used for neck

    head_cls: Type[AnchorHead] = ...  #: define class for head
    head_conv_cls: Type[CONVSEQ] = ...  #: conv class used for head
    head_classifier_cls: Type[DenseClassifier] = ...  #: define class for head classifier
    head_regressor_cls: Type[DenseRegressor] = ...  #: define class for head regressor

    head_sampler_cls: Optional[
        Type[AbstractSampler]
    ] = None  #: [optional] sampler class for negative mining. None = no sampling.

    matcher_cls: Type[Matcher] = ...  #: define class to match anchors to ground truth
    box_post_cls: Type[BoxPostprocessing] = ...  #: define box postprocessing strategy

    segmenter_cls: Optional[Type[Segmenter]] = None  #: [optional] segmentation head as in RetinaUNet

    ########################
    # RoI Head Configuration
    ########################
    roi_conv_cls: Type[CONVSEQ] = ...  #: conv class for RoI head
    roi_module_cls: Type[RoIModule] = ...  #: define class of RoI module (usually `RoIModule` or `CascadeRoIModule`)
    roi_head_cls: Type[RoIBoxHead] = ...  #: define class for RoI box head
    roi_classifier_cls: Type[RoIClassifier] = ...  #: define class for box classifier
    roi_regressor_cls: Type[RoIRegressor] = ...  #: define class for box regressor

    roi_matcher_cls: Type[Matcher] = ...  #:  define class to match proposals to ground truth
    roi_sampler_cls: Type[AbstractSampler] = ...  #: sampler class for negative mining. None = no sampling
    roi_box_pooler_cls: Type[RoIPooler] = ...  #: define pooling operation of RoIs for box branch
    roi_box_post_cls: Type[BoxPostprocessing] = ...  #: define roi box postprocessing strategy

    # optional mask branches
    roi_masker_cls: Optional[Type[Masker]] = None  #: define class of mask branch in RoI module
    roi_mask_pooler_cls: Optional[Type[RoIPooler]] = None  #: define pooling operation of RoIs for mask branch
    roi_mask_post_cls: Optional[Type[MaskPostprocessing]] = None  #: define roi mask postprocessing strategy

    @classmethod
    def from_config_plan(
        cls,
        model_cfg: dict,
        plan_arch: dict,
        plan_anchors: dict,
        patch_size: Optional[Sequence[int]] = None,
        **kwargs,
    ) -> AbstractDetector:
        """
        Create Configurable Two Stage Detector (e.g. Faster R-CNN)

        Args:
            model_cfg: model configurations. See example configs for more info
            plan_arch: plan architecture

                ``"dim"`` int
                    number of spatial dimensions

                ``"in_channels"`` int
                    number of input channels

                ``"classifier_classes"`` int
                    number of classes

                ``"seg_classes"`` int
                    number of classes

                ``"start_channels"`` int
                    number of start channels in backbone

                ``"fpn_channels"`` int
                    number of channels to use for FPN

                ``"head_channels"`` int
                    number of channels to use for head

                ``"decoder_levels"`` int
                    decoder levels to user for detection

                ``"conv_kernels"`` Sequence[Union[Tuple[int], int]]
                    kernel sizes of convolutions for each stage/level

                ``"strides"`` Sequence[Union[Tuple[int], int]]
                    stride of downsampling block for each stage/level
                    Downsampling is alwyas performed at the beginning of the blocks.
                    First stage/level is always full resolution.

            plan_anchors: parameters for anchors (see `AnchorGenerator` for more info)

                ``"stride"``
                    stride # FIXME docs

                ``"aspect_ratios"``
                    aspect ratios # FIXME docs

                ``"sizes"``
                    sized for 2d acnhors # FIXME docs

                ``"zsizes"``
                    (optional) additional z sizes for 3d # FIXME docs

            patch_size: optionally provide the patch size
                to check compatibility with backbone
            **kwargs: ignored
        """
        plan_arch.update(model_cfg["plan_arch_overwrites"])
        logger.info(
            f"Start channels: {plan_arch['start_channels']}; "
            f"head channels: {plan_arch['head_channels']}; "
            f"fpn channels: {plan_arch['fpn_channels']}"
        )

        # build RPN
        rpn = cls._build_rpn(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            plan_anchors=plan_anchors,
            patch_size=patch_size,
            **kwargs,
        )

        # build stage(s)
        coder = BoxCoderND(weights=(1.0,) * (plan_arch["dim"] * 2))

        roi_classifier = cls._build_roi_classifier(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
        )
        roi_regressor = cls._build_roi_regressor(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
        )
        roi_head = cls._build_roi_head(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            classifier=roi_classifier,
            regressor=roi_regressor,
            coder=coder,
        )

        # mask branch
        masker = cls._build_roi_masker(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
        )

        # pooler
        box_pooler = cls._build_box_pooler(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
        )
        mask_pooler = cls._build_mask_pooler(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
        )

        # RoI Module
        roi_matcher = cls.roi_matcher_cls(
            similarity_fn=ops_torch.box_iou,
            **model_cfg["roi_matcher_kwargs"],
        )
        roi_sampler = cls._build_roi_sampler(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
        )

        roi_box_post = cls._build_roi_box_post(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
        )
        roi_mask_post = cls._build_roi_mask_post(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
        )

        roi_module = cls._build_roi_module(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            box_head=roi_head,
            box_pooler=box_pooler,
            box_post=roi_box_post,
            matcher=roi_matcher,
            sampler=roi_sampler,
            # mask heads
            mask_head=masker,
            mask_pooler=mask_pooler,
            mask_post=roi_mask_post,
        )

        return cls.full_detector_cls(
            rpn=rpn,
            roi_module=roi_module,
        )


class MultiStageMixin(RoIBuildMixin, SingleStageMixin):
    full_detector_cls: Type[AbstractDetector] = ...  # Two stage detector class RCNN
    # Use `detector_cls` to set RPN module class
    # define RPN cls
    detector_cls: Type[AbstractOneStageDetector] = ...  #: define detector cls

    ###################
    # RPN Configuration
    ###################
    backbone_cls: Type[AbstractBackbone] = ...  #: define class for backbone
    backbone_conv_cls: Type[CONVSEQ] = ...  #: conv class used for backbone

    neck_cls: Type[AbstractNeck] = ...  #: define class for neck
    neck_conv_cls: Type[CONVSEQ] = ...  #: conv class used for neck

    head_cls: Type[AnchorHead] = ...  #: define class for head
    head_conv_cls: Type[CONVSEQ] = ...  #: conv class used for head
    head_classifier_cls: Type[DenseClassifier] = ...  #: define class for head classifier
    head_regressor_cls: Type[DenseRegressor] = ...  #: define class for head regressor

    head_sampler_cls: Optional[
        Type[AbstractSampler]
    ] = None  #: [optional] sampler class for negative mining. None = no sampling.

    matcher_cls: Type[Matcher] = ...  #: define class to match anchors to ground truth
    box_post_cls: Type[BoxPostprocessing] = ...  #: define box postprocessing strategy

    segmenter_cls: Optional[Type[Segmenter]] = None  #: [optional] segmentation head as in RetinaUNet

    ########################
    # RoI Head Configuration
    ########################
    roi_conv_cls: Type[CONVSEQ] = ...  #: conv class for RoI head
    roi_module_cls: Type[
        CascadeRoIModule
    ] = ...  #: define class of RoI module (usually `RoIModule` or `CascadeRoIModule`)
    roi_head_cls: Type[RoIBoxHead] = ...  #: define class for RoI box head
    roi_classifier_cls: Type[RoIClassifier] = ...  #: define class for box classifier
    roi_regressor_cls: Type[RoIRegressor] = ...  #: define class for box regressor

    roi_matcher_cls: Type[Matcher] = ...  #:  define class to match proposals to ground truth
    roi_sampler_cls: Type[AbstractSampler] = ...  #: sampler class for negative mining. None = no sampling
    roi_box_pooler_cls: Type[RoIPooler] = ...  #: define pooling operation of RoIs for box branch
    roi_box_post_cls: Type[BoxPostprocessing] = ...  #: define roi box postprocessing strategy

    # optional mask branches
    roi_masker_cls: Optional[Type[Masker]] = None  #: define class of mask branch in RoI module
    roi_mask_pooler_cls: Optional[Type[RoIPooler]] = None  #: define pooling operation of RoIs for mask branch
    roi_mask_post_cls: Optional[Type[MaskPostprocessing]] = None  #: define roi mask postprocessing strategy

    @classmethod
    def from_config_plan(
        cls,
        model_cfg: dict,
        plan_arch: dict,
        plan_anchors: dict,
        patch_size: Optional[Sequence[int]] = None,
        **kwargs,
    ) -> AbstractDetector:
        """
        Create Configurable Multi Stage Detector (e.g. Cascade R-CNN)

        Args:
            model_cfg: model configurations. See example configs for more info
            plan_arch: plan architecture

                ``"dim"`` int
                    number of spatial dimensions

                ``"in_channels"`` int
                    number of input channels

                ``"classifier_classes"`` int
                    number of classes

                ``"seg_classes"`` int
                    number of classes

                ``"start_channels"`` int
                    number of start channels in backbone

                ``"fpn_channels"`` int
                    number of channels to use for FPN

                ``"head_channels"`` int
                    number of channels to use for head

                ``"decoder_levels"`` int
                    decoder levels to user for detection

                ``"conv_kernels"`` Sequence[Union[Tuple[int], int]]
                    kernel sizes of convolutions for each stage/level

                ``"strides"`` Sequence[Union[Tuple[int], int]]
                    stride of downsampling block for each stage/level
                    Downsampling is alwyas performed at the beginning of the blocks.
                    First stage/level is always full resolution.

            plan_anchors: parameters for anchors (see `AnchorGenerator` for more info)

                ``"stride"``
                    stride # FIXME docs

                ``"aspect_ratios"``
                    aspect ratios # FIXME docs

                ``"sizes"``
                    sized for 2d acnhors # FIXME docs

                ``"zsizes"``
                    (optional) additional z sizes for 3d # FIXME docs

            patch_size: optionally provide the patch size
                to check compatibility with backbone
            **kwargs: ignored
        """
        # build RPN
        rpn = super().from_config_plan(
            model_cfg=model_cfg,
            plan_arch=plan_arch,
            plan_anchors=plan_anchors,
            patch_size=patch_size,
            **kwargs,
        )

        # build stage(s)
        coder = BoxCoderND(weights=(1.0,) * (plan_arch["dim"] * 2))

        heads = []
        matchers = []
        maskers = []
        for i in range(model_cfg["roi_cascade_stages"]):
            roi_classifier = cls._build_roi_classifier(
                plan_arch=plan_arch,
                model_cfg=model_cfg,
            )
            roi_regressor = cls._build_roi_regressor(
                plan_arch=plan_arch,
                model_cfg=model_cfg,
            )
            roi_head = cls._build_roi_head(
                plan_arch=plan_arch,
                model_cfg=model_cfg,
                classifier=roi_classifier,
                regressor=roi_regressor,
                coder=coder,
            )
            heads.append(roi_head)

            # mask branch
            masker = cls._build_roi_masker(
                plan_arch=plan_arch,
                model_cfg=model_cfg,
            )
            maskers.append(masker)

            # Matcher
            roi_matcher = cls.roi_matcher_cls(
                similarity_fn=ops_torch.box_iou,
                **model_cfg[f"roi_matcher_kwargs_s{i}"],
            )
            matchers.append(roi_matcher)

        # pooler
        box_pooler = cls._build_box_pooler(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
        )
        mask_pooler = cls._build_mask_pooler(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
        )

        roi_box_post = cls._build_roi_box_post(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
        )
        roi_mask_post = cls._build_roi_mask_post(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
        )

        # RoI Module
        roi_sampler = cls._build_roi_sampler(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
        )

        # seq[None] -> None
        if maskers[0] is None:
            maskers = None

        roi_module = cls._build_roi_module(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            box_head=heads,
            box_pooler=box_pooler,
            box_post=roi_box_post,
            matcher=matchers,
            sampler=roi_sampler,
            # mask heads
            mask_head=maskers,
            mask_pooler=mask_pooler,
            mask_post=roi_mask_post,
        )
        return cls.full_detector_cls(
            rpn=rpn,
            roi_module=roi_module,
        )
