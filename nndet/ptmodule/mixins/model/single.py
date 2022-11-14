# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Optional, Sequence, Type

from loguru import logger

from nndet.core.abstract import AbstractOneStageDetector
from nndet.core.boxes.anchors import AnchorGeneratorType, get_anchor_generator
from nndet.core.boxes.coder import BoxCoderND, CoderType
from nndet.core.boxes.matcher import Matcher
from nndet.core.boxes.ops import box_iou
from nndet.core.boxes.sampler import SamplerType
from nndet.core.post.box import BoxPostprocessing
from nndet.nn.backbone.abstract import AbstractBackbone
from nndet.nn.heads.classifier.dense import DenseClassifier
from nndet.nn.heads.comb.base import AnchorHead, AnchorHeadType
from nndet.nn.heads.regressor.dense import DenseRegressor
from nndet.nn.heads.segmenter import Segmenter, SegmenterType
from nndet.nn.layers.wrapper import Generator
from nndet.nn.neck.abstract import AbstractNeck
from nndet.ptmodule.mixins.model.base import ModelMixin
from nndet.utils.typing import CONVSEQ


class SingleStageMixin(ModelMixin):
    """
    This class provides the template to build a detection model.
    By overwriting the class attributes the configuration can be adapted.
    """

    detector_cls: Type[AbstractOneStageDetector] = ...  #: define detector cls

    backbone_cls: Type[AbstractBackbone] = ...  #: define class for backbone
    backbone_conv_cls: Type[CONVSEQ] = ...  #: conv class used for backbone

    neck_cls: Type[AbstractNeck] = ...  #: define class for neck
    neck_conv_cls: Type[CONVSEQ] = ...  #: conv class used for neck

    head_cls: Type[AnchorHead] = ...  #: define class for head
    head_conv_cls: Type[CONVSEQ] = ...  #: conv class used for head
    head_classifier_cls: Type[DenseClassifier] = ...  #: define class for head classifier
    head_regressor_cls: Type[DenseRegressor] = ...  #: define class for head regressor

    head_sampler_cls: Optional[
        Type[SamplerType]
    ] = None  #: [optional] sampler class for negative mining. None = no sampling.

    matcher_cls: Type[Matcher] = ...  #: define class to match anchors to ground truth
    box_post_cls: Type[BoxPostprocessing] = ...  #: define box postprocessing strategy

    segmenter_cls: Optional[Type[Segmenter]] = None  #: [optional] segmentation head as in RetinaUNet

    @classmethod
    def from_config_plan(
        cls,
        model_cfg: dict,
        plan_arch: dict,
        plan_anchors: dict,
        patch_size: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        """
        Create Configurable Single Stage Detector (e.g. Retina U-Net)

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
                    stride # FIXME

                ``"aspect_ratios"``
                    aspect ratios # FIXME

                ``"sizes"``
                    sized for 2d acnhors # FIXME

                ``"zsizes"``
                    (optional) additional z sizes for 3d # FIXME

            patch_size: optionally provide the patch size
                to check compatibility with backbone
            **kwargs: ignored
        """
        logger.info(
            f"Architecture overwrites: {model_cfg['plan_arch_overwrites']} "
            f"Anchor overwrites: {model_cfg['plan_anchors_overwrites']}"
        )
        logger.info(f"Building architecture according to plan of {plan_arch.get('arch_name', 'not_found')}")
        plan_arch.update(model_cfg["plan_arch_overwrites"])
        plan_anchors.update(model_cfg["plan_anchors_overwrites"])
        logger.info(
            f"Start channels: {plan_arch['start_channels']}; "
            f"head channels: {plan_arch['head_channels']}; "
            f"fpn channels: {plan_arch['fpn_channels']}"
        )

        _plan_anchors = copy.deepcopy(plan_anchors)
        coder = BoxCoderND(weights=(1.0,) * (plan_arch["dim"] * 2))
        s_param = False if ("aspect_ratios" in _plan_anchors) and (_plan_anchors["aspect_ratios"] is not None) else True
        anchor_generator = get_anchor_generator(plan_arch["dim"], s_param=s_param)(**_plan_anchors)

        backbone = cls._build_backbone(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            patch_size=patch_size,
        )
        neck = cls._build_neck(
            backbone=backbone,
            plan_arch=plan_arch,
            model_cfg=model_cfg,
        )

        classifier = cls._build_head_classifier(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            anchor_generator=anchor_generator,
        )
        regressor = cls._build_head_regressor(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            anchor_generator=anchor_generator,
        )
        head = cls._build_head(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            classifier=classifier,
            regressor=regressor,
            coder=coder,
        )

        matcher = cls.matcher_cls(
            similarity_fn=box_iou,
            **model_cfg["matcher_kwargs"],
        )
        box_post = cls._build_box_post(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
        )

        # optional modules
        detector_kwargs = {}
        if cls.has_segmenter():
            detector_kwargs["segmenter"] = cls._build_segmenter(
                plan_arch=plan_arch,
                model_cfg=model_cfg,
                neck=neck,
            )

        return cls.detector_cls(
            dim=plan_arch["dim"],
            backbone=backbone,
            neck=neck,
            head=head,
            anchor_generator=anchor_generator,
            matcher=matcher,
            decoder_levels=plan_arch["decoder_levels"],
            box_post=box_post,
            **detector_kwargs,
        )

    @classmethod
    def _build_backbone(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        patch_size: Optional[Sequence[int]] = None,
    ) -> AbstractBackbone:
        """
        Build backbone network

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings
            patch_size: optionally provide the patch size
                to check compatibility with backbone

        Returns:
            AbstractBackbone: backbone instance
        """
        conv = Generator(cls.backbone_conv_cls, plan_arch["dim"])
        backbone: AbstractBackbone = cls.backbone_cls.from_config_plan(
            conv=conv,
            backbone_cfg=model_cfg["backbone_kwargs"],
            plan_arch=plan_arch,
        )
        if patch_size is not None:
            if not backbone.check_patch_size(patch_size):
                raise ValueError(
                    f"Backbone {cls.backbone_cls.__name__} with absolute "
                    f"strides {backbone.get_absolute_strides()} is not compatible "
                    f"with patch size {patch_size}"
                )
            else:
                logger.info("Patch size check complete, backbone is compatible.")
        return backbone

    @classmethod
    def _build_neck(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        backbone: AbstractBackbone,
    ) -> AbstractNeck:
        """
        Build neck network

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings

        Returns:
            AbstractNeck: neck instance
        """
        conv = Generator(cls.neck_conv_cls, plan_arch["dim"])
        logger.info(f"Building:: neck {cls.neck_cls.__name__}: {model_cfg['neck_kwargs']}")

        decoder_levels = plan_arch["decoder_levels"]
        neck = cls.neck_cls(
            conv=conv,
            conv_kernels=plan_arch["conv_kernels"],
            relative_strides=backbone.get_relative_strides(),
            in_channels=backbone.get_channels(),
            first_decoder_level=min(decoder_levels),
            last_decoder_level=max(decoder_levels),
            fpn_out_channels=plan_arch["fpn_channels"],
            **model_cfg["neck_kwargs"],
        )
        return neck

    @classmethod
    def _build_head_classifier(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        anchor_generator: AnchorGeneratorType,
    ) -> DenseClassifier:
        """
        Build classification subnetwork for detection head

        Args:
            anchor_generator: anchor generator instance
            plan_arch: architecture settings
            model_cfg: additional architecture settings

        Returns:
            ClassifierType: classification instance
        """
        conv = Generator(cls.head_conv_cls, plan_arch["dim"])
        name = cls.head_classifier_cls.__name__
        kwargs = model_cfg["head_classifier_kwargs"]

        logger.info(f"Building:: classifier {name}: {kwargs}")
        classifier = cls.head_classifier_cls(
            conv=conv,
            in_channels=plan_arch["fpn_channels"],
            internal_channels=plan_arch["head_channels"],
            num_classes=plan_arch["classifier_classes"],
            anchors_per_pos=anchor_generator.num_anchors_per_location()[0],
            num_levels=len(plan_arch["decoder_levels"]),
            **kwargs,
        )
        return classifier

    @classmethod
    def _build_head_regressor(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        anchor_generator: AnchorGeneratorType,
    ) -> DenseRegressor:
        """
        Build regression subnetwork for detection head

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings
            anchor_generator: anchor generator instance

        Returns:
            RegressorType: classification instance
        """
        conv = Generator(cls.head_conv_cls, plan_arch["dim"])
        name = cls.head_regressor_cls.__name__
        kwargs = model_cfg["head_regressor_kwargs"]

        logger.info(f"Building:: regressor {name}: {kwargs}")
        regressor = cls.head_regressor_cls(
            conv=conv,
            in_channels=plan_arch["fpn_channels"],
            internal_channels=plan_arch["head_channels"],
            anchors_per_pos=anchor_generator.num_anchors_per_location()[0],
            num_levels=len(plan_arch["decoder_levels"]),
            **kwargs,
        )
        return regressor

    @classmethod
    def _build_head(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        classifier: DenseClassifier,
        regressor: DenseRegressor,
        coder: CoderType,
    ) -> AnchorHeadType:
        """
        Build detection head

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings
            classifier: classifier instance
            regressor: regressor instance
            coder: coder instance to encode boxes

        Returns:
            HeadType: instantiated head
        """
        head_name = cls.head_cls.__name__
        head_kwargs = model_cfg["head_kwargs"]

        logger.info(f"Building:: head {head_name}: {head_kwargs} ")

        # optional sampler
        if cls.has_sampler():
            head_kwargs["sampler"] = cls._build_sampler(plan_arch=plan_arch, model_cfg=model_cfg)

        head = cls.head_cls(
            classifier=classifier,
            regressor=regressor,
            coder=coder,
            **head_kwargs,
        )
        return head

    @classmethod
    def has_sampler(cls) -> bool:
        """
        Check if configuration should have a sampler

        Returns:
            bool: True if detector needs sampler, False othterwise
        """
        return cls.head_sampler_cls is not None

    @classmethod
    def _build_sampler(
        cls,
        plan_arch: dict,
        model_cfg: dict,
    ):
        sampler_name = cls.head_sampler_cls.__name__
        sampler_kwargs = model_cfg["head_sampler_kwargs"]

        logger.info(f"Building:: sampler {sampler_name}: {sampler_kwargs}")
        return cls.head_sampler_cls(**sampler_kwargs)

    @classmethod
    def _build_box_post(
        cls,
        plan_arch: dict,
        model_cfg: dict,
    ):
        kwargs = {}

        # model_max_instances_per_batch_element (in mdt per img, per class; here: per img)
        if "detections_per_img" in model_cfg:
            kwargs["detections_per_img"] = model_cfg["detections_per_img"]
        else:
            kwargs["detections_per_img"] = plan_arch.get("detections_per_img", 100)  # FIXME

        kwargs["score_thresh"] = plan_arch.get("score_thresh", 0)
        kwargs["topk_candidates"] = plan_arch.get("topk_candidates", 10000)
        kwargs["remove_small_boxes"] = plan_arch.get("remove_small_boxes", 0.01)
        if "rpn_nms_thresh" in model_cfg:
            kwargs["nms_thresh"] = model_cfg["rpn_nms_thresh"]
            logger.info(f"Found RPN NMS thresh in config, using {kwargs['nms_thresh']}")
        else:
            kwargs["nms_thresh"] = plan_arch.get("nms_thresh", 0.6)

        name = cls.box_post_cls.__name__
        logger.info(f"Building:: box postprocessing {name}: {kwargs}")

        box_post = cls.box_post_cls(
            num_foreground_classes=plan_arch["classifier_classes"],
            class_agnostic=cls.head_regressor_cls.class_agnostic(),
            **kwargs,
        )
        return box_post

    @classmethod
    def has_segmenter(cls):
        """
        Check if configuration should have a segmenter

        Returns:
            bool: True if detector needs segemetner, False othterwise
        """
        return cls.segmenter_cls is not None

    @classmethod
    def _build_segmenter(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        neck: AbstractNeck,
    ) -> SegmenterType:
        """
        Build segmenter head

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings
            neck: neck instance

        Returns:
            SegmenterType: segmenter head
        """
        name = cls.segmenter_cls.__name__
        kwargs = model_cfg["segmenter_kwargs"]
        conv = Generator(cls.neck_conv_cls, plan_arch["dim"])

        logger.info(f"Building:: segmenter {name} {kwargs}")
        segmenter = cls.segmenter_cls(
            conv,
            seg_classes=plan_arch["seg_classes"],
            in_channels=neck.get_channels(),
            decoder_levels=plan_arch["decoder_levels"],
            **kwargs,
        )
        return segmenter
