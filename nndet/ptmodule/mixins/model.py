# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import copy
from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence, Type

from loguru import logger

from nndet.core.abstract import AbstractDetector, AbstractOneStageDetector
from nndet.core.boxes.anchors import AnchorGeneratorType, get_anchor_generator
from nndet.core.boxes.coder import BoxCoderND, CoderType
from nndet.core.boxes.matcher import Matcher
from nndet.core.boxes.ops import box_iou
from nndet.core.boxes.sampler import SamplerType
from nndet.core.post.box import BoxPostprocessing
from nndet.core.post.mask import MaskPostprocessing
from nndet.core.rois.module.base import RoIModule
from nndet.core.rois.pooler import RoIPooler
from nndet.nn.backbone.abstract import AbstractBackbone
from nndet.nn.heads.classifier import DenseClassifierType
from nndet.nn.heads.classifier.dense import DenseClassifier
from nndet.nn.heads.classifier.roi import RoIClassifier
from nndet.nn.heads.comb.base import AnchorHead, AnchorHeadType
from nndet.nn.heads.comb.roi import RoIBoxHead
from nndet.nn.heads.masker.base import Masker
from nndet.nn.heads.regressor import DenseRegressorType
from nndet.nn.heads.regressor.dense import DenseRegressor
from nndet.nn.heads.regressor.roi import RoIRegressor
from nndet.nn.heads.segmenter import Segmenter, SegmenterType
from nndet.nn.layers.wrapper import Generator
from nndet.nn.neck.abstract import AbstractNeck
from nndet.utils.typing import CONVSEQ


class ModelMixin(ABC):
    @classmethod
    @abstractmethod
    def from_config_plan(
        cls,
        model_cfg: dict,
        plan_arch: dict,
        plan_anchors: dict,
        patch_size: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        """
        Create Configurable Model

        Args:
            model_cfg: model configurations.
                Exact parameters depend on subclass.
            plan_arch: plan architecture
                Exact parameters depend on subclass.
            plan_anchors: parameters for anchors
                Exact parameters depend on subclass.
            patch_size: optionally provide the patch size
                to check compatibility with backbone
            **kwargs: ignored
        """
        raise NotImplementedError


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
    head_classifier_cls: Type[
        DenseClassifier
    ] = ...  #: define class for head classifier
    head_regressor_cls: Type[DenseRegressor] = ...  #: define class for head regressor

    head_sampler_cls: Optional[
        Type[SamplerType]
    ] = None  #: [optional] sampler class for negative mining. None = no sampling.

    matcher_cls: Type[Matcher] = ...  #: define class to match anchors to ground truth
    segmenter_cls: Optional[
        Type[Segmenter]
    ] = None  #: [optional] segmentation head as in RetinaUNet

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
        logger.info(
            f"Building architecture according to plan of {plan_arch.get('arch_name', 'not_found')}"
        )
        plan_arch.update(model_cfg["plan_arch_overwrites"])
        plan_anchors.update(model_cfg["plan_anchors_overwrites"])
        logger.info(
            f"Start channels: {plan_arch['start_channels']}; "
            f"head channels: {plan_arch['head_channels']}; "
            f"fpn channels: {plan_arch['fpn_channels']}"
        )

        _plan_anchors = copy.deepcopy(plan_anchors)
        coder = BoxCoderND(weights=(1.0,) * (plan_arch["dim"] * 2))
        s_param = (
            False
            if ("aspect_ratios" in _plan_anchors)
            and (_plan_anchors["aspect_ratios"] is not None)
            else True
        )
        anchor_generator = get_anchor_generator(plan_arch["dim"], s_param=s_param)(
            **_plan_anchors
        )

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
        matcher = cls.matcher_cls(
            similarity_fn=box_iou,
            **model_cfg["matcher_kwargs"],
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

        if "detections_per_img" in model_cfg:
            detections_per_img = model_cfg["detections_per_img"]
        else:
            detections_per_img = plan_arch.get("detections_per_img", 100)  # FIXME
        score_thresh = plan_arch.get("score_thresh", 0)
        topk_candidates = plan_arch.get("topk_candidates", 10000)
        remove_small_boxes = plan_arch.get("remove_small_boxes", 0.01)
        if "rpn_nms_thresh" in model_cfg:
            nms_thresh = model_cfg["rpn_nms_thresh"]
            logger.info(f"Found RPN NMS thresh in config, using {nms_thresh}")
        else:
            nms_thresh = plan_arch.get("nms_thresh", 0.6)

        logger.info(
            f"Model Inference Summary: \n"
            f"detections_per_img: {detections_per_img} \n"
            f"score_thresh: {score_thresh} \n"
            f"topk_candidates: {topk_candidates} \n"
            f"remove_small_boxes: {remove_small_boxes} \n"
            f"nms_thresh: {nms_thresh}",
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
            num_classes=plan_arch["classifier_classes"],
            decoder_levels=plan_arch["decoder_levels"],
            # model_max_instances_per_batch_element (in mdt per img, per class; here: per img)
            detections_per_img=detections_per_img,
            score_thresh=score_thresh,
            topk_candidates=topk_candidates,
            remove_small_boxes=remove_small_boxes,
            nms_thresh=nms_thresh,
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
        logger.info(
            f"Building:: neck {cls.neck_cls.__name__}: {model_cfg['neck_kwargs']}"
        )

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
    ) -> DenseClassifierType:
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
    ) -> DenseRegressorType:
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
        classifier: DenseClassifierType,
        regressor: DenseRegressorType,
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
            head_kwargs["sampler"] = cls._build_sampler(
                plan_arch=plan_arch, model_cfg=model_cfg
            )

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


class RoIBuildMixin:
    # Use `detector_cls` to set RPN module class
    full_detector_cls: Type[AbstractDetector] = ...  #: Two stage detector class RCNN

    # RoI classes
    roi_conv_cls: Type[CONVSEQ] = ...  #: conv class for RoI head
    roi_module_cls: Type[
        RoIModule
    ] = ...  #: define class of RoI module (usually `RoIModule` or `CascadeRoIModule`)
    roi_head_cls: Type[RoIBoxHead] = ...  #: define class for RoI box head
    roi_classifier_cls: Type[RoIClassifier] = ...  #: define class for box classifier
    roi_regressor_cls: Type[RoIRegressor] = ...  #: define class for box regressor

    roi_matcher_cls: Type[
        Matcher
    ] = ...  #:  define class to match proposals to ground truth
    roi_sampler_cls: Type[
        SamplerType
    ] = ...  #: sampler class for negative mining. None = no sampling
    roi_box_pooler_cls: Type[
        RoIPooler
    ] = ...  #: define pooling operation of RoIs for box branch
    roi_box_post_cls: Type[
        BoxPostprocessing
    ] = ...  #: define roi box postprocessing strategy

    # optional mask branches
    roi_masker_cls: Type[Masker] = None  #: define class of mask branch in RoI module
    roi_mask_pooler_cls: Type[
        RoIPooler
    ] = None  #: define pooling operation of RoIs for mask branch
    roi_mask_post_cls: Type[
        MaskPostprocessing
    ] = None  #: define roi mask postprocessing strategy

    @staticmethod
    def get_roi_box_size(
        plan_arch: dict,
        model_cfg: dict,
    ):
        return model_cfg["roi_pooling"]["roi_box_size"]

    @staticmethod
    def get_roi_mask_size(
        plan_arch: dict,
        model_cfg: dict,
    ):
        return model_cfg["roi_pooling"]["roi_mask_size"]

    @classmethod
    def _build_rpn(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        plan_anchors: dict,
        **kwargs,
    ):
        if model_cfg["rpn_class_agnostic"]:
            _plan_arch = copy.deepcopy(plan_arch)
            _plan_arch["classifier_classes"] = 1
        else:
            logger.info("Running class sensitive RPN module!")
            _plan_arch = plan_arch
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
        conv: Callable,
    ):
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
        conv: Callable,
    ):
        name = cls.roi_regressor_cls.__name__
        kwargs = model_cfg["roi_regressor_kwargs"]
        logger.info(f"Building:: roi regressor {name}: {kwargs}")

        roi_channel_multiplier = model_cfg["roi_channel_multiplier"]
        regressor = cls.roi_regressor_cls(
            conv=conv,
            input_size=cls.get_roi_box_size(plan_arch, model_cfg),
            in_channels=plan_arch["fpn_channels"],
            internal_channels=int(roi_channel_multiplier * plan_arch["fpn_channels"]),
            **kwargs,
        )
        return regressor

    @classmethod
    def _build_roi_head(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        classifier,
        regressor,
        coder,
    ):
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
        conv: Callable,
    ):
        if cls.roi_masker_cls is not None:
            name = cls.roi_masker_cls.__name__
            kwargs = model_cfg["roi_masker_kwargs"]
            logger.info(f"Building:: roi masker {name}: {kwargs}")

            roi_mask_channel_multiplier = model_cfg["roi_mask_channel_multiplier"]
            masker = cls.roi_masker_cls(
                conv,
                in_channels=plan_arch["fpn_channels"],
                internal_channels=int(
                    roi_mask_channel_multiplier * plan_arch["fpn_channels"]
                ),
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
    ):
        pooler_name = cls.roi_box_pooler_cls.__name__
        feature_output_size = cls.get_roi_box_size(plan_arch, model_cfg)
        logger.info(
            f"Building:: box pooler {pooler_name} with output size {feature_output_size}"
        )

        box_pooler = cls.roi_box_pooler_cls(
            feature_output_size=feature_output_size,
        )
        return box_pooler

    @classmethod
    def _build_mask_pooler(
        cls,
        plan_arch: dict,
        model_cfg: dict,
    ):
        if cls.roi_mask_pooler_cls is not None:
            pooler_name = cls.roi_box_pooler_cls.__name__
            mask_feature_size = cls.get_roi_mask_size(plan_arch, model_cfg)
            mask_gt_size = [m * 2 for m in mask_feature_size]  # TODO # FIXME

            logger.info(
                f"Building:: mask pooler {pooler_name} with output "
                f"size {mask_feature_size} and gt size {mask_gt_size}"
            )

            mask_pooler = cls.roi_mask_pooler_cls(
                feature_output_size=mask_feature_size,
                mask_output_size=mask_gt_size,
            )
        else:
            mask_pooler = None
        return mask_pooler

    @classmethod
    def _build_roi_box_post(
        cls,
        plan_arch: dict,
        model_cfg: dict,
    ):
        name = cls.roi_box_post_cls.__name__
        kwargs = model_cfg["roi_box_post_kwargs"]
        logger.info(f"Building:: roi box postprocessing {name}: {kwargs}")

        roi_box_post = cls.roi_box_post_cls(
            num_foreground_classes=plan_arch["classifier_classes"],
            class_agnostic=cls.roi_regressor_cls.class_agnostic,
            **model_cfg["roi_box_post_kwargs"],
        )
        return roi_box_post

    @classmethod
    def _build_roi_mask_post(
        cls,
        plan_arch: dict,
        model_cfg: dict,
    ):
        if cls.roi_mask_post_cls is not None:
            name = cls.roi_mask_post_cls.__name__
            kwargs = model_cfg["roi_mask_post_kwargs"]
            logger.info(f"Building:: roi mask postprocessing {name}: {kwargs}")

            roi_mask_post = cls.roi_mask_post_cls(
                class_agnostic=cls.roi_masker_cls.class_agnostic,
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
    ):
        sampler_name = cls.roi_sampler_cls.__name__
        sampler_kwargs = model_cfg["roi_sampler_kwargs"]

        logger.info(f"Building:: roi sampler {sampler_name}: {sampler_kwargs}")
        return cls.roi_sampler_cls(**sampler_kwargs)

    @classmethod
    def _build_roi_module(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        box_head,
        box_pooler,
        box_post,
        matcher,
        sampler,
        # mask heads
        mask_head,
        mask_pooler,
        mask_post,
    ):
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
        # TODO
        """
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
        conv = Generator(cls.roi_conv_cls, plan_arch["dim"])

        roi_classifier = cls._build_roi_classifier(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            conv=conv,
        )
        roi_regressor = cls._build_roi_regressor(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            conv=conv,
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
            conv=conv,
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
            similarity_fn=box_iou,
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
        # TODO
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
        conv = Generator(cls.roi_conv_cls, plan_arch["dim"])

        heads = []
        matchers = []
        maskers = []
        for i in range(model_cfg["roi_cascade_stages"]):
            roi_classifier = cls._build_roi_classifier(
                plan_arch=plan_arch,
                model_cfg=model_cfg,
                conv=conv,
            )
            roi_regressor = cls._build_roi_regressor(
                plan_arch=plan_arch,
                model_cfg=model_cfg,
                conv=conv,
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
                conv=conv,
            )
            maskers.append(masker)

            # Matcher
            roi_matcher = cls.roi_matcher_cls(
                similarity_fn=box_iou,
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
