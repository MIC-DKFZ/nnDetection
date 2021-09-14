import copy
from abc import ABC, abstractmethod
from typing import Callable

from loguru import logger

from nndet.arch.conv import Generator
from nndet.arch.decoder.base import DecoderType
from nndet.arch.encoder.abstract import EncoderType
from nndet.arch.heads.classifier import DenseClassifierType
from nndet.arch.heads.comb.base import AnchorHeadType
from nndet.arch.heads.regressor import DenseRegressorType
from nndet.arch.heads.segmenter import SegmenterType
from nndet.core.boxes.anchors import AnchorGeneratorType, get_anchor_generator
from nndet.core.boxes.coder import BoxCoderND, CoderType
from nndet.core.boxes.ops import box_iou


class ModelMixin(ABC):
    @classmethod
    @abstractmethod
    def from_config_plan(
        cls,
        model_cfg: dict,
        plan_arch: dict,
        plan_anchors: dict,
        **kwargs,
    ):
        """
        Create Configurable RetinaUNet

        Args:
            model_cfg: model configurations.
                Exact parameters depend on subclass.
            plan_arch: plan architecture
                Exact parameters depend on subclass.
            plan_anchors: parameters for anchors
                Exact parameters depend on subclass.
            **kwargs:
        """
        raise NotImplementedError


class SingleStageMixin(ModelMixin):
    """
    This class provides the template to build a detection model
    """

    # define detector cls
    detector_cls = ...

    backbone_cls = ...  # define class for backbone
    backbone_conv_cls = ...  # conv class used for backbone
    backbone_block = ...  # define central building block of backbone

    neck_cls = ...  # define class for neck
    neck_conv_cls = ...  # conv class used for neck

    head_cls = ...  # define class for head
    head_conv_cls = ...  # conv class used for head
    head_classifier_cls = ...  # define class for head classifier
    head_regressor_cls = ...  # define class for head regressor
    # [optional] sampler class for negative mining
    # if None: no sampler will be given to the head
    head_sampler_cls = None

    matcher_cls = ...  # define class to match anchors to ground truth
    segmenter_cls = None  # [optional] segmentation head as in RetinaUNet

    @classmethod
    def from_config_plan(
        cls,
        model_cfg: dict,
        plan_arch: dict,
        plan_anchors: dict,
        **kwargs,
    ):
        """
        Create Configurable RetinaUNet

        Args:
            model_cfg: model configurations
                See example configs for more info
            plan_arch: plan architecture
                `dim` (int): number of spatial dimensions
                `in_channels` (int): number of input channels
                `classifier_classes` (int): number of classes
                `seg_classes` (int): number of classes
                `start_channels` (int): number of start channels in backbone
                `fpn_channels` (int): number of channels to use for FPN
                `head_channels` (int): number of channels to use for head
                `decoder_levels` (int): decoder levels to user for detection
            plan_anchors: parameters for anchors (see
                :class:`AnchorGenerator` for more info)
                    `stride`: stride
                    `aspect_ratios`: aspect ratios
                    `sizes`: sized for 2d acnhors
                    (`zsizes`: additional z sizes for 3d)
            **kwargs:
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

        detections_per_img = plan_arch.get("detections_per_img", 100)
        score_thresh = plan_arch.get("score_thresh", 0)
        topk_candidates = plan_arch.get("topk_candidates", 10000)
        remove_small_boxes = plan_arch.get("remove_small_boxes", 0.01)
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
    ) -> EncoderType:
        """
        Build backbone network

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings

        Returns:
            EncoderType: backbone instance
        """
        conv = Generator(cls.backbone_conv_cls, plan_arch["dim"])
        logger.info(
            f"Building:: backbone {cls.backbone_cls.__name__}: {model_cfg['backbone_kwargs']} "
        )
        backbone = cls.backbone_cls(
            conv=conv,
            conv_kernels=plan_arch["conv_kernels"],
            strides=plan_arch["strides"],
            block_cls=cls.backbone_block,
            in_channels=plan_arch["in_channels"],
            start_channels=plan_arch["start_channels"],
            stage_kwargs=None,
            max_channels=plan_arch.get("max_channels", 320),
            **model_cfg["backbone_kwargs"],
        )
        return backbone

    @classmethod
    def _build_neck(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        backbone: EncoderType,
    ) -> DecoderType:
        """
        Build neck network

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings

        Returns:
            DecoderType: neck instance
        """
        conv = Generator(cls.neck_conv_cls, plan_arch["dim"])
        logger.info(
            f"Building:: neck {cls.neck_cls.__name__}: {model_cfg['neck_kwargs']}"
        )
        neck = cls.neck_cls(
            conv=conv,
            conv_kernels=plan_arch["conv_kernels"],
            strides=backbone.get_strides(),
            in_channels=backbone.get_channels(),
            decoder_levels=plan_arch["decoder_levels"],
            fixed_out_channels=plan_arch["fpn_channels"],
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
        if cls.has_sampler:
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
    def has_sampler(cls):
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
        return cls.segmenter_cls is not None

    @classmethod
    def _build_segmenter(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        neck: DecoderType,
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
    full_detector_cls = ...  # Two stage detector class RCNN

    # RoI classes
    roi_conv_cls = ...
    roi_module_cls = ...  # RoIModule
    roi_head_cls = ...  # RoIBoxHead
    roi_classifier_cls = ...  # RoIClassifierTwoMLP
    roi_regressor_cls = ...  # RoIRegressorConv

    roi_matcher_cls = ...  # IoUMatcher
    roi_sampler_cls = ...  # BalancedHardNegativeSampler
    roi_box_pooler_cls = ...  # RoIAlignNaiveAssign

    # optional mask branches
    roi_masker_cls = None  # BCESingleMasker
    roi_mask_pooler_cls = None  # RoIAlignNaiveAssign

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

        classifier = cls.roi_classifier_cls(
            conv=conv,
            input_size=cls.get_roi_box_size(plan_arch, model_cfg),
            in_channels=plan_arch["fpn_channels"],
            internal_channels=plan_arch["fpn_channels"],
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

        regressor = cls.roi_regressor_cls(
            conv=conv,
            input_size=cls.get_roi_box_size(plan_arch, model_cfg),
            in_channels=plan_arch["fpn_channels"],
            internal_channels=plan_arch["fpn_channels"],
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

            masker = cls.roi_masker_cls(
                conv,
                in_channels=plan_arch["fpn_channels"],
                internal_channels=plan_arch["fpn_channels"],
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
                f"Building:: box pooler {pooler_name} with output "
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
        matcher,
        box_pooler,
        sampler,
        # mask heads
        mask_head,
        mask_pooler,
    ):
        roi_module_name = cls.roi_module_cls.__name__
        roi_module_kwargs = model_cfg["roi_module_kwargs"]

        logger.info(f"Building:: roi module {roi_module_name}: {roi_module_kwargs}")

        roi_module = cls.roi_module_cls(
            box_head=box_head,
            matcher=matcher,
            box_pooler=box_pooler,
            sampler=sampler,
            num_classes=plan_arch["classifier_classes"],
            decoder_levels=plan_arch["decoder_levels"],
            # mask heads
            mask_head=mask_head,
            mask_pooler=mask_pooler,
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
        **kwargs,
    ):
        # build RPN
        rpn = cls._build_rpn(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            plan_anchors=plan_anchors,
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

        roi_module = cls._build_roi_module(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            box_head=roi_head,
            matcher=roi_matcher,
            box_pooler=box_pooler,
            sampler=roi_sampler,
            # mask heads
            mask_head=masker,
            mask_pooler=mask_pooler,
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
        **kwargs,
    ):
        # build RPN
        rpn = super().from_config_plan(
            model_cfg=model_cfg,
            plan_arch=plan_arch,
            plan_anchors=plan_anchors,
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
            matcher=matchers,
            box_pooler=box_pooler,
            sampler=roi_sampler,
            # mask heads
            mask_head=maskers,
            mask_pooler=mask_pooler,
        )
        return cls.full_detector_cls(
            rpn=rpn,
            roi_module=roi_module,
        )
