import copy
from abc import ABC, abstractmethod

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
