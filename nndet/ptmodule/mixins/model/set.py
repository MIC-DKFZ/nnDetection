# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Sequence, Type

from loguru import logger

from nndet.core.abstract import AbstractOneStageDetector
from nndet.core.boxes.criterions.base import BoxCriterion, ClassCriterion
from nndet.core.boxes.matcher1to1.base import BaseMatcher
from nndet.core.detr import BaseDETR
from nndet.core.post.detr import DETRBoxPost
from nndet.nn.backbone.abstract import AbstractBackbone
from nndet.nn.backbone.spine import SpineWrapper
from nndet.nn.heads.classifier.ffn import FFNClassifier
from nndet.nn.heads.detr.base import DETRHead
from nndet.nn.heads.regressor.ffn import FFNRegressor
from nndet.nn.heads.segmenter import Segmenter
from nndet.nn.layers.pos_embed.sine import BasePositionEmbedding
from nndet.nn.layers.wrapper import Generator
from nndet.nn.neck.abstract import AbstractNeck
from nndet.nn.neck.channel_mapper import ChannelMapper
from nndet.nn.transformer.abstract_transformer import AbstractTransformer
from nndet.nn.transformer.layers.abstract import (
    BaseTransformerDecoder,
    BaseTransformerEncoder,
)
from nndet.ptmodule.mixins.model import ModelMixin
from nndet.utils.typing import CONVSEQ, LINEARSEQ


class DETRModelMixin(ModelMixin):
    # define detector cls
    detector_cls: Type[AbstractOneStageDetector] = BaseDETR  #: define base detector class

    backbone_cls: Type[AbstractBackbone] = ...  #: define class for backbone
    backbone_conv_cls: Type[CONVSEQ] = ...  #: conv class used for backbone

    channel_mapper_cls: Type[ChannelMapper] = ...  #: map channels from backbone to transformer
    channel_mapper_conv_cls: Type[CONVSEQ] = ...  #: conv class used for channel mapper

    # transformer
    transformer_cls: Type[AbstractTransformer] = ...  #: define detector transformer architecture
    pos_embed_cls: BasePositionEmbedding = ...  #: define positional embedding for feature maps
    transformer_encoder_cls: BaseTransformerEncoder = ...  #: define encoder class of transformer
    transformer_decoder_cls: BaseTransformerDecoder = ...  #: define decoder class of transformer

    # head blocks
    head_cls: DETRHead = ...  #: main DETR head
    head_linear_cls: LINEARSEQ = ...  #: conv class used for head
    head_classifier_cls: FFNClassifier = ...  #: define classifier class
    head_regressor_cls: FFNRegressor = ...  #: define regressor class
    head_box_post_cls: DETRBoxPost = ...  #: define postprocessing strategy during inference

    matcher_cls: BaseMatcher = ...  #: matching algorithm
    matcher_class_criterion_cls: ClassCriterion = ...  #: criterion to compute class cost matrix
    # either reg or box criterion need to be set
    # reg criterion usually operates on encoded targets while box cirterion operates on raw boxes
    # there is no structural difference though and just a nomenclature
    matcher_reg_criterion_cls: Optional[BoxCriterion] = None  #: criterion to compute regression cost matrix
    matcher_box_criterion_cls: Optional[BoxCriterion] = None  #: criterion to compute regression cost matrix

    # [Optional]
    neck_cls: Optional[Type[AbstractNeck]] = None  #: [optional] define class for neck
    neck_conv_cls: Optional[Type[CONVSEQ]] = None  #: [optional] conv class used for neck

    # [Optional] Semantic Segmentation Head
    segmenter_cls: Optional[Type[Segmenter]] = None  #: [optional] segmentation head

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
        Build set prediction model e.g. DETR

        Args:
            model_cfg: model configuration
            plan_arch: architecture configuration

                ``"dim"`` int
                    number of spatial dimensions

                ``"in_channels"`` int
                    number of input channels

                ``"classifier_classes"`` int
                    number of classes

                ``"start_channels"`` int
                    number of start channels in backbone

                ``"conv_kernels"`` Sequence[Union[Tuple[int], int]]
                    kernel sizes of convolutions for each stage/level

                ``"strides"`` Sequence[Union[Tuple[int], int]]
                    stride of downsampling block for each stage/level
                    Downsampling is alwyas performed at the beginning of the blocks.
                    First stage/level is always full resolution.

                ``"seg_classes"`` int
                    (optional) number of classes

                ``"fpn_channels"`` int
                    (optional) number of channels to use for FPN

                ``"decoder_levels"`` int
                    (optional) decoder levels to user for detection

            plan_anchors: anchor configuration (not used)
            patch_size: patch size for training. Defaults to None.
        """
        if "plan_arch_overwrites" in model_cfg:
            logger.error("plan_arch_overwrites found in model config, this is not supported anymore.")
            raise NotImplementedError("plan_arch_overwrites not supported anymore")
        backbone = cls._build_backbone(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            patch_size=patch_size,
        )

        # transformer
        hidden_dim = model_cfg["transformer"]["hidden_dim"]
        pos_embed_kwargs = model_cfg.get("pos_embed", {})
        logger.info(f"Building:: Pos Embed {cls.pos_embed_cls.__name__} with {pos_embed_kwargs}")
        pos_embed = cls.pos_embed_cls(
            dim=plan_arch["dim"],
            num_pos_feats=hidden_dim,
            **pos_embed_kwargs,
        )

        channel_mapper = cls._build_channel_mapper(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            channels=backbone.get_channels(),
        )

        classifier = cls._build_classifier(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
        )
        regressor = cls._build_regressor(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
        )
        transformer = cls._build_transformer(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            classifier=classifier,
            regressor=regressor,
        )

        # head & matching
        matcher = cls._build_matcher(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
        )
        box_post = cls._build_box_post(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
        )
        head = cls._build_head(
            plan_arch,
            model_cfg,
            classifier=classifier,
            regressor=regressor,
            matcher=matcher,
            box_post=box_post,
        )

        # build optional modules
        # these are not part of the original DETR architecture
        if cls.has_neck():
            neck = cls._build_neck(
                backbone=backbone,
                plan_arch=plan_arch,
                model_cfg=model_cfg,
            )
            backbone = SpineWrapper(
                backbone=backbone,
                neck=neck,
            )

        if cls.has_segmenter():
            segmenter = cls._build_segmenter(
                plan_arch=plan_arch,
                model_cfg=model_cfg,
                backbone=backbone,
            )
        else:
            segmenter = None

        detection_per_img = cls._get_detection_per_img(plan_arch=plan_arch, model_cfg=model_cfg)
        return cls.detector_cls(
            backbone=backbone,
            transformer=transformer,
            channel_mapper=channel_mapper,
            head=head,
            pos_embed=pos_embed,
            hidden_dim=hidden_dim,
            query_dim=hidden_dim,
            segmenter=segmenter,
            detection_per_img=detection_per_img,
            two_stage=model_cfg["transformer"].get("two_stage", False),
            use_pos_queries=model_cfg["transformer"].get("use_pos_queries", False),
        )

    @classmethod
    def _get_detection_per_img(cls, plan_arch: dict, model_cfg: dict) -> int:
        est_instances_patch = plan_arch["est_instances_patch"]["perc95"]
        return max(model_cfg["detector"]["min_detection_per_img"], 3 * est_instances_patch)

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
        backbone_kwargs = {}
        if "backbone_kwargs" in model_cfg:
            backbone_kwargs = model_cfg["backbone_kwargs"]
        backbone: AbstractBackbone = cls.backbone_cls.from_config_plan(
            conv=conv,
            backbone_cfg=backbone_kwargs,
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
    def _build_channel_mapper(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        channels: List[int],
    ) -> ChannelMapper:
        """
        Build class to process backbone feature maps for transformer

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings
            channels: number of output channels of backbone

        Returns:
            ChannelMapper: module to perform channel mapping
        """
        conv = Generator(cls.channel_mapper_conv_cls, plan_arch["dim"])
        channel_mapper_kwargs = model_cfg["channel_mapper_kwargs"]

        num_in_features = model_cfg["transformer"]["num_feature_levels"]
        num_total_levels = num_in_features + channel_mapper_kwargs.pop("extra_levels")
        kernel_size = channel_mapper_kwargs.pop("kernel_size")
        conv_kwargs = channel_mapper_kwargs.pop("conv_kwargs")

        return cls.channel_mapper_cls(
            conv=conv,
            in_channels=channels,
            num_in_features=num_in_features,
            kernel_size=kernel_size,
            out_channels=model_cfg["transformer"]["hidden_dim"],
            num_outs=num_total_levels,
            **channel_mapper_kwargs,
            **conv_kwargs,
        )

    @classmethod
    def _build_transformer(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        classifier: Optional[FFNClassifier] = None,
        regressor: Optional[FFNRegressor] = None,
    ) -> AbstractTransformer:
        """
        Build transformer (encoder & decoder)

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings

        Returns:
            AbstractTransformer: transformer module
        """
        encoder_kwargs = model_cfg["transformer_encoder_kwargs"]
        encoder = cls.transformer_encoder_cls(
            embed_dim=model_cfg["transformer"]["hidden_dim"],
            num_heads=encoder_kwargs["attention_heads"],
            num_layers=encoder_kwargs["num_layers"],
            attn_dropout=encoder_kwargs["attn_dropout"],
            proj_dropout=encoder_kwargs["proj_dropout"],
            feedforward_dim=encoder_kwargs["dim_feedforward"],
            ffn_dropout=encoder_kwargs["ffn_dropout"],
            post_norm=encoder_kwargs["post_norm"],
            dim=plan_arch["dim"],
            batch_first=cls.transformer_cls.is_batch_first(),
        )
        decoder_kwargs = model_cfg["transformer_decoder_kwargs"]
        decoder = cls.transformer_decoder_cls(
            embed_dim=model_cfg["transformer"]["hidden_dim"],
            num_heads=decoder_kwargs["attention_heads"],
            num_layers=decoder_kwargs["num_layers"],
            attn_dropout=decoder_kwargs["attn_dropout"],
            proj_dropout=decoder_kwargs["proj_dropout"],
            feedforward_dim=decoder_kwargs["dim_feedforward"],
            ffn_dropout=decoder_kwargs["ffn_dropout"],
            post_norm=decoder_kwargs["post_norm"],
            dim=plan_arch["dim"],
            batch_first=cls.transformer_cls.is_batch_first(),
        )
        return cls.transformer_cls(
            encoder=encoder,
            decoder=decoder,
        )

    @classmethod
    def _build_classifier(
        cls,
        plan_arch: dict,
        model_cfg: dict,
    ) -> FFNClassifier:
        """
        Build classifier module for predictions

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings

        Returns:
            FFNClassifier: classification module
        """
        num_classes = plan_arch["classifier_classes"]
        name = cls.head_classifier_cls.__name__
        kwargs = model_cfg["head_classifier_kwargs"]

        logger.info(f"Building:: classifier {name} with {kwargs}")
        return cls.head_classifier_cls(
            linear=cls.head_linear_cls,
            in_channels=model_cfg["transformer"]["hidden_dim"],
            num_classes=num_classes,
            **kwargs,
        )

    @classmethod
    def _build_regressor(
        cls,
        plan_arch: dict,
        model_cfg: dict,
    ) -> FFNRegressor:
        """
        Build regressor module for predictions

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings

        Returns:
            FFNRegressor: regression module
        """
        dim = plan_arch["dim"]
        name = cls.head_regressor_cls.__name__
        kwargs = model_cfg["head_regressor_kwargs"]

        logger.info(f"Building:: regressor {name} with {kwargs}")
        return cls.head_regressor_cls(
            linear=cls.head_linear_cls,
            in_channels=model_cfg["transformer"]["hidden_dim"],
            dim=dim,
            **kwargs,
        )

    @classmethod
    def _build_matcher(
        cls,
        plan_arch: dict,
        model_cfg: dict,
    ) -> BaseMatcher:
        """
        Build matching module to assign ground truth objects to predictions

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings

        Raises:
            RuntimeError: raised if neither a box nor a regression criterion
                is provided

        Returns:
            BaseMatcher: object to perform matching
        """
        if cls.matcher_box_criterion_cls is None and cls.matcher_reg_criterion_cls is None:
            raise RuntimeError("Need at least one regression or box criterion!")

        class_criterions = []
        box_criterions = []

        name = cls.matcher_class_criterion_cls.__name__
        class_kwargs = model_cfg["matcher_class_criterion_kwargs"]
        logger.info(f"Building:: matcher class criterion {name} with {class_kwargs}")
        class_criterions.append(cls.matcher_class_criterion_cls(**class_kwargs))

        if cls.matcher_reg_criterion_cls is not None:
            name = cls.matcher_reg_criterion_cls.__name__
            reg_kwargs = model_cfg["matcher_reg_criterion_kwargs"]
            logger.info(f"Building:: matcher reg criterion {name} with {reg_kwargs}")
            box_criterions.append(cls.matcher_reg_criterion_cls(**reg_kwargs))

        if cls.matcher_box_criterion_cls is not None:
            name = cls.matcher_box_criterion_cls.__name__
            box_kwargs = model_cfg["matcher_box_criterion_kwargs"]
            logger.info(f"Building:: matcher box criterion {name} with {box_kwargs}")
            box_criterions.append(cls.matcher_box_criterion_cls(**box_kwargs))

        name = cls.matcher_cls.__name__
        matcher_kwargs = model_cfg["matcher_kwargs"]
        logger.info(f"Building:: matcher {name} with {matcher_kwargs}")

        return cls.matcher_cls(
            class_criterion=class_criterions,
            box_criterion=box_criterions,
            **matcher_kwargs,
        )

    @classmethod
    def _build_box_post(
        cls,
        plan_arch: dict,
        model_cfg: dict,
    ) -> DETRBoxPost:
        """
        Postprocessing module for predictions

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings

        Returns:
            DETRBoxPost: module to perform postprocessing
        """
        name = cls.head_box_post_cls.__name__
        kwargs = model_cfg["head_box_post_kwargs"]
        kwargs["topk"] = cls._get_detection_per_img(plan_arch=plan_arch, model_cfg=model_cfg)

        logger.info(f"Building:: box post {name} with {kwargs}")
        return cls.head_box_post_cls(**kwargs)

    @classmethod
    def _build_head(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        classifier: FFNClassifier,
        regressor: FFNRegressor,
        matcher: BaseMatcher,
        box_post: DETRBoxPost,
    ) -> DETRHead:
        name = cls.head_cls.__name__
        kwargs = model_cfg["head_kwargs"]

        logger.info(f"Building:: head {name} with {kwargs}")
        return cls.head_cls(
            classifier=classifier,
            regressor=regressor,
            matcher=matcher,
            box_post=box_post,
            **kwargs,
        )

    @classmethod
    def has_neck(cls):
        """
        Optional: Check if configuration should have a neck

        Returns:
            bool: True if detector needs neck, False othterwise
        """
        has_neck = cls.neck_cls is not None
        if has_neck and cls.neck_conv_cls is None:
            raise ValueError("Neck class was provided without conv class.")
        return has_neck

    @classmethod
    def _build_neck(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        backbone: AbstractBackbone,
    ) -> AbstractNeck:
        """
        Optional: Build neck network

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
    def has_segmenter(cls):
        """
        Optional: Check if configuration should have a segmenter

        Returns:
            bool: True if detector needs segemetner, False othterwise
        """
        return cls.segmenter_cls is not None

    @classmethod
    def _build_segmenter(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        backbone: AbstractBackbone,
    ) -> Segmenter:
        """
        Optional: Build segmenter head

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings
            neck: neck instance

        Returns:
            SegmenterType: segmenter head
        """
        name = cls.segmenter_cls.__name__
        kwargs = model_cfg.get("segmenter_kwargs", {})
        conv = Generator(cls.backbone_conv_cls, plan_arch["dim"])

        logger.info(f"Building:: segmenter {name} {kwargs}")
        segmenter = cls.segmenter_cls(
            conv,
            seg_classes=plan_arch["seg_classes"],
            in_channels=backbone.get_channels(),
            decoder_levels=plan_arch["decoder_levels"],
            **kwargs,
        )
        return segmenter


class ConditionalDETRModelMixin(DETRModelMixin):
    @classmethod
    def _build_transformer(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        classifier: Optional[FFNClassifier] = None,
        regressor: Optional[FFNRegressor] = None,
    ) -> AbstractTransformer:
        """
        Build transformer (encoder & decoder)

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings

        Returns:
            AbstractTransformer: transformer module
        """
        encoder_kwargs = model_cfg["transformer_encoder_kwargs"]
        encoder = cls.transformer_encoder_cls(
            embed_dim=model_cfg["transformer"]["hidden_dim"],
            num_heads=encoder_kwargs["attention_heads"],
            num_layers=encoder_kwargs["num_layers"],
            attn_dropout=encoder_kwargs["attn_dropout"],
            proj_dropout=encoder_kwargs["proj_dropout"],
            feedforward_dim=encoder_kwargs["dim_feedforward"],
            ffn_dropout=encoder_kwargs["ffn_dropout"],
            post_norm=encoder_kwargs["post_norm"],
            dim=plan_arch["dim"],
            batch_first=cls.transformer_cls.is_batch_first(),
        )
        decoder_kwargs = model_cfg["transformer_decoder_kwargs"]
        decoder = cls.transformer_decoder_cls(
            embed_dim=model_cfg["transformer"]["hidden_dim"],
            num_heads=decoder_kwargs["attention_heads"],
            num_layers=decoder_kwargs["num_layers"],
            attn_dropout=decoder_kwargs["attn_dropout"],
            proj_dropout=decoder_kwargs["proj_dropout"],
            feedforward_dim=decoder_kwargs["dim_feedforward"],
            ffn_dropout=decoder_kwargs["ffn_dropout"],
            post_norm=decoder_kwargs["post_norm"],
            dim=plan_arch["dim"],
            ffn_regressor_cls=cls.head_regressor_cls,
            batch_first=cls.transformer_cls.is_batch_first(),
        )
        return cls.transformer_cls(
            encoder=encoder,
            decoder=decoder,
        )


class DeformableSetModelMixin(DETRModelMixin):
    @classmethod
    def _build_transformer(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        classifier: Optional[FFNClassifier] = None,
        regressor: Optional[FFNRegressor] = None,
    ):
        if model_cfg["transformer"]["two_stage"]:
            encoder_classifier = classifier
            encoder_regressor = regressor
            decoder_regressor = regressor
        else:
            encoder_classifier, encoder_regressor, decoder_regressor = None, None, None

        encoder_kwargs = model_cfg["transformer_encoder_kwargs"]
        encoder = cls.transformer_encoder_cls(
            embed_dim=encoder_kwargs["hidden_dim"],
            num_heads=encoder_kwargs["attention_heads"],
            num_layers=encoder_kwargs["num_layers"],
            attn_dropout=encoder_kwargs["attn_dropout"],
            proj_dropout=encoder_kwargs["proj_dropout"],
            feedforward_dim=encoder_kwargs["dim_feedforward"],
            ffn_dropout=encoder_kwargs["ffn_dropout"],
            post_norm=encoder_kwargs["post_norm"],
            num_feature_levels=model_cfg["transformer"]["num_feature_levels"],
            num_points=encoder_kwargs["num_points"],
            dim=plan_arch["dim"],
            batch_first=cls.transformer_cls.is_batch_first(),
        )
        decoder_kwargs = model_cfg["transformer_decoder_kwargs"]
        decoder = cls.transformer_decoder_cls(
            embed_dim=decoder_kwargs["hidden_dim"],
            num_heads=decoder_kwargs["attention_heads"],
            num_layers=decoder_kwargs["num_layers"],
            attn_dropout=decoder_kwargs["attn_dropout"],
            proj_dropout=decoder_kwargs["proj_dropout"],
            feedforward_dim=decoder_kwargs["dim_feedforward"],
            ffn_dropout=decoder_kwargs["ffn_dropout"],
            num_feature_levels=model_cfg["transformer"]["num_feature_levels"],
            num_points=decoder_kwargs["num_points"],
            dim=plan_arch["dim"],
            regressor=decoder_regressor,
            batch_first=cls.transformer_cls.is_batch_first(),
        )

        transformer_kwargs = model_cfg["transformer"]["transformer_kwargs"]
        transformer_kwargs["two_stage_num_proposals"] = cls._get_detection_per_img(
            plan_arch=plan_arch, model_cfg=model_cfg
        )
        return cls.transformer_cls(
            encoder=encoder,
            decoder=decoder,
            classifier=encoder_classifier,
            regressor=encoder_regressor,
            num_feature_levels=model_cfg["transformer"]["num_feature_levels"],
            two_stage=model_cfg["transformer"]["two_stage"],
            **transformer_kwargs,
        )
