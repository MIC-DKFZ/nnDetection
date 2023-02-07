from typing import Optional, Sequence

import torch.nn as nn
from loguru import logger

from nndet.core.boxes.matcher1to1.base import BaseMatcher
from nndet.core.post.detr import DETRBoxPost
from nndet.nn.backbone.spine import SpineWrapper
from nndet.nn.heads.detr.base import DETRHead
from nndet.nn.transformer.deformable_transformer import DeformableDETRTransformer
from nndet.nn.transformer.layers.deformable_detr import (
    DeformableDETRTransformerDecoder,
    DeformableDETRTransformerEncoder,
)
from nndet.ptmodule.mixins.model.detr import SetModelMixin


class DeformableSetModelMixin(SetModelMixin):
    def from_config_plan(
        cls,
        model_cfg: dict,
        plan_arch: dict,
        plan_anchors: dict,
        patch_size: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        if "plan_arch_overwrites" in model_cfg:
            logger.info(f"Architecture overwrites: {model_cfg['plan_arch_overwrites']} ")
            plan_arch.update(model_cfg["plan_arch_overwrites"])
        logger.info(
            f"Start channels: {plan_arch['start_channels']}; "
            f"head channels: {plan_arch['head_channels']}; "
            f"fpn channels: {plan_arch['fpn_channels']}"
        )
        backbone = cls._build_backbone(plan_arch, model_cfg)

        # transformer
        hidden_dim = model_cfg["hidden_dim"]
        pos_embed_kwargs = model_cfg.get("pos_embed", {})
        logger.info(f"Building:: Pos Embed {cls.pos_embed_cls.__name__} with {pos_embed_kwargs}")
        pos_embed = cls.pos_embed_cls(
            dim=plan_arch["dim"],
            num_pos_feats=hidden_dim,
            **pos_embed_kwargs,
        )
        transformer_encoder = DeformableDETRTransformerEncoder(
            embed_dim=model_cfg["hidden_dim"],
            num_heads=model_cfg["attention_heads"],
            feedforward_dim=model_cfg["dim_feedforward"],
            attn_dropout=model_cfg["transformer_attn_dropout"],
            ffn_dropout=model_cfg["transformer_ffn_dropout"],
            num_layers=model_cfg["num_encoder_layers"],
            num_feature_levels=model_cfg["num_feature_levels"],
            num_points=model_cfg["num_points"],
        )
        transformer_decoder = DeformableDETRTransformerDecoder(
            embed_dim=model_cfg["hidden_dim"],
            num_heads=model_cfg["attention_heads"],
            attn_dropout=model_cfg["transformer_attn_dropout"],
            ffn_dropout=model_cfg["transformer_ffn_dropout"],
            feedforward_dim=model_cfg["dim_feedforward"],
            num_layers=model_cfg["num_decoder_layers"],
            num_feature_levels=model_cfg["num_feature_levels"],
            num_points=model_cfg["num_points"],
        )
        transformer = DeformableDETRTransformer(
            encoder=transformer_encoder,
            decoder=transformer_decoder,
            num_feature_levels=model_cfg["num_feature_levels"],
            as_two_stage=model_cfg["two_stage"],
            two_stage_num_proposals=model_cfg["detection_per_img"],
        )

        # head & matching
        classifier = cls._build_classifier(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
        )
        regressor = cls._build_regressor(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
        )
        matcher = cls._build_matcher(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
        )
        box_post = cls._build_box_post(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
        )
        """num_pred = (
            model_cfg["num_decoder_layers"] + 1
            if model_cfg["two_stage"]
            else model_cfg["num_decoder_layers"]
        )
        if model_cfg["box_refine"]:
            classifier = nn.ModuleList(
                [copy.deepcopy(classifier) for i in range(num_pred)]
            )
            regressor = nn.ModuleList(
                [copy.deepcopy(regressor) for i in range(num_pred)]
            )
            # TODO Understand this
            nn.init.constant_(regressor[0].mlp[-1].fc.bias.data[3:], -2.0)
        else:
            nn.init.constant_(regressor.mlp[-1].fc.bias.data[3:], -2.0)
            classifier = nn.ModuleList([classifier for i in range(num_pred)])
            regressor = nn.ModuleList([regressor for i in range(num_pred)])"""

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

        # Parse model kwargs
        model_kwargs = {}
        if "kwargs" in model_cfg.keys():
            model_kwargs.update(model_cfg["kwargs"])

        return cls.detector_cls(
            backbone=backbone,
            transformer=transformer,
            head=head,
            pos_embed=pos_embed,
            hidden_dim=hidden_dim,
            detection_per_img=model_cfg["detection_per_img"],
            query_dim=model_cfg["query_dim"],
            segmenter=segmenter,
            num_feature_levels=model_cfg["num_feature_levels"],
            **model_kwargs,
        )

    @classmethod
    def _build_head(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        classifier: nn.ModuleList,
        regressor: nn.ModuleList,
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
