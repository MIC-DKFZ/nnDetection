import os
from pathlib import Path
from typing import Optional, Sequence, Type

import torch
from loguru import logger

from nndet.core.abstract import AbstractOneStageDetector
from nndet.core.detr import BaseDETR
from nndet.nn.backbone.abstract import AbstractBackbone
from nndet.nn.layers.pos_embed.sine import BasePositionEmbedding
from nndet.nn.layers.wrapper import Generator
from nndet.ptmodule.mixins.model import ModelMixin
from nndet.utils.typing import CONVSEQ


class DETRMixin(ModelMixin):
    # define detector cls
    detector_cls: Type[AbstractOneStageDetector] = BaseDETR

    backbone_cls: Type[AbstractBackbone] = ...  # define class for backbone
    backbone_conv_cls: Type[CONVSEQ] = ...  # conv class used for backbone
    # transformer
    pos_embed_cls: BasePositionEmbedding = ...
    transformer_cls = ...
    # head blocks
    head_cls = ...  # main head

    @classmethod
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
        pos_embed = cls.pos_embed_cls(dim=plan_arch["dim"], num_pos_feats=hidden_dim)
        transformer = cls.transformer_cls(
            d_model=hidden_dim,
            nhead=model_cfg["attention_heads"],
            num_encoder_layers=model_cfg["num_encoder_layers"],
            num_decoder_layers=model_cfg["num_decoder_layers"],
            dim_feedforward=model_cfg["dim_feedforward"],
            **model_cfg["transformer_kwargs"],
        )

        # head
        head = cls._build_head(plan_arch, model_cfg)

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
            **model_kwargs,
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

        # If configured, load weights from a nnDetection pretrained encoder
        # FIXME this gives error when continuing training
        if "pretrained_encoder" in model_cfg:
            if model_cfg["pretrained_encoder"]:
                path = Path(model_cfg["pretrain_dir"]) / "model_best.ckpt"
                assert os.path.exists(path), f"No state dict found at {path}"
                pretrain_dict = torch.load(path)["state_dict"]
                weight_dict = {
                    k[15:]: v  # Copy all keys and values from the pretrained state dict
                    for k, v in pretrain_dict.items()  # backbone. k[15:] filters out the "model.backbone." which is not
                    if k[:15] == "model.backbone."  # needed to load the weights into the encoder
                }
                backbone.load_state_dict(weight_dict)
                logger.info(f"Using Pretrained Model Weights for {cls.backbone_cls.__name__} from {path}.")
        return backbone

    @classmethod
    def _build_head(cls, plan_arch, model_cfg):
        # Obtain needed parameters
        num_classes = plan_arch["classifier_classes"]
        hidden_dim = model_cfg["hidden_dim"]
        losses = ["labels", "boxes", "cardinality"]
        # Build all necessary modules
        classifier = cls.head_cls.classifier_cls(in_features=hidden_dim, num_classes=num_classes)
        regressor = cls.head_cls.regressor_cls(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=6,
            num_layers=model_cfg["regressor_depth"],
        )

        matcher_kwargs = {}
        if "matcher_kwargs" in model_cfg:
            matcher_kwargs = model_cfg["matcher_kwargs"]

        matcher = cls.head_cls.matcher_cls(
            model_cfg["loss_ce_matcher"],
            model_cfg["loss_bbox_matcher"],
            model_cfg["loss_giou_matcher"],
            **matcher_kwargs,
        )

        weight_dict = {
            "loss_ce": model_cfg["loss_ce"],
            "loss_bbox": model_cfg["loss_bbox"],
            "loss_giou": model_cfg["loss_giou"],
        }
        head_kwargs = {}
        if "head_kwargs" in model_cfg:
            head_kwargs = model_cfg["head_kwargs"]
        return cls.head_cls(
            classifier,
            regressor,
            matcher,
            weight_dict=weight_dict,
            losses=losses,
            num_classes=num_classes,
            **head_kwargs,
        )
