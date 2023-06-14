# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from nndet.nn.heads.classifier.ffn import FFNClassifier
from nndet.nn.heads.regressor.ffn import FFNRegressor
from nndet.ptmodule.mixins.model.detr import SetModelMixin


class DeformableSetModelMixin(SetModelMixin):
    @classmethod
    def _build_transformer(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        classifier: Optional[FFNClassifier] = None,
        regressor: Optional[FFNRegressor] = None,
    ):
        if model_cfg["two_stage"]:
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
            num_feature_levels=model_cfg["num_feature_levels"],
            num_points=encoder_kwargs["num_points"],
            dim=plan_arch["dim"],
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
            post_norm=decoder_kwargs["post_norm"],
            num_feature_levels=model_cfg["num_feature_levels"],
            num_points=encoder_kwargs["num_points"],
            dim=plan_arch["dim"],
            regressor=decoder_regressor,
        )
        return cls.transformer_cls(
            encoder=encoder,
            decoder=decoder,
            classifier=encoder_classifier,
            regressor=encoder_regressor,
            num_feature_levels=model_cfg["num_feature_levels"],
            two_stage=model_cfg["two_stage"],
            two_stage_num_proposals=model_cfg["detection_per_img"],
        )
