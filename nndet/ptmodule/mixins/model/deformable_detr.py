from nndet.ptmodule.mixins.model.detr import SetModelMixin


class DeformableSetModelMixin(SetModelMixin):
    @classmethod
    def _build_transformer(
        cls,
        plan_arch: dict,
        model_cfg: dict,
    ):
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
        )
        return cls.transformer_cls(
            encoder=encoder,
            decoder=decoder,
            num_feature_levels=model_cfg["num_feature_levels"],
            as_two_stage=model_cfg["two_stage"],
            two_stage_num_proposals=model_cfg["detection_per_img"],
        )
