import re
from typing import Dict, List, Optional, Tuple, Type

import torch
from loguru import logger

from nndet.core.boxes.criterions.base import BoxCriterion, ClassCriterion
from nndet.core.boxes.criterions.box import GIoUCenterBoxCriterion, L1RegCriterion
from nndet.core.boxes.criterions.cls import FocalClassCriterionSigmoid
from nndet.core.boxes.matcher1to1.base import BaseMatcher
from nndet.core.boxes.matcher1to1.hungarian import HungarianMatcher
from nndet.core.post.detr import DETRBoxPost, TopKBoxPost
from nndet.nn.backbone.abstract_ResEnc import WrapperResEncAbstractBackbone
from nndet.nn.backbone.blueprints.ResEnc import (
    ResidualEncoderUNetbackBoneWrapper,
    ResidualEncoderUNetbackBoneWrapper_dyn,
)
from nndet.nn.heads.classifier.ffn import FFNClassifier, FocalFFNClassifier
from nndet.nn.heads.detr.base import DETRHead
from nndet.nn.heads.detr.deformable_detr import DeformableDETRHead
from nndet.nn.heads.regressor.ffn import FFNRegressor, L1UGIoUFFNRegressor
from nndet.nn.layers.conv import ConvInstanceRelu
from nndet.nn.layers.conv.group import ConvGroupRelu
from nndet.nn.layers.linear import LayerLinearReluDrop
from nndet.nn.layers.pos_embed.base import BasePositionEmbedding
from nndet.nn.layers.pos_embed.sine import PositionEmbeddingSine
from nndet.nn.neck.channel_mapper import ChannelMapper
from nndet.nn.transformer.abstract_transformer import AbstractTransformer
from nndet.nn.transformer.deformable_transformer import DeformableDETRTransformer
from nndet.nn.transformer.layers.abstract import (
    BaseTransformerDecoder,
    BaseTransformerEncoder,
)
from nndet.nn.transformer.layers.deformable_detr import (
    DeformableDETRTransformerDecoder,
    DeformableDETRTransformerEncoder,
)
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.mixins.evaluation.boxes import BoxEvalMixin
from nndet.ptmodule.mixins.model.set import DeformableSetModelMixin_ResEnc
from nndet.ptmodule.mixins.prediction.boxes import BoxPredictionMixinV2
from nndet.ptmodule.mixins.prepare.boxes import BoxesPrepareMixin
from nndet.ptmodule.mixins.warmup import TwoPhaseWarmupMixin
from nndet.ptmodule.module import LightningBaseModule
from nndet.utils.load_weights_utils import resolve_stem_override
from nndet.utils.typing import CONVSEQ, LINEARSEQ


@MODULE_REGISTRY.register
class BoxDeformableDETRV002_ResEnc(
    LightningBaseModule,  # Main module
    BoxesPrepareMixin,  # prepare batch for box training
    BoxEvalMixin,  # Bounding Box Evaluation
    DeformableSetModelMixin_ResEnc,  # DETR Mixin to build the model
    BoxPredictionMixinV2,  # Bounding Box Sweep
):
    backbone_cls: Type[WrapperResEncAbstractBackbone] = ResidualEncoderUNetbackBoneWrapper #: define class for backbone
    backbone_conv_cls: Type[CONVSEQ] = ConvInstanceRelu  #: conv class used for backbone
    channel_mapper_cls: Type[ChannelMapper] = ChannelMapper  #: map channels from backbone to transformer
    channel_mapper_conv_cls: Type[CONVSEQ] = ConvGroupRelu  #: conv class used for channel mapper

    # transformer
    transformer_cls: Type[AbstractTransformer] = DeformableDETRTransformer  #: define detector transformer architecture
    pos_embed_cls: BasePositionEmbedding = PositionEmbeddingSine  #: define positional embedding for feature maps
    transformer_encoder_cls: BaseTransformerEncoder = (
        DeformableDETRTransformerEncoder  #: define encoder class of transformer
    )
    transformer_decoder_cls: BaseTransformerDecoder = (
        DeformableDETRTransformerDecoder  #: define decoder class of transformer
    )

    # head blocks
    head_cls: DETRHead = DeformableDETRHead  #: main DETR head
    head_linear_cls: LINEARSEQ = LayerLinearReluDrop  #: conv class used for head
    head_classifier_cls: FFNClassifier = FocalFFNClassifier  #: define classifier class
    head_regressor_cls: FFNRegressor = L1UGIoUFFNRegressor  #: define regressor class
    head_box_post_cls: DETRBoxPost = TopKBoxPost  #: define postprocessing strategy during inference

    matcher_cls: BaseMatcher = HungarianMatcher  #: matching algorithm
    matcher_class_criterion_cls: ClassCriterion = FocalClassCriterionSigmoid  #: criterion to compute class cost matrix
    # either reg or box criterion need to be set
    # reg criterion usually operates on encoded targets while box cirterion operates on raw boxes
    # there is no structural difference though and just a nomenclature
    matcher_reg_criterion_cls: Optional[BoxCriterion] = L1RegCriterion  #: criterion to compute regression cost matrix
    matcher_box_criterion_cls: Optional[
        BoxCriterion
    ] = GIoUCenterBoxCriterion  #: criterion to compute regression cost matrix)


@MODULE_REGISTRY.register
class BoxDeformableDETRV002_ResEnc_dyn(
    LightningBaseModule,  # Main module
    BoxesPrepareMixin,  # prepare batch for box training
    BoxEvalMixin,  # Bounding Box Evaluation
    DeformableSetModelMixin_ResEnc,  # DETR Mixin to build the model
    BoxPredictionMixinV2,  # Bounding Box Sweep
):
    backbone_cls: Type[WrapperResEncAbstractBackbone] = ResidualEncoderUNetbackBoneWrapper_dyn #: define class for backbone
    backbone_conv_cls: Type[CONVSEQ] = ConvInstanceRelu  #: conv class used for backbone
    channel_mapper_cls: Type[ChannelMapper] = ChannelMapper  #: map channels from backbone to transformer
    channel_mapper_conv_cls: Type[CONVSEQ] = ConvGroupRelu  #: conv class used for channel mapper

    # transformer
    transformer_cls: Type[AbstractTransformer] = DeformableDETRTransformer  #: define detector transformer architecture
    pos_embed_cls: BasePositionEmbedding = PositionEmbeddingSine  #: define positional embedding for feature maps
    transformer_encoder_cls: BaseTransformerEncoder = (
        DeformableDETRTransformerEncoder  #: define encoder class of transformer
    )
    transformer_decoder_cls: BaseTransformerDecoder = (
        DeformableDETRTransformerDecoder  #: define decoder class of transformer
    )

    # head blocks
    head_cls: DETRHead = DeformableDETRHead  #: main DETR head
    head_linear_cls: LINEARSEQ = LayerLinearReluDrop  #: conv class used for head
    head_classifier_cls: FFNClassifier = FocalFFNClassifier  #: define classifier class
    head_regressor_cls: FFNRegressor = L1UGIoUFFNRegressor  #: define regressor class
    head_box_post_cls: DETRBoxPost = TopKBoxPost  #: define postprocessing strategy during inference

    matcher_cls: BaseMatcher = HungarianMatcher  #: matching algorithm
    matcher_class_criterion_cls: ClassCriterion = FocalClassCriterionSigmoid  #: criterion to compute class cost matrix
    # either reg or box criterion need to be set
    # reg criterion usually operates on encoded targets while box cirterion operates on raw boxes
    # there is no structural difference though and just a nomenclature
    matcher_reg_criterion_cls: Optional[BoxCriterion] = L1RegCriterion  #: criterion to compute regression cost matrix
    matcher_box_criterion_cls: Optional[
        BoxCriterion
    ] = GIoUCenterBoxCriterion  #: criterion to compute regression cost matrix)


@MODULE_REGISTRY.register
class BoxDeformableDETRV002_ResEnc_TL(BoxDeformableDETRV002_ResEnc):

    def load_custom_state_dict(self, path):
        """
        Load custom state_dict

        Args:
            path: filepath to model checkpoint
        """
        logger.info("Loading custom state_dict")

        key_to_encoder = 'model.backbone.resenc.stages'
        key_to_stem = 'model.backbone.resenc.stem'
        downstream_input_channels = self.model.backbone.input_channels

        #take info from ckpt path (allows to overwrite plan specifications)
        ckp =  torch.load(path, weights_only=True)
        pre_train_statedict: dict[str, torch.Tensor] = ckp["network_weights"]  # Get pre-trained state dict

        pt_input_channels = ckp['nnssl_adaptation_plan']['pretrain_num_input_channels']
        pt_key_to_stem, pt_keys_to_in_proj = resolve_stem_override(self.model_cfg, ckp['nnssl_adaptation_plan'])
        pt_key_to_encoder =  ckp['nnssl_adaptation_plan']['key_to_encoder']

        def strip_dot_prefix(s) -> str:
            """Mini func to strip the dot prefix from the keys"""
            if s.startswith("."):
                return s[1:]
            return s

        encoder_weights = {k: v for k, v in pre_train_statedict.items() if k.startswith(pt_key_to_encoder)}
        stem_weights = {k: v for k, v in pre_train_statedict.items() if k.startswith(pt_key_to_stem)}

        if downstream_input_channels > pt_input_channels:
            k_proj = pt_keys_to_in_proj[0] + ".weight"  # Get the projection weights
            vals = (
                        stem_weights[k_proj].repeat(1, downstream_input_channels, 1, 1, 1)
                    ) / downstream_input_channels
            for k in pt_keys_to_in_proj:
                stem_weights[k + ".weight"] = vals
        new_encoder_weights = {
            strip_dot_prefix(k.replace(pt_key_to_encoder, "")): v for k, v in encoder_weights.items()
        }

        new_stem_weights = {strip_dot_prefix(k.replace(pt_key_to_stem, "")): v for k, v in stem_weights.items()}

        # ------------------------------- Load weights ------------------------------- #
        encoder_module = self.get_submodule(key_to_encoder)
        encoder_module.load_state_dict(new_encoder_weights)
        logger.info("Successfully loaded encoder state dict")
        stem_module = self.get_submodule(key_to_stem)
        stem_module.load_state_dict(new_stem_weights)
        logger.info("Successfully loaded stem state dict")
        del  new_stem_weights, stem_weights


@MODULE_REGISTRY.register
class BoxDeformableDETRV002_ResEnc_dyn_TL(BoxDeformableDETRV002_ResEnc):

    backbone_cls: Type[WrapperResEncAbstractBackbone] = ResidualEncoderUNetbackBoneWrapper_dyn

    def load_custom_state_dict(self, path: str):
        """
        Load custom state_dict with:
        - prefix remapping (pt -> downstream module-relative keys)
        - stem input-channel adaptation (your existing behavior)
        - encoder stage mismatch handling (skip too-deep pretrained stages)
        - conv kernel mismatch handling (supports reduction to 1 via mean)
        - only load keys that exist in target AND end up shape-compatible
        """
        logger.info("Loading custom state_dict (compatible + kernel-adapt + stage-adapt)")

        key_to_encoder = "model.backbone.resenc.stages"
        key_to_stem = "model.backbone.resenc.stem"
        downstream_input_channels = self.model.backbone.input_channels

        ckp = torch.load(path, weights_only=True)
        pre_train_statedict: Dict[str, torch.Tensor] = ckp["network_weights"]

        plan = ckp.get("nnssl_adaptation_plan", {})
        pt_input_channels = plan.get("pretrain_num_input_channels", None)
        pt_key_to_encoder = plan.get("key_to_encoder", None)
        stem_override = self.model_cfg.get("stem_override") or None
        if stem_override:
            pt_key_to_stem = stem_override
            pt_keys_to_in_proj = [f"{stem_override}.convs.0.conv", f"{stem_override}.convs.0.all_modules.0"]
        else:
            pt_key_to_stem = plan.get("key_to_stem", None)
            pt_keys_to_in_proj = plan.get("keys_to_in_proj", [])

        # Optional metadata (nice to have, not required)
        pt_n_stages = plan.get("pretrained_n_stages", None)
        tgt_n_stages = plan.get("target_n_stages", None)

        if pt_key_to_encoder is None or pt_key_to_stem is None:
            raise KeyError(
                "Checkpoint is missing nnssl_adaptation_plan keys 'key_to_encoder'/'key_to_stem'."
            )

        # -------------------------------------------------------------------------
        # Inner helper functions (live only inside this method)
        # -------------------------------------------------------------------------
        def strip_dot_prefix(s: str) -> str:
            return s[1:] if s.startswith(".") else s

        def remap_prefix(pt_prefix: str, sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            """Remove pt_prefix from keys to make them module-relative."""
            out: dict[str, torch.Tensor] = {}
            for k, v in sd.items():
                if not k.startswith(pt_prefix):
                    continue
                new_k = strip_dot_prefix(k.replace(pt_prefix, "", 1))
                out[new_k] = v
            return out

        def get_stage_and_block_info_from_key(key: str) -> Optional[Tuple[int, int, str]]:
            """
            Expected: "{stage_idx}.blocks.{block_idx}.{rest}"
            e.g. "0.blocks.0.conv1.conv.weight" -> (0, 0, "conv1.conv.weight")
            """
            m = re.match(r'^(\d+)\.blocks\.(\d+)\.(.+)$', key)
            if not m:
                return None
            return int(m.group(1)), int(m.group(2)), m.group(3)

        def adapt_conv_kernel_size(
            pretrained_weight: torch.Tensor,
            target_kernel_size: List[int],
            pretrained_kernel_size: List[int],
        ) -> torch.Tensor:
            """
            Adapt convolution weights when kernel sizes differ.
            Supports reducing a spatial dimension to 1 by averaging along that dim.
            """
            if len(target_kernel_size) != len(pretrained_kernel_size):
                raise AssertionError(
                    f"Kernel dims mismatch: {len(pretrained_kernel_size)} -> {len(target_kernel_size)}"
                )

            adapted = pretrained_weight
            for dim_idx, (tgt_k, pt_k) in enumerate(zip(target_kernel_size, pretrained_kernel_size)):
                spatial_dim = dim_idx + 2  # skip out_ch and in_ch
                if tgt_k == pt_k:
                    continue
                if tgt_k < pt_k:
                    if tgt_k == 1:
                        adapted = adapted.mean(dim=spatial_dim, keepdim=True)
                    else:
                        raise NotImplementedError(
                            f"Kernel reduction {pt_k}->{tgt_k} not supported (only ->1 supported)."
                        )
                else:
                    raise NotImplementedError(f"Kernel expansion {pt_k}->{tgt_k} not supported.")
            return adapted

        def adapt_encoder_weights_for_architecture(
            pretrained_encoder_weights: Dict[str, torch.Tensor],
            target_state_dict: Dict[str, torch.Tensor],
            pretrained_n_stages: Optional[int],
            target_n_stages: Optional[int],
        ) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
            """
            Returns (adapted_weights, adaptation_log).

            Rules:
            - skip if key not in target
            - if stage_idx >= min(pt_n_stages, tgt_n_stages) => skip (pretrained too deep)
            - if conv kernel mismatch: adapt (only reductions to 1 supported), then require shape match
            - else require exact shape match
            """
            adapted: Dict[str, torch.Tensor] = {}
            log: Dict[str, str] = {}

            if pretrained_n_stages is not None and target_n_stages is not None:
                n_stages_to_transfer = min(pretrained_n_stages, target_n_stages)
                if pretrained_n_stages > target_n_stages:
                    log["__stage_note__"] = (
                        f"pretrained deeper ({pretrained_n_stages} > {target_n_stages}); "
                        f"skipping stages >= {n_stages_to_transfer}"
                    )
                elif pretrained_n_stages < target_n_stages:
                    log["__stage_note__"] = (
                        f"target deeper ({target_n_stages} > {pretrained_n_stages}); "
                        f"deeper stages keep random init"
                    )
            else:
                n_stages_to_transfer = None

            for key, pt_w in pretrained_encoder_weights.items():
                if key not in target_state_dict:
                    log[key] = "skipped (not in target architecture)"
                    continue

                tgt_w = target_state_dict[key]

                parsed = get_stage_and_block_info_from_key(key)
                rest = parsed[2] if parsed is not None else ""

                # stage pruning
                if parsed is not None and n_stages_to_transfer is not None:
                    stage_idx = parsed[0]
                    if stage_idx >= n_stages_to_transfer:
                        log[key] = "skipped (pretrained too deep stage)"
                        continue

                # kernel adaptation (conv weights)
                is_conv_weight = (
                    ("conv.weight" in key) or ("conv.weight" in rest) or key.endswith(".conv.weight")
                )
                if is_conv_weight and pt_w.ndim >= 4 and tgt_w.ndim == pt_w.ndim:
                    pt_kernel = list(pt_w.shape[2:])
                    tgt_kernel = list(tgt_w.shape[2:])
                    if pt_kernel != tgt_kernel:
                        try:
                            pt_w2 = adapt_conv_kernel_size(pt_w, tgt_kernel, pt_kernel)
                        except NotImplementedError as e:
                            log[key] = f"skipped (kernel adapt unsupported: {e})"
                            continue

                        if pt_w2.shape != tgt_w.shape:
                            log[key] = f"skipped (post-adapt shape mismatch: {tuple(pt_w2.shape)} vs {tuple(tgt_w.shape)})"
                            continue

                        adapted[key] = pt_w2
                        log[key] = f"kernel adapted: {pt_kernel} -> {tgt_kernel}"
                        continue

                # default: strict shape match
                if pt_w.shape != tgt_w.shape:
                    log[key] = f"skipped (shape mismatch: {tuple(pt_w.shape)} vs {tuple(tgt_w.shape)})"
                    continue

                adapted[key] = pt_w

            return adapted, log

        def repeat_in_channels_like_template(w: torch.Tensor, new_in_ch: int) -> torch.Tensor:
            """
            Equivalent intent to template's repeat along in_channel dim.
            Works for 2D/3D conv weights.
            """
            reps = [1, new_in_ch] + [1] * (w.ndim - 2)
            return w.repeat(*reps) / new_in_ch

        has_stem_prefix_weights = any(k.startswith(pt_key_to_stem) for k in pre_train_statedict.keys())
        stem_in_encoder = not has_stem_prefix_weights

        logger.info(f"stem_in_encoder={stem_in_encoder} (has_stem_prefix_weights={has_stem_prefix_weights})")

        # ------------------------- target modules ------------------------- #
        encoder_module = self.get_submodule(key_to_encoder)
        stem_module = self.get_submodule(key_to_stem)

        enc_target_sd = encoder_module.state_dict()

        if stem_in_encoder:
            # -------- encoder weights from pt_key_to_encoder -------- #
            encoder_weights_raw = {k: v for k, v in pre_train_statedict.items() if k.startswith(pt_key_to_encoder)}

            # Handle input channel mismatch IN ENCODER (template does it here)
            if pt_input_channels is not None and downstream_input_channels > pt_input_channels and pt_keys_to_in_proj:
                k_proj_w = pt_keys_to_in_proj[0] + ".weight"
                if k_proj_w in encoder_weights_raw:
                    vals = repeat_in_channels_like_template(encoder_weights_raw[k_proj_w], downstream_input_channels)
                    for k in pt_keys_to_in_proj:
                        encoder_weights_raw[k + ".weight"] = vals
                    logger.info(f"[encoder] adapted input projection {pt_input_channels}->{downstream_input_channels}")
                else:
                    logger.warning(f"[encoder] wanted channel adapt but '{k_proj_w}' not found; skipping")

            # Strip prefix
            new_encoder_weights = remap_prefix(pt_key_to_encoder, encoder_weights_raw)

            # Adapt encoder for architecture
            adapted_encoder_weights, enc_log = adapt_encoder_weights_for_architecture(
                pretrained_encoder_weights=new_encoder_weights,
                target_state_dict=enc_target_sd,
                pretrained_n_stages=pt_n_stages,
                target_n_stages=tgt_n_stages,
            )
            logger.debug(f"[encoder] adaptation log: {enc_log}")

            # strict=True by merging into current target sd
            enc_target_sd.update(adapted_encoder_weights)
            encoder_module.load_state_dict(enc_target_sd, strict=True)

            logger.info(f"[encoder] loaded {len(adapted_encoder_weights)}/{len(enc_target_sd)} tensors (stem_in_encoder=True)")
            logger.info("[stem] skipped loading (stem_in_encoder=True, matches template)")

        else:
            # -------- separate stem and encoder -------- #
            stem_target_sd = stem_module.state_dict()
            encoder_weights_raw = {k: v for k, v in pre_train_statedict.items() if k.startswith(pt_key_to_encoder)}
            stem_weights_raw = {k: v for k, v in pre_train_statedict.items() if k.startswith(pt_key_to_stem)}

            logger.info(f"[ckpt] encoder_weights_raw: {len(encoder_weights_raw)}")
            logger.info(f"[ckpt] stem_weights_raw: {len(stem_weights_raw)}")
            logger.info(f"[target] encoder_module tensors: {len(enc_target_sd)}")
            logger.info(f"[target] stem_module tensors: {len(stem_target_sd)}")
            logger.info(f"stem_in_encoder={stem_in_encoder}")

            # Handle input channel mismatch IN STEM (template does it here)
            if pt_input_channels is not None and downstream_input_channels > pt_input_channels and pt_keys_to_in_proj:
                k_proj_w = pt_keys_to_in_proj[0] + ".weight"
                if k_proj_w in stem_weights_raw:
                    vals = repeat_in_channels_like_template(stem_weights_raw[k_proj_w], downstream_input_channels)
                    for k in pt_keys_to_in_proj:
                        stem_weights_raw[k + ".weight"] = vals
                    logger.info(f"[stem] adapted input projection {pt_input_channels}->{downstream_input_channels}")
                else:
                    logger.warning(f"[stem] wanted channel adapt but '{k_proj_w}' not found; skipping")

            # Strip prefixes
            new_encoder_weights = remap_prefix(pt_key_to_encoder, encoder_weights_raw)
            new_stem_weights = remap_prefix(pt_key_to_stem, stem_weights_raw)

            # Adapt encoder for architecture
            adapted_encoder_weights, enc_log = adapt_encoder_weights_for_architecture(
                pretrained_encoder_weights=new_encoder_weights,
                target_state_dict=enc_target_sd,
                pretrained_n_stages=pt_n_stages,
                target_n_stages=tgt_n_stages,
            )
            logger.debug(f"[encoder] adaptation log: {enc_log}")

            # Stem kernel adaptation
            for k in list(new_stem_weights.keys()):
                if k not in stem_target_sd:
                    continue
                pt_w = new_stem_weights[k]
                tgt_w = stem_target_sd[k]
                is_conv_w = ("conv.weight" in k) or (k.endswith(".weight") and "conv" in k)
                if is_conv_w and pt_w.ndim >= 4 and tgt_w.ndim == pt_w.ndim:
                    pt_k = list(pt_w.shape[2:])
                    tgt_k = list(tgt_w.shape[2:])
                    if pt_k != tgt_k:
                        logger.info(f"[stem] adapting kernel for {k}: {pt_k}->{tgt_k}")
                        new_stem_weights[k] = adapt_conv_kernel_size(pt_w, tgt_k, pt_k)

            # Filter stem by existence+shape (your current style)
            filtered_stem = {}
            for k, v in new_stem_weights.items():
                if k in stem_target_sd and tuple(v.shape) == tuple(stem_target_sd[k].shape):
                    filtered_stem[k] = v

            # strict=True loads via merge
            enc_target_sd.update(adapted_encoder_weights)
            encoder_module.load_state_dict(enc_target_sd, strict=True)

            stem_target_sd.update(filtered_stem)
            stem_module.load_state_dict(stem_target_sd, strict=True)
            logger.info("[stem] loaded keys:\n" + "\n".join(sorted(filtered_stem.keys())))

            logger.info(f"[encoder] loaded {len(adapted_encoder_weights)}/{len(enc_target_sd)} tensors")
            logger.info(f"[stem] loaded {len(filtered_stem)}/{len(stem_target_sd)} tensors")
            logger.info("Done loading.")


# ---------------------------------------------------------------------------
# Two-phase warmup variants. All of these differ only in which model parts
# train during the warmup phase vs. afterwards (TwoPhaseWarmupMixin, in
# nndet/ptmodule/mixins/warmup.py, holds the actual training_step/
# configure_optimizers logic, identical across every variant/backbone
# family). Each class still needs its own name: nndetection derives the
# experiment output folder name from the module class name, so collapsing
# these into one config-driven class would make different warmup schedules
# overwrite each other's output directory.
# ---------------------------------------------------------------------------

@MODULE_REGISTRY.register
class BoxDeformableDETRV002_ResEnc_TL_warmuptransformer_head(TwoPhaseWarmupMixin, BoxDeformableDETRV002_ResEnc_TL):
    phase1_parts_to_train = ["channel_mapper", "transformer", "head"]
    phase2_parts_to_train = ["backbone", "channel_mapper", "transformer", "head"]


@MODULE_REGISTRY.register
class DetSeg_DeformableDETR_ResEnc(BoxDeformableDETRV002_ResEnc):
    """Same as :class:`BoxDeformableDETRV002_ResEnc`; registered under a
    separate name for backwards-compatible experiment naming."""
