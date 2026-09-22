from typing import Optional, Type

import torch

from nndet.core.boxes.criterions.base import BoxCriterion, ClassCriterion
from nndet.core.boxes.criterions.box import GIoUCenterBoxCriterion, L1RegCriterion
from nndet.core.boxes.criterions.cls import FocalClassCriterionSigmoid
from nndet.core.boxes.matcher1to1.base import BaseMatcher
from nndet.core.boxes.matcher1to1.hungarian import HungarianMatcher
from nndet.core.post.detr import DETRBoxPost, TopKBoxPost
from nndet.nn.backbone.abstract_primus import WrapperPrimusAbstractBackbone
from nndet.nn.backbone.blueprints.primus import PrimusbackBoneWrapper
from nndet.nn.heads.classifier.ffn import FFNClassifier, FocalFFNClassifier
from nndet.nn.heads.detr.base import DETRHead
from nndet.nn.heads.detr.deformable_detr import DeformableDETRHead
from nndet.nn.heads.regressor.ffn import FFNRegressor, L1UGIoUFFNRegressor
from nndet.nn.layers.conv import ConvInstanceRelu
from nndet.nn.layers.conv.group import ConvGroupRelu
from nndet.nn.layers.linear import LayerLinearReluDrop
from nndet.nn.layers.pos_embed.base import BasePositionEmbedding
from nndet.nn.layers.pos_embed.sine import PositionEmbeddingSine
from nndet.nn.neck.channel_mapper_resolution import ChannelMapper_resolution
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
from nndet.ptmodule.module import LightningBaseModule
from nndet.utils.load_weights_utils import filter_state_dict, handle_pos_embed_resize
from nndet.utils.typing import CONVSEQ, LINEARSEQ

@MODULE_REGISTRY.register
class BoxDeformableDETRV002_Primus(
    LightningBaseModule,  # Main module
    BoxesPrepareMixin,  # prepare batch for box training
    BoxEvalMixin,  # Bounding Box Evaluation
    DeformableSetModelMixin_ResEnc,  # DETR Mixin to build the model
    BoxPredictionMixinV2,  # Bounding Box Sweep
):
    backbone_cls: Type[WrapperPrimusAbstractBackbone] = PrimusbackBoneWrapper   #: define class for backbone
    backbone_conv_cls: Type[CONVSEQ] = ConvInstanceRelu  #: conv class used for backbone
    channel_mapper_cls: Type[ChannelMapper_resolution] = ChannelMapper_resolution  #: map channels from backbone to transformer
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
class BoxDeformableDETRV002_Primus_TL(BoxDeformableDETRV002_Primus):

    def load_custom_state_dict(self, path):
        """
        Load custom state_dict

        Args:
            path: filepath to model checkpoint
        """
        print("Loading custom state_dict")
        key_to_encoder = 'model.backbone.eva'
        key_to_stem = 'model.backbone.down_projection'
        key_to_lpe = 'model.backbone.eva.pos_embed'
        downstream_input_channels = self.model.backbone.input_channels

        #take info from ckpt path (allows to overwrite plan specifications)
        ckp =  torch.load(path, weights_only=True)
        pre_train_statedict: dict[str, torch.Tensor] = ckp["network_weights"]  # Get pre-trained state dict

        pt_input_channels = ckp['nnssl_adaptation_plan']['pretrain_num_input_channels']
        pt_key_to_stem =  ckp['nnssl_adaptation_plan']['key_to_stem']
        pt_key_to_encoder =  ckp['nnssl_adaptation_plan']['key_to_encoder']
        pt_keys_to_in_proj =  ckp['nnssl_adaptation_plan']['keys_to_in_proj']
        pt_key_to_lpe =  ckp['nnssl_adaptation_plan']['key_to_lpe']


        ####we need the pretraining model input patch size. Allows overwrites (e.g for voco needed)
        config_pt =next(iter(
            ckp["nnssl_adaptation_plan"]["pretrain_plan"]["configurations"]
        ))
        pt_input_patchsize = ckp["nnssl_adaptation_plan"]["pretrain_plan"]['configurations'][config_pt]['patch_size']

        stem_in_encoder = pt_key_to_stem in pre_train_statedict
        pt_weight_in_ch_mismatch = False
        need_to_adapt_lpe = False  # I.e. Learnable positional embedding
        lpe_in_stem = False

        downstream_input_patchsize = self.model.backbone.input_shape

        if key_to_lpe is not None:
            lpe_in_encoder = key_to_lpe.startswith(key_to_encoder)
            lpe_in_stem = key_to_lpe.startswith(key_to_stem)
            if pt_input_patchsize != downstream_input_patchsize:
                need_to_adapt_lpe = True# LPE shape won't fit -> resize it

        def strip_dot_prefix(s) -> str:
            """Mini func to strip the dot prefix from the keys"""
            if s.startswith("."):
                return s[1:]
            return s

        # ----- Match the keys of pretrained weights to the current architecture ----- #
        if stem_in_encoder:
            encoder_weights = {k: v for k, v in pre_train_statedict.items() if k.startswith(pt_key_to_encoder)}
            if downstream_input_channels > pt_input_channels:
                pt_weight_in_ch_mismatch = True
                k_proj = pt_keys_to_in_proj[0] + ".weight"  # Get the projection weights
                vals = (encoder_weights[k_proj].repeat(1, downstream_input_channels, 1, 1)) / downstream_input_channels
                for k in pt_keys_to_in_proj:
                    encoder_weights[k] = vals
            # Fix the path to the weights:
            new_encoder_weights = {
                strip_dot_prefix(k.replace(pt_key_to_encoder, "")): v for k, v in encoder_weights.items()
            }
            # --------------------------------- Adapt LPE -------------------------------- #
            if need_to_adapt_lpe:
                if lpe_in_encoder:
                    handle_pos_embed_resize(new_encoder_weights,
                                            self.get_submodule(key_to_encoder).state_dict(),
                                            'interpolate_trilinear',
                                            downstream_input_patchsize,
                                            pt_input_patchsize,
                                            new_encoder_weights['down_projection.proj.weight'].shape[2:])
                    new_encoder_weights["pos_embed"].to(next(self.parameters()).device)
                if "cls_token" in encoder_weights.keys():
                    skip_strings_in_pretrained = ["cls_token"]
                    new_encoder_weights, found_cls_token = filter_state_dict(encoder_weights, skip_strings_in_pretrained)

            # ------------------------------- Load weights ------------------------------- #
            encoder_module = self.get_submodule(key_to_encoder)
            encoder_module.load_state_dict(new_encoder_weights)
        else:
            encoder_weights = {k: v for k, v in pre_train_statedict.items() if k.startswith(pt_key_to_encoder)}
            stem_weights = {k: v for k, v in pre_train_statedict.items() if k.startswith(pt_key_to_stem)}
            if downstream_input_channels > pt_input_channels:
                pt_weight_in_ch_mismatch = True
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
            # --------------------------------- Adapt LPE -------------------------------- #
            if need_to_adapt_lpe:
                if lpe_in_stem:  # Since stem not in encoder we need to take care of lpe in it here
                    handle_pos_embed_resize(new_stem_weights,
                                            self.get_submodule(key_to_stem).state_dict(),
                                            'interpolate_trilinear',
                                            downstream_input_patchsize,
                                            pt_input_patchsize,
                                            new_stem_weights['proj.weight'].shape[2:])
                    new_stem_weights["pos_embed"].to(next(self.parameters()).device)
                elif lpe_in_encoder:
                    handle_pos_embed_resize(new_encoder_weights,
                                            self.get_submodule(key_to_encoder).state_dict(),
                                            'interpolate_trilinear',
                                            downstream_input_patchsize,
                                            pt_input_patchsize,
                                            new_stem_weights['proj.weight'].shape[2:])
                    new_encoder_weights["pos_embed"].to(next(self.parameters()).device)
                else:
                    pass
            if "cls_token" in new_encoder_weights.keys():
                skip_strings_in_pretrained = ["cls_token"]
                new_encoder_weights, found_cls_token = filter_state_dict(new_encoder_weights, skip_strings_in_pretrained)

            # ------------------------------- Load weights ------------------------------- #
            encoder_module = self.get_submodule(key_to_encoder)
            encoder_module.load_state_dict(new_encoder_weights)
            stem_module = self.get_submodule(key_to_stem)
            stem_module.load_state_dict(new_stem_weights)
            del  new_stem_weights, stem_weights
