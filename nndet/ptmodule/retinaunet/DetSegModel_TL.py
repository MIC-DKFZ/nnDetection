# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger

from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.mixins.warmup import TwoPhaseWarmupMixin
from nndet.ptmodule.retinaunet.run_v002 import RetinaUNetFocalV002


def _repeat_stem_conv_for_channels(weight: torch.Tensor, downstream_channels: int, pretrain_channels: int) -> torch.Tensor:
    """
    Repeats (and averages) a pretrained stem conv's input-channel dim when
    the downstream task has more input channels than the checkpoint's
    pretraining task did (e.g. a single-channel pretrained stem reused for
    a 3-channel T1/FA/MD downstream task).
    """
    if downstream_channels <= pretrain_channels:
        return weight
    return weight.repeat(1, downstream_channels, 1, 1, 1) / downstream_channels


@MODULE_REGISTRY.register
class DetSegModel(RetinaUNetFocalV002):
    """Same as :class:`RetinaUNetFocalV002`; registered under a separate
    name for backwards-compatible experiment naming."""


@MODULE_REGISTRY.register
class DetSegModel_RetinaUNet(DetSegModel):
    """Same as :class:`DetSegModel`; registered under a separate name for
    backwards-compatible experiment naming."""


@MODULE_REGISTRY.register
class DetSegModel_TL(DetSegModel):

    def load_custom_state_dict(self, path):
        """"
        Load custom state_dict

        Args:
            path: filepath to model checkpoint
        """

        new_state_dict = {}
        for key,value in torch.load(path)["network_weights"].items():
            new_key = "model." + key
            if "decoder" in new_key.split("."):
                list_new_key = new_key.split(".")
                list_new_key[1]="neck"
                new_key = ".".join(list_new_key)

            new_state_dict[new_key] = value
        return self.load_state_dict(new_state_dict, strict=False)


@MODULE_REGISTRY.register
class DetSegModel_TL_MultiTalentStem(DetSegModel_TL):
    """
    Loads a MultiTalent-style ConvBackbone checkpoint whose first level's
    input-channel-mapping conv was trained as a separate per-dataset stem
    module instead of living inside backbone.levels.0 (nndetection's own
    ConvBackbone has no standalone stem -- levels.0 maps input channels
    directly). Encoder+stem only -- the neck/decoder is left randomly
    initialized (matches the checkpoint's actually-validated finetuning
    recipe, not the checkpoint's own trained neck).

    key_to_stem and pretrain_num_input_channels are read from the
    checkpoint's own nnssl_adaptation_plan by default -- unified with how
    the ResEnc TL loaders work, nothing about which stem to use is
    hardcoded here. To run CT-stem and MRI-stem finetuning from the same
    checkpoint, set `model_cfg.stem_override` (e.g.
    `-o model_cfg.stem_override=stem.004`) to pick a different stem than
    the checkpoint's plan defaults to, and give each run its own
    `-o exp.tag=...` so they don't collide into the same experiment output
    folder -- see docs/finetuning.md §3.4.1. No checkpoint copying or
    editing needed.
    """

    def load_custom_state_dict(self, path):
        """
        Load custom state_dict

        Args:
            path: filepath to model checkpoint
        """
        ckpt = torch.load(path)
        raw_state_dict = ckpt["network_weights"]
        adaptation_plan = ckpt["nnssl_adaptation_plan"]
        key_to_stem = self.model_cfg.get("stem_override") or adaptation_plan["key_to_stem"]
        pretrain_input_channels = adaptation_plan["pretrain_num_input_channels"]
        downstream_input_channels = self.model.backbone.in_channels

        stem_prefix = f"{key_to_stem}."
        old_level0_prefix = "backbone.levels.0.component.0."
        # the stem's own nn.Sequential(conv, norm) uses numeric indices,
        # unlike backbone.levels' named "conv"/"norm" submodules
        stem_index_to_name = {"0": "conv", "1": "norm"}

        new_state_dict = {}
        for key, value in raw_state_dict.items():
            if key.startswith(stem_prefix):
                # fuse the checkpoint's chosen stem into level 0's first
                # conv block -- nndetection has no standalone stem
                index, _, rest = key[len(stem_prefix):].partition(".")
                name = stem_index_to_name[index]
                if name == "conv" and rest == "weight":
                    value = _repeat_stem_conv_for_channels(value, downstream_input_channels, pretrain_input_channels)
                new_state_dict[f"model.backbone.levels.0.component.0.{name}.{rest}"] = value
            elif key.startswith(old_level0_prefix):
                # the checkpoint's own level-0 block becomes the SECOND
                # conv block, since the stem now supplies the first one
                suffix = key[len(old_level0_prefix):]
                new_state_dict[f"model.backbone.levels.0.component.1.{suffix}"] = value
            elif "backbone" in key:
                new_state_dict["model." + key] = value
            # everything else (decoder, seg_layer, other datasets' unused
            # stems) is intentionally dropped -- encoder+stem only

        orig_keys = {key for key in self.state_dict() if "backbone" in key or "stem" in key}
        new_keys = set(new_state_dict.keys())
        assert orig_keys == new_keys, (
            f"State dict mismatch!\nMissing keys: {orig_keys - new_keys}\nExtra keys: {new_keys - orig_keys}"
        )
        logger.info(f"Keys loaded from checkpoint: {new_keys}")
        return self.load_state_dict(new_state_dict, strict=False)


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
class DetSegModel_TL_MultiTalentStem_warmupdecoder_heads(TwoPhaseWarmupMixin, DetSegModel_TL_MultiTalentStem):
    phase1_parts_to_train = ["neck", "head"]
    phase2_parts_to_train = ["backbone", "neck", "head"]


@MODULE_REGISTRY.register
class DetSegModel_TL_MultiTalentStem_warmupdecoder_heads_1e3(DetSegModel_TL_MultiTalentStem_warmupdecoder_heads):
    """Same as :class:`DetSegModel_TL_MultiTalentStem_warmupdecoder_heads`;
    registered under a separate name so a Hydra config can select a
    different (1e-3) trainer_cfg."""


@MODULE_REGISTRY.register
class DetSegModel_TL_MultiTalentStem_warmupnet_1e3(TwoPhaseWarmupMixin, DetSegModel_TL_MultiTalentStem):
    phase1_parts_to_train = ["backbone", "neck", "head"]
    phase2_parts_to_train = ["backbone", "neck", "head"]


@MODULE_REGISTRY.register
class DetSegModel_TL_warmupdecoder_heads(TwoPhaseWarmupMixin, DetSegModel_TL):
    phase1_parts_to_train = ["neck", "head"]
    phase2_parts_to_train = ["backbone", "neck", "head"]


@MODULE_REGISTRY.register
class DetSegModel_TL_warmupdecoder_heads_1e3(DetSegModel_TL_warmupdecoder_heads):
    """Same as :class:`DetSegModel_TL_warmupdecoder_heads`; registered under a
    separate name so a Hydra config can select a different (1e-3) trainer_cfg."""


@MODULE_REGISTRY.register
class DetSegModel_TL_warmupnet_1e3(TwoPhaseWarmupMixin, DetSegModel_TL):
    phase1_parts_to_train = ["backbone", "neck", "head"]
    phase2_parts_to_train = ["backbone", "neck", "head"]
