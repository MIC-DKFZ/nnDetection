# Finetuning pretrained backbones

This document covers the pretrained-backbone / finetuning support added on top
of the standard nnDetection training flow: using an nnU-Net **ResEnc**
(Residual Encoder) or a self-supervised **Primus** (EVA-style) vision
transformer as the detector's backbone, optionally initialized from a
supervised or self-supervised pretraining checkpoint.

Everything here is additive to the normal nnDetection workflow described in
the main [README](../README.md) -- preprocessing (`nndet_prep`), training
(`nndet_train`), and evaluation are unchanged; this document only covers the
parts that are new. If you're new to nnDetection itself, read that README
first -- this document assumes you already know how to preprocess a dataset
and run a normal (non-finetuning) training.

**A few terms used throughout, if you're coming from outside this project:**

- **nnssl** -- a separate self-supervised pretraining pipeline (a sibling
  project to nnU-Net/nnDetection) used to produce most of the checkpoints
  referenced here.
- **MultiTalent** -- a joint multi-dataset training approach: one shared
  backbone trained across many datasets/modalities at once, with a small
  separate "stem" module per dataset handling that dataset's own input. A
  "MultiTalent" checkpoint therefore contains *several* alternative stems
  (§3.4) rather than a single fixed input layer.
- **Adaptation plan** -- a metadata block bundled inside every compatible
  checkpoint (`checkpoint["nnssl_adaptation_plan"]`) describing how to load
  it (§3.1). You never write this yourself; the loader code reads it.

## Quick start

Using one of the provided checkpoints below with a ResEnc backbone. Preprocess
your dataset as usual first (standard nnDetection, unchanged):

```bash
nndet_prep Task<XXX>_YourDataset
```

Then train, initializing the backbone from a pretrained checkpoint --
**RetinaUNet** head:

```bash
nndet_train Task<XXX>_YourDataset residual_encoder_retinaunet_focal_v002 0 \
    -o module=RetinaUNetFocalV002_ResEnc_TL exp.tag=_<checkpoint_name> \
    +transfer_learning_ckpt=/path/to/checkpoint_final.pth \
    --transfer_learning --load_adapt_plan
```

or **Deformable DETR** head instead (same checkpoints work with either head):

```bash
nndet_train Task<XXX>_YourDataset residual_encoder_def_detr_v002 0 \
    -o module=BoxDeformableDETRV002_ResEnc_TL exp.tag=_<checkpoint_name> \
    +transfer_learning_ckpt=/path/to/checkpoint_final.pth \
    --transfer_learning --load_adapt_plan
```

(`exp.tag` picks the output folder name -- e.g. `_MAE`, `_VoCo`. The folder
name is `${module}_${plan}${exp.tag}` and does *not* encode which checkpoint
you passed, so always set it to something identifying the checkpoint,
otherwise a second run silently overwrites the first.)

Replace `Task<XXX>_YourDataset` with your own preprocessed task, and the
checkpoint path with one from the table below. `--load_adapt_plan` reconciles
the model's architecture to match the checkpoint; `--transfer_learning` loads
the weights. Note that each top-level config has a default `planner` (e.g.
`D3V002`) and `nndet_train` looks for a plan preprocessed under that exact
name -- if you preprocessed under a different one, add `-o planner=<name>`.

## Provided checkpoints

| Checkpoint | Pretraining | Backbone | Recommended config |
|---|---|---|---|
| [`ResEncL-MissingPiece-MAE`](https://huggingface.co/MIC-DKFZ/ResEncL-MissingPiece-MAE) | Masked Autoencoder (MAE) | ResEnc | `residual_encoder_retinaunet_focal_v002` or `residual_encoder_def_detr_v002` + matching `_TL` module (Quick start, or §3.3) |
| [`ResEncL-MissingPiece-MG`](https://huggingface.co/MIC-DKFZ/ResEncL-MissingPiece-MG) | Models Genesis (MG) | ResEnc | same as above |
| [`ResEncL-MissingPiece-S3D`](https://huggingface.co/MIC-DKFZ/ResEncL-MissingPiece-S3D) | Spark 3D (S3D) | ResEnc | same as above |
| [`ResEncL-MissingPiece-VoCo`](https://huggingface.co/MIC-DKFZ/ResEncL-MissingPiece-VoCo) | Volume Contrastive (VoCo) | ResEnc | same as above -- pre-training patch size `[192, 192, 64]` |
| [`ResEncL-MissingPiece-MultiTalent`](https://huggingface.co/MIC-DKFZ/ResEncL-MissingPiece-MultiTalent) | MultiTalent (supervised) | ResEnc | same as above; see §3.4 to pick a different stem |
| [`RetinaUNet-MissingPiece-MultiTalent`](https://huggingface.co/MIC-DKFZ/RetinaUNet-MissingPiece-MultiTalent) | MultiTalent (supervised) | nndetection's native `ConvBackbone` | `retinaunet_focal_v002_for_ConvBackboneMultiTalent` (§3.3); see §3.4 to pick a different stem |
| [`nnFoundationCNN`](https://huggingface.co/MIC-DKFZ/nnFoundationCNN) | nnFoundation | ResEnc | `residual_encoder_retinaunet_focal_v002` or `residual_encoder_def_detr_v002` + matching `_TL` module (Quick start, or §3.3) |
| [`nnFoundationViT`](https://huggingface.co/MIC-DKFZ/nnFoundationViT) | nnFoundation | Primus (ViT) | `Primus_def_detr_v002` + `BoxDeformableDETRV002_Primus_TL` (§3.3) |

Each checkpoint name links to its Hugging Face repository. MAE, MG, S3D, VoCo
and the two `nnFoundation` ones are self-supervised; the two MultiTalent ones
are supervised joint segmentation pretraining.
`RetinaUNet-MissingPiece-MultiTalent` contains the encoder and FPN decoder of
a Retina U-Net, without the detection heads.

All pretrained on single-channel (grayscale) 3D volumes; the MultiTalent ones
across ~87 datasets of mixed modalities, hence the multiple stems. The same
`-o module=...` / `+transfer_learning_ckpt=...` / `--transfer_learning
--load_adapt_plan` pattern from Quick start works for every row -- only the
top-level config and module name differ (§3.3 has the exact command per row).

MAE, MG, S3D and VoCo all pretrain the exact
same ResEncL encoder+stem architecture (byte-identical key names/shapes) --
they only differ in the pretraining objective's own head (a reconstruction
decoder for the MAE-style ones, a contrastive projector for VoCo), which
isn't loaded downstream either way, so all four load through the same `_TL`
classes identically.

**Where to get these checkpoint files:** each is a separate Hugging Face model
repository (linked from the table above) holding the weights
(`checkpoint_final.pth`), a standalone `adaptation_plan.json`, and a model card
with the papers to cite. The six Missing Piece checkpoints are grouped in the
[The Missing Piece: Pre-trained nnDetection Backbones](https://huggingface.co/collections/MIC-DKFZ/the-missing-piece-pre-trained-nndetection-backbones-6ab62e4d50b7219ef37d1d50) collection; the two
nnFoundation ones are released with their own publication
([arXiv:2609.26924](https://arxiv.org/abs/2609.26924)) in the
[nnFoundation](https://huggingface.co/collections/MIC-DKFZ/nnfoundation-6ab4e3a5a7a8152d83ed86bf) collection.

Download a single checkpoint with:

```bash
pip install huggingface_hub
hf download <repo-id> checkpoint_final.pth --local-dir ./checkpoints/<repo-name>
# e.g.
hf download MIC-DKFZ/ResEncL-MissingPiece-MAE checkpoint_final.pth --local-dir ./checkpoints/ResEncL-MissingPiece-MAE
```

Each checkpoint also carries the papers to cite (§3.1); they are printed to the
training log when the weights are loaded.

## 1. Supported combinations

| Backbone | Detection head | Base module class | Checkpoint-loading variant |
|---|---|---|---|
| ResEnc (fixed architecture) | RetinaUNet | `RetinaUNetFocalV002_ResEnc` | `RetinaUNetFocalV002_ResEnc_TL` |
| ResEnc (**dynamic** architecture) | RetinaUNet | `RetinaUNetFocalV002_ResEnc_dyn` | `RetinaUNetFocalV002_ResEnc_dyn_TL` |
| ResEnc (fixed architecture) | Deformable DETR | `BoxDeformableDETRV002_ResEnc` | `BoxDeformableDETRV002_ResEnc_TL` |
| ResEnc (**dynamic** architecture) | Deformable DETR | `BoxDeformableDETRV002_ResEnc_dyn` | `BoxDeformableDETRV002_ResEnc_dyn_TL` |
| Primus (ViT) | Deformable DETR | `BoxDeformableDETRV002_Primus` | `BoxDeformableDETRV002_Primus_TL` |
| ConvBackbone (fixed architecture, MultiTalent-style stem) | RetinaUNet | `DetSegModel` | `DetSegModel_TL_MultiTalentStem` |

"Fixed architecture" means the backbone's shape comes from the model config
you write. "Dynamic architecture" (`_dyn` classes) means the shape instead
comes from nnDetection's own planned architecture -- use this when you want
nnDetection's normal architecture search/dataset adaptation while still
loading as much as possible from a checkpoint, even if its depth/shape
doesn't exactly match. Only ResEnc has a dynamic variant.

Fixed-architecture configs train with a fixed patch size of `[128, 128, 128]`
(set in `model_cfg.backbone_kwargs`); dynamic configs take the patch size,
kernel sizes and strides from the nnDetection plan.

The last row (`ConvBackbone`) is nndetection's own native architecture, not
ResEnc -- it loads MultiTalent checkpoints whose input stem was trained as a
separate per-dataset module.

Only the `_TL` classes actually load pretrained weights (§3). The non-`_TL`
classes (e.g. `BoxDeformableDETRV002_ResEnc`) build the same architecture
with random initialization -- useful for training from scratch, or as the
class `--build_from_pretrained_arch` builds against (§3.2).

## 2. Picking a config

Start from one of the top-level train configs in `nndet/conf/train/`:

- `residual_encoder_retinaunet_focal_v002.yaml` -- RetinaUNet + fixed ResEnc
- `residual_encoder_retinaunet_focal_v002_dyn.yaml` -- RetinaUNet + **dynamic** ResEnc
- `residual_encoder_def_detr_v002.yaml` -- Deformable DETR + fixed ResEnc
- `residual_encoder_def_detr_v002_dyn.yaml` -- Deformable DETR + **dynamic** ResEnc
- `Primus_def_detr_v002.yaml` -- Deformable DETR + Primus

Each selects a `model_cfg` (in `nndet/conf/train/model_cfg/`) that defines the
backbone's architecture -- `retinaunet_l1_focal_atss_ms_ema_for_ResEnc.yaml`,
its `_dyn` counterpart, `deformable_detr_sigm_for_ResEnc.yaml`, its `_dyn`
counterpart, and `deformable_detr_sigm_for_primus.yaml`.

Train as usual, then override the module directly for a specific variant:

```bash
nndet_train Task007_Pancreas residual_encoder_def_detr_v002 0
```

```bash
nndet_train Task007_Pancreas residual_encoder_retinaunet_focal_v002 0 \
    -o module=RetinaUNetFocalV002_ResEnc_TL \
    --transfer_learning
```

### 2.1 `backbone_kwargs`: what to fill in

For the fixed-architecture configs, `model_cfg.backbone_kwargs` fully
specifies the ResEnc/Primus backbone (`n_stages`, `features_per_stage`,
`kernel_sizes`, `strides`, `conv_op`, `norm_op`, ..., plus `input_shape`/
`patch_size`/`batch_size`). Copy an existing config's block and adjust for
your case -- nothing dataset-specific beyond patch size/channels.

For the dynamic config (`..._dyn.yaml`), `backbone_kwargs` intentionally
leaves out `n_stages`/`features_per_stage`/etc. -- those come from the plan.

If you're using `--load_adapt_plan` (§3.2), everything in `backbone_kwargs`
except `input_shape`/`patch_size` gets overwritten from the checkpoint at
train time anyway, so the exact starting values mostly don't matter -- just
make sure the keys exist so the config validates.

### 2.2 Default settings, and what to actually use for finetuning

What you get from each top-level config's own default `trainer_cfg`, unmodified:

| Config | Default `trainer_cfg` | Optimizer | Epochs | Built-in LR ramp-up? |
|---|---|---|---|---|
| `residual_encoder_retinaunet_focal_v002[_dyn]` | `sgd_base` | SGD | 50 | Yes -- `warm_iterations: 4000` |
| `residual_encoder_def_detr_v002[_dyn]` | `adamw_100ep_high_lr_wd` | AdamW | 100 | **No** -- `warm_iterations: 0` |
| `Primus_def_detr_v002` | `adamw_100ep_low_lr_wd_warm` | AdamW | 100 | Yes -- `warm_iterations: 10000` |

DETR's default is the odd one out -- it has no LR ramp-up at all. This
matters for finetuning specifically: jumping straight to a high LR on a
pretrained backbone tends to be less stable than easing into it.

**Recommendation, based on real usage:** for finetuning a checkpoint,
use the plain `_TL` module class (not a `_warmup*` class, §4) with a
`trainer_cfg` that has a nonzero `warm_iterations` LR ramp-up. RetinaUNet's
and Primus's defaults already have this; for DETR, override it explicitly:

```bash
nndet_train Task<XXX>_YourDataset residual_encoder_def_detr_v002 0 \
    -o module=BoxDeformableDETRV002_ResEnc_TL exp.tag=_<checkpoint_name> \
       train/trainer_cfg@trainer_cfg=adamw_100ep_high_lr_wd_warm \
    +transfer_learning_ckpt=/path/to/checkpoint_final.pth \
    --transfer_learning --load_adapt_plan
```

This is what real finetuning experiments (the "rocket" project's own
DETR+ResEnc runs) mostly used in practice -- plain `_TL` class, `_warm`
trainer_cfg for the LR ramp-up, no two-phase freeze/unfreeze. The §4
two-phase mechanism (freezing most of the model, then unfreezing) is a
heavier, less-tested-in-practice alternative -- verified to work end-to-end
(this document's own testing), but only found in one real ablation-style
experiment rather than routine use. Reach for it if you specifically want
to compare against the simpler LR-ramp-up approach, not as your first try.

## 3. Loading pretrained weights

### 3.1 Checkpoint format

Checkpoints are expected to contain:

```python
{
    "network_weights": {...},          # raw encoder/stem state dict
    "nnssl_adaptation_plan": {
        "pretrain_num_input_channels": <int>,
        "key_to_stem": "<prefix into network_weights>",
        "key_to_encoder": "<prefix into network_weights>",
        "keys_to_in_proj": [<module path prefixes>],
        "key_to_lpe": "<prefix, Primus only>",           # learned positional embedding
        "pretrained_n_stages": <int, optional>,          # ResEnc dyn only
        "target_n_stages": <int, optional>,              # ResEnc dyn only
        "architecture_plans": {
            "arch_class_name": "<str>",
            "arch_kwargs": {...},       # same keys as backbone_kwargs
        },
        "pretrain_plan": {"configurations": {"<name>": {"patch_size": [...]}}},
    },
    "citations": [...],                # optional, see below
}
```

Set the checkpoint path via a **top-level** CLI override --
`+transfer_learning_ckpt=/path/to/checkpoint_final.pth` (`+` adds a brand-new
key, `-o` only overrides one the config already declares).

#### Citations

A checkpoint can carry the papers users should cite when finetuning from it.
The format matches nnssl / nnU-Net's `PretrainedTrainer`:

```python
"citations": [
    {
        "type": "Pretraining Method",       # Architecture | Pretraining Method |
                                            # Pre-Training Dataset | Framework | ...
        "name": "Masked Auto Encoder",
        "apa_citations": ["<full APA reference string>", ...],
    },
]
```

When `--transfer_learning` loads a checkpoint, these are printed to the training
log, grouped by `type`. Checkpoints without the field are silently accepted.

The field is read from **either** the checkpoint's top level (`ckpt["citations"]`,
where nnssl writes it and where nnU-Net's `nnUNetv2_preprocess_like_nnssl` reads
it from) **or** from inside `nnssl_adaptation_plan` (where the published
`adaptation_plan.json` model cards carry it). The checkpoints in "Provided
checkpoints" above set both, so they work with nnU-Net's pretraining tooling
unchanged.

### 3.2 CLI flags

| Flag | Effect |
|---|---|
| `-tl` / `--transfer_learning` | Load pretrained weights via the module's `load_custom_state_dict(path)`. Requires a `_TL` module class. |
| `--load_adapt_plan` | Before building the model, overwrite `model_cfg.backbone_kwargs` (and the plan's `architecture` section) from the checkpoint's own `nnssl_adaptation_plan.architecture_plans` -- forces the model's architecture to match the checkpoint. |
| `--build_from_pretrained_arch` | Same architecture reconciliation as `--load_adapt_plan`, but doesn't load weights. Mutually exclusive with `--transfer_learning`. |
| `--val_best` | Sweep/evaluate using the best checkpoint instead of the last one. |

Two workflows:

**A. Force the architecture to match the checkpoint exactly** (fixed
ResEnc/Primus configs) -- `--transfer_learning --load_adapt_plan` together.

**B. Keep nnDetection's own planned architecture, adapt whatever matches**
(the `_dyn` variant) -- `--transfer_learning` only, **without**
`--load_adapt_plan` (that would defeat the point). The `_dyn_TL` module's
`load_custom_state_dict` handles stage-count/kernel-size mismatches itself.

```bash
# A: force checkpoint's architecture
nndet_train Task007_Pancreas residual_encoder_def_detr_v002 0 \
    --transfer_learning --load_adapt_plan

# B: keep nnDetection's planned architecture
nndet_train Task007_Pancreas residual_encoder_def_detr_v002_dyn 0 \
    --transfer_learning
```

### 3.3 Example commands

The RetinaUNet/DETR fixed-ResEnc commands are in Quick start above -- they
work for any checkpoint in the "Provided checkpoints" table, ResEnc or
MultiTalent-on-ResEnc alike, since `RetinaUNetFocalV002_ResEnc_TL` /
`BoxDeformableDETRV002_ResEnc_TL` read `key_to_encoder`/`key_to_stem` from
the checkpoint's own plan regardless of how deeply nested those keys are.

**Dynamic ResEnc** (workflow B) -- module already defaults correctly, no
`-o module=...` needed:

```bash
nndet_train Task<XXX>_YourDataset residual_encoder_retinaunet_focal_v002_dyn 0 \
    -o exp.tag=_<checkpoint_name> \
    +transfer_learning_ckpt=/path/to/checkpoint_final.pth \
    --transfer_learning
```

```bash
nndet_train Task<XXX>_YourDataset residual_encoder_def_detr_v002_dyn 0 \
    -o exp.tag=_<checkpoint_name> \
    +transfer_learning_ckpt=/path/to/checkpoint_final.pth \
    --transfer_learning
```

**`RetinaUNet-MissingPiece-MultiTalent`** (native `ConvBackbone`, not ResEnc; module
already defaults correctly). Its fixed architecture has a total stride of 32,
so every patch dimension must be divisible by 32:

```bash
nndet_train Task<XXX>_YourDataset retinaunet_focal_v002_for_ConvBackboneMultiTalent 0 \
    -o exp.tag=_<checkpoint_name> \
    +transfer_learning_ckpt=/path/to/checkpoint_final.pth \
    --transfer_learning --load_adapt_plan
```

**Primus** (needs `-o module=...`):

```bash
nndet_train Task<XXX>_YourDataset Primus_def_detr_v002 0 \
    -o module=BoxDeformableDETRV002_Primus_TL exp.tag=_<checkpoint_name> \
    +transfer_learning_ckpt=/path/to/checkpoint_final.pth \
    --transfer_learning --load_adapt_plan
```

(`exp.tag` again picks the output folder name, same as in Quick start --
every command in this section needs one to avoid overwriting a previous
run.)

If your downstream patch size differs from the checkpoint's pretraining
patch size, Primus's loader trilinearly interpolates the position embedding
automatically -- no extra flag needed.


### 3.4 Using a different MultiTalent stem

Which stem gets loaded defaults to the checkpoint's own
`nnssl_adaptation_plan.key_to_stem` -- unified across every backbone family
(ResEnc and ConvBackbone TL classes all read it the same way). To use a
*different* stem, no checkpoint copying or editing needed -- pass
`model_cfg.stem_override` instead:

```bash
nndet_train Task<XXX>_YourDataset residual_encoder_retinaunet_focal_v002 0 \
    -o module=RetinaUNetFocalV002_ResEnc_TL \
       model_cfg.stem_override=encoder.stem.004 \
       exp.tag=_MRIstem \
    +transfer_learning_ckpt=/path/to/checkpoint_final.pth \
    --transfer_learning --load_adapt_plan
```

ConvBackbone checkpoints use the bare `stem.<id>` prefix instead of
`encoder.stem.<id>` (e.g. `model_cfg.stem_override=stem.004`). Always set a
distinct `exp.tag` alongside a stem change. To find which stem id to
use: the checkpoint's trainer `training_log` records the dataset -> stem-id
mapping; the CT stem (`351_0`) is what the provided checkpoints' plans
default to.

## 4. Two-phase warmup finetuning

Instead of finetuning the whole model at one learning rate from step one, the
`*_TL_warmup*` module classes freeze most of the model for an initial warmup
phase (training only the newly-attached parts), then switch to training
everything.

| Class suffix | Phase 1 trains | Phase 2 trains |
|---|---|---|
| `_warmupdecoder_heads` (RetinaUNet) | neck, head | backbone, neck, head |
| `_warmupnet_1e3` (RetinaUNet) | backbone, neck, head | backbone, neck, head (different LR) |
| `_warmuptransformer_head` (DETR) | channel_mapper, transformer, head | backbone, channel_mapper, transformer, head |

These classes use PyTorch Lightning's manual-optimization mode with **two
separate optimizers**: `trainer_cfg.opt_class_1` during warmup
(`trainer_cfg.num_warmup_epochs`), `trainer_cfg.opt_class_2` after.
Trainer configs already set up for this (`nndet/conf/train/trainer_cfg/`):
SGD-based `sgd_base_warmupdecoder_heads(_1e3)`, `sgd_base_warmupnet_1e3`;
AdamW-based `adamw_100ep_high_lr_wd_warmuptransformer_head(_3e5)`,
`adamw_100ep_high_lr_wd_warmupnet_3e5`, `adamw_100ep_high_lr_wd_warmup_and_restart`.

```bash
nndet_train Task007_Pancreas residual_encoder_retinaunet_focal_v002 0 \
    -o module=RetinaUNetFocalV002_ResEnc_TL_warmupdecoder_heads \
       train/trainer_cfg@trainer_cfg=sgd_base_warmupdecoder_heads \
       exp.tag=_<checkpoint_name> \
    +transfer_learning_ckpt=/path/to/checkpoint_final.pth \
    --transfer_learning --load_adapt_plan
```
**Pass `--load_adapt_plan` here** -- the warmup classes are all fixed-architecture (workflow A, §3.2), so without it
the model builds with whatever shape the config's `backbone_kwargs` happens
to declare rather than the checkpoint's. The shipped ResEnc configs default
to exactly the ResEncL preset, so for a ResEncL checkpoint it currently
makes no difference -- but any checkpoint whose architecture differs (or any
edit to those defaults) fails at load with a `size mismatch` error without
it, so it is worth passing unconditionally. Verified end-to-end: ran a real `nndet_train`
subprocess across the warmup/finetune phase boundary (crossing
`num_warmup_epochs`) with a real checkpoint, confirming both the checkpoint
load and the two-optimizer phase switch work correctly.


## 5. Environment requirements

The ResEnc/Primus backbones depend on the external
`dynamic_network_architectures` package, pinned in
`requirements/finetuning.txt`:

```
dynamic_network_architectures>=0.4.4
timm<1.0.23
```

Install with:

```bash
pip install -r requirements/finetuning.txt
```
