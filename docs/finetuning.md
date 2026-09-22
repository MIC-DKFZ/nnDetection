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
    -o module=RetinaUNetFocalV002_ResEnc_TL \
    +transfer_learning_ckpt=/path/to/checkpoint_final.pth \
    --transfer_learning --load_adapt_plan
```

or **Deformable DETR** head instead (same checkpoints work with either head):

```bash
nndet_train Task<XXX>_YourDataset residual_encoder_def_detr_v002 0 \
    -o module=BoxDeformableDETRV002_ResEnc_TL \
    +transfer_learning_ckpt=/path/to/checkpoint_final.pth \
    --transfer_learning --load_adapt_plan
```

Replace `Task<XXX>_YourDataset` with your own preprocessed task, and the
checkpoint path with one from the table below. `--load_adapt_plan` reconciles
the model's architecture to match the checkpoint; `--transfer_learning` loads
the weights. **Before running this for real, skim §5 (Troubleshooting)** --
in particular the notes on `planner` mismatches and `exp.tag`, which trip up
almost every first run.

## Provided checkpoints

| Checkpoint (`trainer_name`) | What it is | Backbone | Recommended config |
|---|---|---|---|
| `ModelGenesis` | "Model Genesis"-style self-supervised pretraining | ResEnc | `residual_encoder_retinaunet_focal_v002` or `residual_encoder_def_detr_v002` + matching `_TL` module (Quick start, or §3.3) |
| `VariableSpark` | Variable Spark MAE self-supervised pretraining | ResEnc | same as above |
| `VoCo` | "VoCo"-style self-supervised pretraining | ResEnc | same as above |
| `MultiTalent` | MultiTalent joint segmentation pretraining across many datasets (multiple stems; defaults to a CT stem) | ResEnc | same as above; see §3.4 to pick a different stem |
| `MultiTalent_meets_nndet` | MultiTalent joint pretraining (multiple stems; defaults to a CT stem) | nndetection's native `ConvBackbone` | `retinaunet_focal_v002_for_ConvBackboneMultiTalent` (§3.3); see §3.4 to pick a different stem |
| `nnFoundationCNN` | Masked-autoencoder (MAE) self-supervised pretraining | ResEnc | same as above |
| `nnFoundationViT` | EVA-style masked-autoencoder self-supervised pretraining | Primus (ViT) | `Primus_def_detr_v002` + `BoxDeformableDETRV002_Primus_TL` (§3.3) |

All pretrained on single-channel (grayscale) 3D volumes; the MultiTalent ones
across ~87 datasets of mixed modalities, hence the multiple stems. The same
`-o module=...` / `+transfer_learning_ckpt=...` / `--transfer_learning
--load_adapt_plan` pattern from Quick start works for every row -- only the
top-level config and module name differ (§3.3 has the exact command per row).

**Where to get these checkpoint files:** <!-- TODO: fill in the actual
download location (e.g. a Zenodo/HuggingFace release, or an institutional
download link) before publishing this document externally -->.

## 1. Supported combinations

| Backbone | Detection head | Base module class | Checkpoint-loading variant |
|---|---|---|---|
| ResEnc (fixed architecture) | RetinaUNet | `RetinaUNetFocalV002_ResEnc` | `RetinaUNetFocalV002_ResEnc_TL` |
| ResEnc (**dynamic** architecture) | RetinaUNet | `RetinaUNetFocalV002_ResEnc_dyn` | `RetinaUNetFocalV002_ResEnc_dyn_TL` |
| ResEnc (fixed architecture) | Deformable DETR | `BoxDeformableDETRV002_ResEnc` | `BoxDeformableDETRV002_ResEnc_TL` |
| ResEnc (**dynamic** architecture) | Deformable DETR | `BoxDeformableDETRV002_ResEnc_dyn` | `BoxDeformableDETRV002_ResEnc_dyn_TL` |
| Primus (EVA-style ViT) | Deformable DETR | `BoxDeformableDETRV002_Primus` | `BoxDeformableDETRV002_Primus_TL` |
| ConvBackbone (fixed architecture, MultiTalent-style stem) | RetinaUNet | `DetSegModel` | `DetSegModel_TL_MultiTalentStem` |

"Fixed architecture" means the backbone's shape comes from the model config
you write. "Dynamic architecture" (`_dyn` classes) means the shape instead
comes from nnDetection's own planned architecture -- use this when you want
nnDetection's normal architecture search/dataset adaptation while still
loading as much as possible from a checkpoint, even if its depth/shape
doesn't exactly match. Only ResEnc has a dynamic variant.

The last row (`ConvBackbone`) is nndetection's own native architecture, not
ResEnc -- it loads MultiTalent checkpoints whose input stem was trained as a
separate per-dataset module. See §6 for exactly how the loading differs
mechanically from the ResEnc classes.

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
}
```

Set the checkpoint path via a **top-level** CLI override --
`+transfer_learning_ckpt=/path/to/checkpoint.pt` (§5 explains the `+`).

### 3.2 CLI flags

| Flag | Effect |
|---|---|
| `-tl` / `--transfer_learning` | Load pretrained weights via the module's `load_custom_state_dict(path)`. Requires a `_TL` module class. |
| `--load_adapt_plan` | Before building the model, overwrite `model_cfg.backbone_kwargs` (and the plan's `architecture` section) from the checkpoint's own `nnssl_adaptation_plan.architecture_plans` -- forces the model's architecture to match the checkpoint. §6 has the exact resolution algorithm. |
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
    +transfer_learning_ckpt=/path/to/checkpoint.pt \
    --transfer_learning
```

```bash
nndet_train Task<XXX>_YourDataset residual_encoder_def_detr_v002_dyn 0 \
    +transfer_learning_ckpt=/path/to/checkpoint.pt \
    --transfer_learning
```

**MultiTalent-meets-nndet** (native `ConvBackbone`, not ResEnc; module
already defaults correctly) -- see §5 for a real patch-size gotcha with
this one:

```bash
nndet_train Task<XXX>_YourDataset retinaunet_focal_v002_for_ConvBackboneMultiTalent 0 \
    +transfer_learning_ckpt=/path/to/checkpoint.pt \
    --transfer_learning --load_adapt_plan
```

**Primus** (needs `-o module=...`; its `arch_kwargs` isn't guaranteed
populated for every Primus checkpoint -- see §6 if not) -- see §5 for a
real memory gotcha with this one:

```bash
nndet_train Task<XXX>_YourDataset Primus_def_detr_v002 0 \
    -o module=BoxDeformableDETRV002_Primus_TL \
    +transfer_learning_ckpt=/path/to/checkpoint.pt \
    --transfer_learning --load_adapt_plan
```

If your downstream patch size differs from the checkpoint's pretraining
patch size, Primus's loader trilinearly interpolates the position embedding
automatically -- no extra flag needed.

**Verification status:** every command above (both heads x fixed/dynamic
ResEnc, ConvBackboneMultiTalent, Primus) was run as a real `nndet_train`
subprocess end-to-end (checkpoint load -> real training iterations -> clean
stop) against real checkpoints from every trainer family in the table above,
with zero missing/unexpected keys on load. The §5 gotchas came directly out
of those runs.

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
    +transfer_learning_ckpt=/path/to/checkpoint.pt \
    --transfer_learning --load_adapt_plan
```

ConvBackbone checkpoints use the bare `stem.<id>` prefix instead of
`encoder.stem.<id>` (e.g. `model_cfg.stem_override=stem.004`). Always set a
distinct `exp.tag` alongside a stem change (§5). To find which stem id to
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

(`_warmupdecoder`/`_warmupheads`/`_warmuptransformer` -- warming up only
*part* of the randomly-initialized new parts -- were removed as
unmotivated: nothing justifies freezing part of an already-random submodule.
`_warmupnet_1e3`, training everything at a different LR, is a genuinely
different idea and was kept.)

e.g. `RetinaUNetFocalV002_ResEnc_TL_warmupdecoder_heads`,
`DetSegModel_TL_warmupdecoder_heads`,
`BoxDeformableDETRV002_ResEnc_TL_warmuptransformer_head`,
`DetSegModel_TL_MultiTalentStem_warmupdecoder_heads`. `_1e3`-suffixed classes
(e.g. `..._warmupdecoder_heads_1e3`) are identical to their non-suffixed
counterpart -- they exist only so a Hydra config can select a different
`trainer_cfg` (1e-3 LR), not because the class differs.

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
    --transfer_learning
```

Note the `train/trainer_cfg@trainer_cfg=<name>` syntax for selecting a
different trainer_cfg -- see §5.

## 5. Troubleshooting

Real gotchas hit while building and testing this, roughly in the order
you'll run into them:

- **`+` vs `-o` on the CLI.** `+key=value` adds a brand-new key (Hydra's
  syntax for a key not already declared anywhere); `-o key=value` only
  overrides a key that's already part of the resolved config. `transfer_learning_ckpt`
  needs `+` (it's a fresh top-level key); `model_cfg.stem_override` needs
  plain `-o` (it's pre-declared with an empty-string default in every
  relevant model_cfg). If you hit `"... is not in struct"`, swap `-o` for
  `+` on that specific override.
- **Selecting a different `trainer_cfg`/`model_cfg`/`io_cfg` file needs `@`,
  not `-o`.** These are chosen via a *nested* `defaults:` entry inside each
  `train/*.yaml`. `-o trainer_cfg=<name>` silently replaces the whole
  resolved `trainer_cfg` dict with the literal string `"<name>"`, failing
  later with a confusing `TypeError`. Use
  `train/trainer_cfg@trainer_cfg=<name>` instead. Overriding a single
  **leaf key** inside an already-selected file (e.g.
  `model_cfg.stem_override=...`) is unaffected -- only *swapping which file*
  a nested group loads needs `@`.
- **Plan/planner mismatch.** Every top-level config has a default `planner`
  (e.g. `D3V002`), and `nndet_train` looks for a plan preprocessed under
  that exact name. If you preprocessed under a different name (e.g.
  `D3V002Blosc` -- same thing, just Blosc2 storage format instead of
  `.npz`), add `-o planner=D3V002Blosc`. Check
  `<det_data>/<your task>/preprocessed/` for the plan names you actually have.
- **`exp.tag` and checkpoint provenance.** The output folder name is
  `${module}_${plan}${exp.tag}` -- it does *not* encode which checkpoint you
  passed via `transfer_learning_ckpt`. Running two different checkpoints
  through the same module+config without a distinct `exp.tag` each time
  silently overwrites the first run's output directory. Adopt a convention
  (`-o exp.tag=_VoCo`, `_ModelGenesis`, ...). Every run also saves
  `meta.json` (raw CLI overwrites) and `config_resolved.yaml` (fully
  resolved config) regardless, so you can always check after the fact which
  checkpoint a given run used.
- **`arch_class_name` unknown to this repo.** `--load_adapt_plan` silently
  does nothing (just a logged warning) if the checkpoint's plan has
  `arch_kwargs: None` and an `arch_class_name` not in `BACKBONE_PRESETS`
  (currently only `"ResEncL"` is registered). Your model then builds with
  whatever `backbone_kwargs` your yaml already had -- the failure only
  surfaces later, as a `load_state_dict` shape/key mismatch. §6 has the full
  resolution algorithm.
- **ConvBackboneMultiTalent patch-size requirement.** Its fixed architecture
  has a total stride of 32 (6 stages / 5 downsampling steps) -- your task's
  patch size needs every dimension divisible by 32, or model construction
  fails with `Backbone ConvBackbone with absolute strides [...] is not
  compatible with patch size [...]`. Encountered directly during testing on
  a task with patch size `[80, 160, 128]` (80 isn't divisible by 32);
  `[160, 128, 128]` worked.
- **Primus memory.** Its default `batch_size: 4` at the full 192³ patch OOM'd
  on a single RTX 3090 during testing (embed_dim 864, 16 encoder layers is a
  large transformer). `-o model_cfg.backbone_kwargs.batch_size=1` trained
  cleanly. Try that first if you hit an OOM here.

## 6. Internals

Deeper mechanism notes, for extending this code rather than just using it.

**`arch_kwargs` vs. `BACKBONE_PRESETS` resolution** (`--load_adapt_plan`,
`nndet_scripts/train.py`): reads `nnssl_adaptation_plan.architecture_plans`.
If `arch_kwargs` is a dict, it's used directly -- for each key already
present in your `model_cfg.backbone_kwargs`, if that key also exists in
`arch_kwargs` it gets overwritten (keys not already in your `backbone_kwargs`
are ignored, hence §2.1's placeholder-keys note); no preset lookup happens.
If `arch_kwargs` is `None`, it falls back to
`BACKBONE_PRESETS[arch_class_name]` (`nndet/utils/pretrained_backbone_presets.py`)
the same way, or does nothing if `arch_class_name` isn't registered there
(§5).

**Weight-loading mechanics per family:** The ResEnc/Primus `_TL` classes
strip the checkpoint's `key_to_encoder`/`key_to_stem` prefixes and load into
the corresponding submodule directly (`get_submodule` + `load_state_dict`).
If the checkpoint has fewer input channels than your downstream data, the
first projection layer's weights are repeated across the extra channels
(`weight.repeat(1, N, 1, 1, 1) / N`). The `_dyn_TL` variant additionally
drops pretrained stages beyond the target architecture's depth (or leaves
extra target stages at random init if the target is deeper) and adapts
convolution kernels by mean-reducing a spatial dimension to size 1 when
shapes differ (expansion isn't supported). Primus's loader additionally
trilinearly interpolates the absolute position embedding when patch sizes
differ.

`DetSegModel_TL`/`DetSegModel_TL_MultiTalentStem` (ConvBackbone) work
differently, since `ConvBackbone` has no standalone stem/encoder submodule
to target with `get_submodule`: they build one remapped `state_dict` for the
whole model (renaming `decoder.*` -> `neck.*`, fusing the checkpoint's chosen
stem into `levels.0`'s first conv block and shifting the checkpoint's own
`levels.0` block into the second one) and call `self.load_state_dict(...,
strict=False)` once, relying on `strict=False` to leave neck/head at random
init. `retinaunet_focal_v002_for_ConvBackboneMultiTalent.yaml`'s `model_cfg`
pre-populates `backbone_kwargs` with placeholder
`features_per_stage`/`kernel_sizes`/`strides`/`fpn_channels`/`decoder_levels`
purely so `--load_adapt_plan` has existing keys to overwrite (`ConvBackbone`
never reads these directly -- `nndet_scripts/train.py`'s `nnssl_to_plan`
mapping relays them into `plan["architecture"]`, which is what `ConvBackbone`/
`UFPN` actually read).

## 7. Environment requirements

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
