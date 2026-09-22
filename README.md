<div align="center">

<img src=assets/nnDetection.svg width="600px">

![Version](https://img.shields.io/badge/nnDetection-v2.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-orange)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red)

</div>

# What is nnDetection?

Accurate object detection in 3D volumetric images is a fundamental challenge of computer vision and a key component of image-based biomedical workflows.
While many concurrent models focus on semantic segmentation, downstream applications often depend on object characteristics rather than individual voxels.
Deep learning has catalyzed a surge in volumetric 3D object detection methods, each using varying forms of supervision signals and detector types.
Variations in imaging modalities, image contexts, object structures, annotation types, and evaluation metrics complicate model selection and often result in sub-optimal performance.
These challenges restrict real-world applicability and reduce adaptability to new tasks, leading to limited adoption and reduced scientific progress.

nnDetection is a self-configuring method that unifies different detector types and training strategies by automating the complex design choices required for state-of-the-art detection.
On 10 diverse development datasets and 9 additional unseen datasets, nnDetection consistently outperforms baseline models across critical metrics, including mean Average Precision and Free-response Receiver Operating Characteristic.
Notably, on the three public benchmarks available today — LUNA16, PN9 and CTA-A — nnDetection sets a new state-of-the-art performance.
nnDetection represents a paradigm shift in volumetric 3D object detection by moving away from task-specific model design and towards a self-configuring and generalizing concept.
As an open-source tool, it provides an accessible, off-the-shelf solution that accelerates scientific progress and broadens adoption across the research community.

## Citation

**If you use nnDetection please cite our papers:**

```
Baumgartner M., Kovacs B., Ickler M.K., Jäger P.F., Isensee F., Ulrich C., Wald T., Holzschuh J.C., Ghosh P., for the ALFA study, Maier-Hein K.H.
nnDetection: A Self-configuring Method for Volumetric 3D Object Detection.
Nature Methods (in press, 2026).
```

```
Baumgartner M., Jäger P.F., Isensee F., Maier-Hein K.H. (2021)
nnDetection: A Self-configuring Method for Medical Object Detection.
MICCAI 2021. https://doi.org/10.1007/978-3-030-87240-3_51
```

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Using nnDetection](#using-nndetection)
  - [Planning & Preprocessing](#1-planning--preprocessing)
  - [Training](#2-training)
  - [Consolidation](#3-consolidation)
  - [Ensembling](#4-ensembling)
  - [Inference](#5-inference)
  - [Evaluation & Metrics](#evaluation--metrics)
- [Dataset Format](#dataset-format)
- [Dataset Guides](#dataset-guides)
- [Advanced Usage](#advanced-usage)
  - [Finetuning pretrained backbones](#finetuning-pretrained-backbones)
- [Developer Guide](#developer-guide)
- [FAQ & Troubleshooting](#faq--troubleshooting)
- [Acknowledgements](#acknowledgements)

# Installation

## Requirements

| Requirement | Version |
|---|---|
| CUDA | >= 10.1 (11.0+ with cuDNN 8.1+ recommended) |
| Python | >= 3.8 |
| PyTorch | >= 2.0 |
| GPU VRAM | >= 16 GB |

nnDetection requires a CUDA GPU — CPU deployment is limited to a subset of models and experimental (use at your own risk).
The default `D3V002Blosc` planner performs the VRAM estimation offline with a fixed set of heuristics, which keeps plans reproducible across GPUs and software versions.
The estimation targets a single reference architecture — the remaining models consume more than the estimated budget, so **at least 16 GB of VRAM are required**.
Scaling to larger GPUs currently has to be done manually (see some of our challenge participations for inspiration on scaling options).
The `*EstV1` planners (e.g. `D3V002EstV1`) perform live VRAM estimation on the current GPU like nnDetection V1 and can be used for arbitrary GPUs.

> **Note:** nnDetection was developed on Linux. Windows is not supported — use Docker there.

## Environment variables

| Variable | Description |
|---|---|
| `det_data` | Path to the source directory where all data is located |
| `det_models` | Path to the directory where all models are saved |
| `OMP_NUM_THREADS=1` | **Must** be set to 1, otherwise bad things will happen (see batchgenerators docs) |
| `det_num_threads` | Number of processes used for augmentation (at least 6, default 12) |

Optional:

| Variable | Description |
|---|---|
| `nndet_eval_max_detections_image_based` | Number of predictions per image per class used for evaluation (default 400) |
| `det_verbose` | Set to 0 to deactivate progress bars |
| `det_logging` | Logging directory (default: current training directory) |
| `det_logger` | `tensorboard` \| `mlflow` \| `none` (only tensorboard is installed by default) |
| `det_extended_logging` | Enable extended logging for some models (e.g. DETR criterions) |

## Source install with conda CUDA (recommended)

```bash
# PyTorch CUDA 12.1 was the default when this was written — adapt to your setup!
pip install torch torchvision
conda install cuda -c nvidia/label/cuda-12.1.1  # match the CUDA version of PyTorch
conda install gxx_linux-64  # specify version if needed
export CUDA_HOME=$CONDA_PREFIX
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc
git clone https://github.com/MIC-DKFZ/nnDetection
cd nnDetection
pip install -e . -v
```

Afterwards set the environment variables above and verify the installation (not from inside the repository folder):

```bash
python -c "import torch; import nndet._C; import nndet"
```

For a full development install (adds all unittest dependencies) use `pip install -e ".[dev]"`.

## Source install with local CUDA

Best performance, but only recommended for experienced users.

1. Install a [CUDA version compatible with your PyTorch build](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html).
2. [Optional] Depending on your GPU set `TORCH_CUDA_ARCH_LIST` (see [compute capabilities](https://developer.nvidia.com/cuda-gpus)).
3. Install `torch` and a matching `torchvision`.
4. `git clone ... && cd nnDetection && pip install -e .`
5. Set the environment variables and test the installation as above.

## Docker

Install [docker and nvidia-docker2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) first.
All projects based on nnDetection assume the base image was built with the `nnDetection:[version]` tagging scheme:

```bash
docker build -t nndetection:0.1 --build-arg env_det_num_threads=6 --build-arg env_det_verbose=1 .
```

The container expects data and models in `/opt/data` and `/opt/models`, which need to be mounted via `-v`.
Increasing the [shared memory](https://stackoverflow.com/questions/30210362/how-to-increase-the-size-of-the-dev-shm-in-docker-container) via `--shm-size` is required for training:

```bash
docker run --gpus all -v ${det_data}:/opt/data -v ${det_models}:/opt/models -it --shm-size=24gb nndetection:0.1 /bin/bash
```

# Quick Start

`nndet_example` generates a toy dataset of 3D squares and squares with holes which can be used to verify the installation or to experiment with prototype code.

```bash
# small data set to test the installation/environment (10 train, 10 test)
nndet_example

# full data set for prototyping (1000 train, 1000 test)
nndet_example --full [--num_processes]
```

Run the full pipeline on it:

```bash
nndet_prep 000
nndet_unpack_task 000 D3V001Blosc_3d
nndet_cv_split Task000D3_Example

# `toy` is a very short training schedule specific to the toy data set — never use it elsewhere!
nndet_train 000 toy 0 --sweep

nndet_predict_with_imagesTs 000 RetinaUNetFocalV002_D3V002Blosc_3d 0
```

The `full` problem is very easy and results should be near perfect even after a very short training.

> **Note on naming:** the plan is named `{planner}_{mode}` (e.g. `D3V002Blosc_3d`), the preprocessed data folder uses the *data identifier* of the planner (e.g. `D3V001Blosc_3d`, since `D3V001` and `D3V002` share the same preprocessing), and training directories are named `{module}_{plan}{exp tag}` (e.g. `RetinaUNetFocalV002_D3V002Blosc_3d`).

# Using nnDetection

Internally nnDetection is based on a complex set of rules and fixed/empirical optimisation, but it executes everything automatically — as a user you only run a few commands in sequence.
Detailed information for every command is available via `nndet_[command] -h`.

The typical flow of commands:

```
nndet_prep → nndet_unpack_task → nndet_cv_split → nndet_train → nndet_consolidate
           → nndet_determine_best_ensemble_with_task → nndet_predict_with_imagesTs
           → nndet_ensemble_with_determined_model
```

## 1. Planning & Preprocessing

Preprocessing normalizes and resamples the data, while the analyzed properties are used to create the plan that configures training.

```bash
nndet_prep [tasks] [-o / --overwrites] [-np / --num_processes] [-npp / --num_processes_preprocessing] [--full_check]

# Example
nndet_prep 000
```

`-o` overwrites planning/preprocessing parameters, `-np` controls the processes used for cropping and analysis, `-npp` those used for resampling (defaults are safe with 64 GB RAM).
`--full_check` iterates over the data before preprocessing and validates the format of data and labels — run it whenever preprocessing fails.

Resulting folder structure:

```text
[Task000_Example]
    [raw_splitted]
    [raw_cropped]                    # only needed for different resampling strategies
        [imagesTr]                   # cropped image data (npz files)
        [labelsTr]
    [preprocessed]
        [analysis]                   # plots visualizing properties of the data set
        [properties]                 # sufficient for new plans
        [labelsTr]                   # labels in original spacing
        [labelsTs]                   # optional
        [D3V001Blosc_3d]                  # data identifier of the plan
            [imagesTr]               # preprocessed data
            [labelsTr]               # preprocessed labels (resampled spacing)
        - D3V002Blosc_3d.json             # the plan
```

Before training, copy the task folder (dataset info + preprocessed folder are needed) to an SSD (highly recommended) and unpack the image data.
Note that unpacking takes the **data identifier** (the folder name inside `preprocessed`), not the plan name:

```bash
nndet_unpack_task [task] [data identifiers] [-p / --num_processes]
# Example
nndet_unpack_task 000 D3V001Blosc_3d

# alternative, path based
nndet_unpack ${det_data}/Task000D3_Example/preprocessed/D3V001Blosc_3d/imagesTr 6
```

Finally create the cross-validation split:

```bash
nndet_cv_split [task] [--num_folds] [--with_patients]

# Example
nndet_cv_split Task000D3_Example
```

For hierarchical data (e.g. multiple sessions per patient) use `--with_patients` or provide a [custom split](#custom-splits).

## 2. Training

nnDetection supports several object detection models which can be trained on multiple resolutions (if triggered during planning).
The default setup uses a 5-fold cross-validation, so `nndet_train` has to be run for `fold` 0–4.

```bash
nndet_train [task] [config_name] [fold] [-o / --overwrites] [--sweep] [--continue_training] [--log_net] [--log_aug]
```

Check which plans were generated by looking for the plan files in the preprocessed folder.
In most cases only the default plan (`D3V002Blosc_3d`) is generated, but for some data sets (e.g. KiTS) a low resolution plan (`D3V002Blosc_3dlr1`) is created as well.

### Which models to train

| Family | Command | Model |
|---|---|---|
| One-stage | `nndet_train 000 retinaunet_focal_v002 0 --sweep` | Retina U-Net V2 |
| One-stage | `nndet_train 000 retinaunet_focal_v002 0 -o module=RetinaNetFocalV002 --sweep` | Retina Net V2 |
| Two-stage | `nndet_train 000 retinaunet2sm_v002 0 --sweep` | Retina U-Net 2SM V2 |
| Two-stage | `nndet_train 000 retinaunet2sm_v002 0 -o module=RetinaNet2SV002 --sweep` | Retina Net 2S V2 |
| Set prediction | `nndet_train 000 def_detr_v002 0 --sweep` | Deformable DETR V2 |

Based on the `annotation_style` in your `dataset.yaml`, the model proposal stage recommends:

- **Dense annotations** (instance segmentations, including converted semantic segmentations): **Retina U-Net V2, Retina U-Net 2SM V2, Deformable DETR V2**
- **Weak annotations** (bounding boxes, spheres, etc.): **Retina Net V2, Retina Net 2S V2, Deformable DETR V2**

If multiple resolutions are available, we recommend first training the selected **one-stage** model on the different resolutions, picking the best resolution, and training the remaining models on it — the one-stage model has the shortest training time and offers the best tradeoff.

Training time varies between ~1 day (A100) and ~2 days (RTX2080TI) *per fold* (with working mixed precision acceleration of 3D convolutions and no other bottlenecks, see [FAQ](#faq--troubleshooting)).
`--continue_training` resumes from the last saved checkpoint.

### Sweeping

`--sweep` makes nnDetection search for the best inference hyperparameters by empirically evaluating them on the validation set. It can also be run separately:

```bash
nndet_sweep [task] [model] [fold]

# Example
nndet_sweep 000 RetinaUNetFocalV002_D3V002Blosc_3d 0
```

## 3. Consolidation

After all folds have been trained, collect the models and create a unified inference plan.
With `--sweep`, the empirical hyperparameter optimization is performed across all folds.

```bash
nndet_consolidate [task] [model] [--overwrites] [--consolidate] [--num_folds] [--no_model] [--sweep]

# Example
nndet_consolidate 000 RetinaUNetFocalV002_D3V002Blosc_3d --sweep
```

## 4. Ensembling

nnDetection determines the best ensemble of the trained models automatically:

```bash
nndet_determine_best_ensemble_with_task [task] [new model name] [+models to ensemble]

# for segmentation annotations
nndet_determine_best_ensemble_with_task 000 nnDetection_ensemble RetinaUNetFocalV002_D3V002Blosc_3d RetinaUNet2SMV002_D3V002Blosc_3d BoxDeformableDETRV002_D3V002Blosc_3d

# for weak annotations
nndet_determine_best_ensemble_with_task 000 nnDetection_ensemble RetinaNetFocalV002_D3V002Blosc_3d RetinaNet2SV002_D3V002Blosc_3d BoxDeformableDETRV002_D3V002Blosc_3d
```

This creates a new model folder with the config files needed for the ensembling, which is then executed via:

```bash
nndet_ensemble_with_determined_model [task] [model] [fold] [--test]

# Example
nndet_ensemble_with_determined_model 000 nnDetection_ensemble -1
```

For custom use cases, `nndet_ensemble_with_task`, `nndet_ensemble_with_models` and `nndet_ensemble_with_folders` allow parameters and models to be defined manually (see [nndet_scripts/ensemble.py](nndet_scripts/ensemble.py)).

## 5. Inference

Data located in `raw_splitted/imagesTs` is automatically preprocessed and predicted by:

```bash
nndet_predict_with_imagesTs [task] [model] [fold] [-ntta] [--skip_preprocessing] [-npp / --num_processes_preprocessing] [--load_models] [-o]

# Example (fold -1 uses the consolidated model)
nndet_predict_with_imagesTs 000 RetinaUNetFocalV002_D3V002Blosc_3d -1
```

For more fine-grained control over input and output use `nndet_predict_with_task`, `nndet_predict_with_folders` or `nndet_predict_test_split` (see [nndet_scripts/predict2.py](nndet_scripts/predict2.py)).

## Evaluation & Metrics

Evaluation requires access to the model and the preprocessed data:

```bash
nndet_eval [task] [model] [fold] [--test] [--case] [--boxes] [--analyze_boxes]

# Example (evaluate and analyze box predictions; --test evaluates the test set)
nndet_eval 000 RetinaUNetFocalV002_D3V002Blosc_3d 0 --boxes --analyze_boxes
```

`nndet_eval_with_folders` runs the evaluation on individual folders.

### Model directory

| Folder | Content |
|---|---|
| `[val/test]_predictions` | Predictions in the restored (original) image space |
| `[val/test]_results` | Validation/test results computed by nnDetection |
| `sweep` | Information from the parameter sweeps (debugging only) |
| `sweep_predictions` | Predictions with additional ensembler state used during empirical parameter optimization (larger than normal predictions, avoids re-running the model per sweep) |
| `val_predictions_preprocessed` | Predictions in the preprocessed image space (debugging only) |
| `val_results_preprocessed` | Validation results in the preprocessed image space (debugging only) |

### Metrics

Metrics are written to `[val/test]_results/results_boxes.json`.
Most metrics carry their IoU threshold in the name (e.g. `IoU_0.10`), are additionally reported per class (e.g. `YY_AP_IoU_0.10` for class `YY`), and some are extended with analysis functions such as volume ranges (indicated by extra letters, exact values in `results_curves`).

| Metric | Description |
|---|---|
| `AP_IoU_0.XX` | Main metric used in our papers. Detections per image per class are limited to 400 by default (`nndet_eval_max_detections_image_based`) |
| `mAP_IoU_0.XX_0.XX_0.XX` | COCO-style mAP over multiple IoU values. *The IoU thresholds differ from COCO to account for the generally lower IoU in 3D data* |
| `FROCwp_IoU_0.10` | FROC at FPPI values (1/8, 1/4, 1/2, 1, 2, 4, 8); sensitivity determined by the last working point (not interpolated). Pools all predictions, i.e. **not** per class |
| `mc_FROCwp_IoU_0.10` | FROC computed per class and averaged. Prefer this in multi-class settings to stratify for the number of objects per class |

### Visualizing predictions

- `nndet_boxes2mitkv2` — creates files in `[val/test]_predictions` for the latest version of MITK. Drag and drop the json files into MITK to see boxes with class and score. **Recommended**, best user experience.
- `nndet_boxes2nii` — creates nifti predictions in the original image space, viewable in any medical image viewer; scores are in the accompanying json files. Overlapping predictions are partially occluded.

# Dataset Format

nnDetection relies on a standardized input format very similar to [nnU-Net](https://github.com/MIC-DKFZ/nnUNet).

## Folders

All data sets reside inside `Task[Number]_[Name]` folders in `${det_data}`.
To avoid conflicts with provided pretrained models, use task numbers starting from 100.
Avoid `.` inside file names since it influences how paths are split.

File names follow `{patient id}_{session id}_{modality id}.{ext}` (if `session_id` is enabled in the dataset info) or `{patient id}_{modality id}.{ext}` (if disabled).
The first format groups multiple scans of the same patient to avoid leakage between training, validation and test sets.

```text
${det_data}
    [Task000_Example]
        - dataset.yaml                    # dataset.json works too
        [raw_splitted]
            [imagesTr]
                - case0000_000_0000.nii.gz  # patient case0000, session 000, modality 0
                - case0000_000_0001.nii.gz  # patient case0000, session 000, modality 1
                - case0001_000_0000.nii.gz
                - case0001_000_0001.nii.gz
            [labelsTr]
                - case0000_000.nii.gz       # instance segmentation of case0000, session 000
                - case0000_000.json         # instance properties of case0000, session 000
                - case0001_000.nii.gz
                - case0001_000.json
            [imagesTs]                      # optional, same structure as imagesTr
            [labelsTs]                      # optional, same structure as labelsTr
```

## Dataset info

`dataset.yaml` (or `dataset.json`) provides general information about the data set.
**Classes and modalities start at index 0!**

```yaml
# [mandatory information]
task: Task000D3_Example
dim: 3                      # number of spatial dimensions of the data

labels:                     # classes of the data set; need to start at 0
    "0": "Square"
    "1": "SquareHole"

modalities:                 # modalities of the data set; need to start at 0
    "0": "CT"

# "seg"  -> dense instance segmentation masks
# "weak" -> box or spherical annotation masks
# nnDetection proposes different models based on this information
annotation_style: "seg"

# [optional information]
session_id: False           # group multiple sessions of the same patient, default False
target_class: 1             # class of interest for patient level evaluations, default None
test_labels: True           # manually split test set for further evaluation, default False
```

## Image format

nnDetection uses the same image format as nnU-Net: each case consists of at least one 3D nifti file per modality, where the trailing number indicates the modality (consistent with the dataset info across the whole data set).

```text
- case001_000_0000.nii.gz  # patient case001; session 000; modality 0
- case001_000_0001.nii.gz  # patient case001; session 000; modality 1
```

All images need the same number of modalities.
nnDetection does **not** perform registration — check beforehand whether your modalities need to be registered and do so before preprocessing.

## Label format

Labels consist of two files per case: a nifti file with the instance segmentation and a json file with the "meta" information of each instance.
In the nifti file every instance has a unique number in consecutive order (`0` is ALWAYS background, `1` the first instance, `2` the second, ...).
The `case[XXXX].json` file assigns a class to every instance — below, instance `1` belongs to class `0` and instance `2` to class `1`:

```json
{
    "instances": {
        "1": 0,
        "2": 1
    }
}
```

Image and label files need to have the same size (in voxels).

Weak annotations must be converted into segmentation maps before use — we recommend representing them as boxes or spheres.
Depending on the chosen network the segmentation is not used during training, but we observed better results when augmenting dense masks rather than extreme points.
Since in 3D a single point can only be occupied by a single object, this can be done without loss of generality.
If boxes overlap slightly, start with the largest boxes and sequentially paste smaller objects on top to retain the size of the smallest objects.

# Dataset Guides

Besides being a self-configuring method, nnDetection acts as a standard interface for many data sets.
The [tasks](tasks) folder contains guides to prepare all data sets used in our evaluation.

> **Note:** The data sets are neither hosted nor maintained by us — please give credit to the original authors. Some labels were corrected in the data sets we converted (download links can be found in the individual guides).

<details>
<summary><b>Development pool (D01–D10)</b></summary>

| ID | Task | Guide |
|---|---|---|
| D01 | Task003 Liver | [tasks/Task001_Decathlon](tasks/Task001_Decathlon) |
| D02 | Task007 Pancreas | [tasks/Task001_Decathlon](tasks/Task001_Decathlon) |
| D03 | Task008 HepaticVessel | [tasks/Task001_Decathlon](tasks/Task001_Decathlon) |
| D04 | Task010 Colon | [tasks/Task001_Decathlon](tasks/Task001_Decathlon) |
| D05 | Task017 CADA | [tasks/Task017_CADA](tasks/Task017_CADA) |
| D06 | Task020 RibFrac | [tasks/Task020_RibFrac](tasks/Task020_RibFrac) |
| D07 | Task035 KiTS21 | [tasks/Task035_KiTS21](tasks/Task035_KiTS21) |
| D08 | Task036 PICAI | [tasks/Task036_PICAI](tasks/Task036_PICAI) |
| D09 | Task037 ADAM TOF A | [tasks/Task037_ADAM_TOF_A](tasks/Task037_ADAM_TOF_A) |
| D10 | Task045 LIDC | [tasks/Task044_LIDC_pylidc](tasks/Task044_LIDC_pylidc) |

</details>

<details>
<summary><b>Generalization pool (D11–D19)</b></summary>

| ID | Task | Guide |
|---|---|---|
| D11 | Task038 KiPA | [tasks/Task038_KiPA](tasks/Task038_KiPA) |
| D12 | Task052 MRAAneurysms | [tasks/Task052_MRAAneurysms](tasks/Task052_MRAAneurysms) |
| D13 | Task041 PancreasCysts | [tasks/Task041_PancreasCysts](tasks/Task041_PancreasCysts) |
| D14 | Task042 Duke | [tasks/Task042_Duke](tasks/Task042_Duke) |
| D15 | Task057 BraTSMets | [tasks/Task057_BraTSMets](tasks/Task057_BraTSMets) |
| D16 | Task056 PanoramaSubset | [tasks/Task056_PanoramaSubset](tasks/Task056_PanoramaSubset) |
| D17 | Task050 MELA | [tasks/Task050_MELA](tasks/Task050_MELA) |
| D18 | Task051 VALDO Microbleeds | [tasks/Task051_VALDO_Microbleeds](tasks/Task051_VALDO_Microbleeds) |
| D19 | Task054 LNDb | [tasks/Task054_LNDb](tasks/Task054_LNDb) |

</details>

<details>
<summary><b>Benchmarking pool</b></summary>

| ID | Task | Guide |
|---|---|---|
| D20 | Task016 Luna | [tasks/Task016_Luna](tasks/Task016_Luna) |
| D21 | Task053 PN9 | [tasks/Task053_PN9](tasks/Task053_PN9) |
| D22 | Task059 AneurysmCTA (CTA-A) | [tasks/Task059_AneurysmCTA](tasks/Task059_AneurysmCTA/README.md) |

</details>

<details>
<summary><b>Legacy data sets from nnDetection V1</b></summary>

| Task | Guide |
|---|---|
| Task011 KiTS | [tasks/Task011_Kits](tasks/Task011_Kits) |
| Task012 LIDC | [tasks/Task012_LIDC](tasks/Task012_LIDC) |
| Task021 ProstateX | [tasks/Task021_ProstateX](tasks/Task021_ProstateX) |
| Task025 LymphNodes (updated to Task046) | [tasks/Task025_LymphNodes](tasks/Task025_LymphNodes) |

</details>

# Advanced Usage

## Finetuning pretrained backbones

Using a pretrained ResEnc or Primus backbone (optionally initialized
from a self-supervised pretraining checkpoint, with a two-phase warmup
finetuning schedule) is covered in
[docs/finetuning.md](docs/finetuning.md).

## Custom splits

Place a `[your_split].json` file inside the `preprocessed` folder of the task.
It contains a list over the folds, where each item is a dict with the keys `train` and `val` holding the case ids of the respective split.
Extend the training command with `+io_cfg.splits=[your_split]`.

## Standalone evaluation framework

If labels and predictions are in the nnDetection format, evaluating box predictions is as simple as:

```python
from nndet.eval.registry import evaluate_box_dir

results = evaluate_box_dir(
    classes=[YOUR CLASSES HERE],
    pred_dir=[YOUR PATH HERE],
    gt_dir=[YOUR PATH HERE],
)
```

`nndet_eval_with_folders` is a direct command line entrypoint to this functionality.

The nnDetection format expects ground truth labels in `npz` files with the keys `boxes` and `classes`, and predictions in `pkl` files with the keys `pred_boxes`, `pred_labels` and `pred_scores`.
All boxes need to be in the same coordinate system and follow the `ax0_min, ax1_min, ax0_max, ax1_max, ax2_min, ax2_max` format (`ax` denotes arbitrary axes).

Custom evaluation scripts can be created by passing predictions and ground truth boxes to the evaluator directly:

```python
from nndet.eval.det import BoxEvaluator

evaluator = BoxEvaluator.create(
    classes=[YOUR CLASSES HERE],
    fast=False,
    verbose=True,
    similarity_fn=[YOUR SIMILARITY FUNCTION e.g. box_iou_np],
)

for case_id in case_ids:
    gt = [LOAD GROUND TRUTH]
    pred = [LOAD PREDICTION]

    evaluator.run_online_evaluation(
        pred_boxes=[pred["pred_boxes"]],
        pred_classes=[pred["pred_labels"]],
        pred_scores=[pred["pred_scores"]],
        gt_boxes=[gt["boxes"]],
        gt_classes=[gt["classes"]],
        gt_ignore=None,
        case_ids=[case_id],
    )

results = evaluator.finish_online_evaluation()
```

# Developer Guide

<details>
<summary><b>Registries</b></summary>

nnDetection uses multiple registries to keep track of exchangeable modules.
To register a new component, import the registry in your python file and wrap the component with the decorator:

```python
from nndet.ptmodule import MODULE_REGISTRY

@MODULE_REGISTRY.register  # registers under the class name
class RetinaUNetV001(...):
    ...
```

The registry can then be accessed like any dictionary:

```python
from nndet.ptmodule import MODULE_REGISTRY

module_cls = MODULE_REGISTRY["RetinaUNetV001"]
module = module_cls(model_cfg=model_cfg, trainer_cfg=trainer_cfg, plan=plan)
```

Available components can be listed from the command line — useful when overwriting config keys:

```bash
nndet_print_reg module        # list all modules
nndet_print_reg augmentation  # list all augmentations
```

| Registry | Import from | Purpose |
|---|---|---|
| `MODULE_REGISTRY` | `nndet.ptmodule` | Core PyTorch Lightning modules used for training and inference (examples in `nndet.ptmodule.retinaunet`) |
| `AUGMENTATION_REGISTRY` | `nndet.io.augmentation` | Augmentation configurations (examples in `nndet.io.augmentation.bg_aug`) |
| `DATALOADER_REGISTRY` | `nndet.io.datamodule` | Dataloader classes to customize IO (examples in `nndet.io.datamodule.bg_loader`) |
| `PLANNER_REGISTRY` | `nndet.planning.experiment` | Architecture and preprocessing schemes (examples in `nndet.planning.experiment.v001`) |
| `OPTIMIZER_REGISTRY` | — | Optimizers, selected via `trainer_cfg.opt_class` |

> **Warning:** A component is only registered if its python file is imported at runtime. The core repository does this for the provided base configurations automatically. Components living outside the repository need to be imported manually, e.g. via the `additional_imports: ["my_package"]` entry in a custom config file.

</details>

<details>
<summary><b>Config files</b></summary>

The config files provide information for model configuration (fixed parameters), data loading, augmentation and training.
Training directories are composed of three parts: `{module name}_{plan name}_{exp tag}` — by changing the exp tag, different training runs with varied hyperparameters can be created.
Parts of the configs can be overwritten via `-o train/{XXX}_cfg@{key}_cfg={value}`, e.g. `-o train/augment_cfg@augment_cfg=my_custom_aug`.

- **Augmentation** — `name` is a short identifier for lookup in the saved json config, `transforms` defines the augmentation transformations retrieved through the augmentation registry. Remaining parameters depend on the selected pipeline. nnDetection also provides an interface to MONAI augmentations (see `MonaiTransform`).
- **Data loading** — `dataloader` specifies the dataloader class retrieved from the dataloader registry, remaining parameters depend on the selection.
- **Trainer config** — learning rate, training length, observed metrics and optimizer hyperparameters. `opt_class` selects the optimizer from the optimizer registry.
- **Accelerator config** — hardware resources and model optimisation settings. Additional speedups are sometimes possible with `gpu1_mixed16_bench`, but it does not work for all patch sizes / model configurations.
- **Model config** — fixed parameters of the selected architecture. The exact set varies between models and needs to be cross-referenced with the model parameters in the code.

</details>

<details>
<summary><b>Custom preprocessing</b></summary>

The experiment planner defines the entire planning and preprocessing pipeline; the primary entry point during planning is `plan_experiment`.
Individual parameters can be customized by overwriting the respective function, e.g. `determine_dummy_2d_data_augmentation`, `determine_forward_backward_permutation`, `determine_target_spacing` and `trigger_low_res_model`.
`create_architecture_planner` (batch size, patch size, kernels, ...) and `create_preprocessor` (resampling, intensity normalisation, ...) can be overwritten to implement other planner/preprocessor classes.

</details>

<details>
<summary><b>Custom models</b></summary>

All modules build on `LightningBaseModule` (`nndet.ptmodule.module`), which extends the PyTorch Lightning module with procedures to set up transformations, evaluation and the prediction pipeline, and configures standardized callbacks like `EpochTimerCallback` and Stochastic Weight Averaging (enabled via the `swa_epochs` key).

Each detection module is a combination of `LightningBaseModule` and multiple mixins, which lets nnDetection cover various input/output formats (bounding box detection and instance segmentation, each with auxiliary task training):

```python
class RetinaNetModule(
    LightningBaseModule,   # nnDetection base module integrating the other mixins
    BoxesPrepareMixin,     # convert the dataloader output to bounding boxes
    BoxEvalMixin,          # run bounding box evaluation during training
    SingleStageMixin,      # use the model structure of a single stage detector
    BoxPredictionMixin,    # run the default bounding box prediction and sweep
):
    # customize SingleStageMixin attributes to switch between backbones, necks,
    # heads, sampling strategies, losses and much more
    ...
```

**Model mixins** (`nndet.ptmodule.mixins.model`) build the model and provide the `from_config_plan` classmethod called by `LightningBaseModule`.
Most changes can be made through class attributes without overwriting methods:

```python
from nndet.nn.heads.classifier import BCECLassifier

class SingleStageDetectorBCELoss(..., SingleStageMixin, ...):
    head_classifier_cls = BCECLassifier
```

**Prepare mixins** convert the dataloader output into the desired target format (customizable via `get_pre_transforms`). Multiple prepare mixins can be combined — e.g. Retina U-Net requires boxes and semantic segmentations:

| Mixin | Output keys |
|---|---|
| `BoxesPrepareMixin` | `boxes`, `classes` |
| `SemanticPrepareMixin` | `target_seg` |
| `SemanticFgPrepareMixin` | `target_seg` (foreground vs background) |
| `BinaryMasksPrepareMixin` | `target_binary_masks` |

**Eval mixins** define the metrics tracked during training via `evaluation_init` (initialize the `Evaluator`, see `nndet.evaluator`), `evaluation_step` (called every validation step, caches intermediate results) and `evaluation_end` (aggregates predictions and ground truth to compute the final metrics).

> **Warning:** Cached values inside the `Evaluator` are not synchronized between workers in distributed training — every worker computes its own metrics which are then averaged.

**Predict mixins** cover `sweep` (determine the best inference parameters after training) and `get_predictor` (create the predictor for inference), and can be used to exchange the `Sweeper`, `Predictor` and `Ensembler` objects.

</details>

<details>
<summary><b>Custom losses</b></summary>

There are four loss categories in nnDetection (`#dims` = number of spatial dimensions, `N` = number of anchors, `R` = number of RoIs, `T` = number of matched boxes, `B` = batch size, `C` = number of classes):

- **`regression`** — usually used for regression tasks (e.g. bounding boxes) and receive inputs of shape `[*, #dims * 2]`, e.g. RetinaNet `[N, #dims * 2]`, RCNN `[R, #dims * 2]`, DETR `[T, #dims * 2]`.
- **`classification`** — receive inputs of shape `[*, C]` where `C` is the number of *foreground* classes, e.g. RetinaNet `[N, C]`, RCNN `[R, C]`, DETR `[B, T, C]`. Note this differs from PyTorch, where the class dimension usually comes first. Targets are numerical values where `0` is background, so when creating the one hot encoding the first channel (filled with 0s) is removed and the maximal target value is `C+1`. This differs from the segmentation losses, where the number of classes refers to the *total* number of classes (maximal target value `C`).
- **`segmentation`** — per-location classification of feature maps, input shape `[B, C, *]`.
- **`mask`** — per-location binary classification of feature maps. Same input format as segmentation losses, but the targets are already one hot encoded, i.e. shape `[B, C, *]`.

</details>

<details>
<summary><b>Unittests and distribution packages</b></summary>

Run the unittests from the root directory (requires the `dev` installation):

```bash
pytest .
```

Build distribution packages:

```bash
python setup.py bdist_wheel  # binary distribution (wheel)
python setup.py sdist        # source distribution (tarball)
```

Both end up in the `dist` directory. To build them inside docker:

```bash
docker run --rm --gpus all -v .:/opt/nndet --shm-size=48gb continuumio/miniconda3 /bin/bash -c "conda create --name venv python=3.10 -y && source activate base && conda activate venv && export CXX=\$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++ && export CC=\$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc && pip3 install torch torchvision torchaudio && conda install cuda -c nvidia/label/cuda-\$(python -c \"import torch; print(torch.version.cuda)\") -y && conda install gxx_linux-64 -y && cd /opt/nndet && rm -rf build nndet.egg-info && python setup.py bdist_wheel && python setup.py sdist"
```

</details>

# FAQ & Troubleshooting

<details>
<summary><b>Undefined CUDA symbols / import errors from <code>nndet._C</code> / CUDA ARCH errors</b></summary>

nnDetection includes CUDA code which is compiled during installation and therefore requires a correct configuration of the CUDA dependencies.
Double check the CUDA versions of your machine, PyTorch, torchvision and the nnDetection build by running `nndet_env`.

Things to look out for:

- The CUDA versions of PyTorch and NVCC should match (a minor version mismatch usually works but can introduce subtle bugs).
- `OMP_NUM_THREADS` must always be 1 and `det_num_threads` must be lower or equal to the system CPU count.
- Delete the `build` folder before rerunning the installation, otherwise the code is not recompiled.

</details>

<details>
<summary><b>Error: No kernel image is available for execution</b></summary>

The build was probably executed on a machine with a GPU architecture that was not present/set during the build.
Look up the correct SM architecture [here](https://developer.nvidia.com/cuda-gpus) and set `TORCH_CUDA_ARCH_LIST` appropriately (see the Dockerfile for an example).
Delete the `build` folder before rerunning the installation.

</details>

<details>
<summary><b>Training doesn't start or is stuck</b></summary>

- Run `nndet_env` and make sure `OMP_NUM_THREADS` is set to 1 — no other value is supported. To increase the number of workers for IO and augmentation adjust `det_num_threads`.
- As a sanity check, run the training without multiprocessing: `nndet_train XXX -o augment_cfg.multiprocessing=False`. This is a debugging step only, don't use it for a full training.
- If the problem persists, open an issue with the output of `nndet_env` and report whether the training without multiprocessing started correctly.

</details>

<details>
<summary><b>Multi GPU training</b></summary>

Multi GPU training is not officially supported.
It can be performed by increasing the number of GPUs in lightning, but the online validation will not compute meaningful metrics (they are simply averaged across GPUs) and inference including the final validation does not support multi GPU setups.
Other aspects influencing model performance (e.g. scaling the number of steps) were never tested. **Use multi GPU at your own risk.**

</details>

<details>
<summary><b>2D data sets</b></summary>

2D data sets are not supported since there are already excellent external repositories for these detection tasks, e.g. [generalized_yolov5](https://github.com/MIC-DKFZ/generalized_yolov5).

</details>

<details>
<summary><b>GPU requirements</b></summary>

At least 16 GB of VRAM are required.
The offline VRAM estimation of the default `D3V002Blosc` planner targets a single reference architecture, and the remaining models consume more than that budget — with less memory the training will run out of memory or performance will suffer significantly.
Scaling to larger GPUs currently has to be done manually (see some of our challenge participations for inspiration on scaling options).

</details>

If your problem is not covered here, please open an issue and provide your environment as obtained by `nndet_env`.

# Acknowledgements

nnDetection combines information from multiple open source repositories which we wish to acknowledge for their awesome work — please check them out!

- **[nnU-Net](https://github.com/MIC-DKFZ/nnUNet)** — a self-configuring method for semantic segmentation; many steps of nnDetection follow in the footsteps of nnU-Net.
- **[Medical Detection Toolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)** — introduced the first codebase for 3D object detection; multiple tricks were transferred to nnDetection to assure optimal configuration for medical object detection.
- **[Torchvision](https://github.com/pytorch/vision)** — nnDetection follows the torchvision interfaces to make it easy to understand for everyone coming from the 2D (and video) detection scene, and bases some of its core modules on the torchvision implementation.
- **[transoar](https://github.com/bwittmann/transoar)** — 3D Deformable Attention for Deformable DETR was integrated from transoar and was extremely helpful. We are grateful for the open source release of this code.
- **DETR** — components from multiple repositories were adapted for 3D use. We would like to thank the authors for their great work and for open sourcing their code under nice licenses: [DETR](https://github.com/facebookresearch/detr), [Conditional DETR](https://github.com/Atten4Vis/ConditionalDETR), [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), [detrex](https://github.com/IDEA-Research/detrex).

## Funding

Part of this work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – 410981386 and the Helmholtz Imaging Platform (HIP), a platform of the Helmholtz Incubator on Information and Data Science.

## License

This project is licensed under multiple licenses, please refer to the [LICENSES](LICENSES) directory for an overview.

## Copyright

Copyright German Cancer Research Center (DKFZ) and contributors.
