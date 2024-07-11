<div align="center">

<img src=docs/source/nnDetection.svg width="600px">

![Version](https://img.shields.io/badge/nnDetection-v0.1-blue)
![Python](https://img.shields.io/badge/python-3.8+-orange)
![CUDA](https://img.shields.io/badge/CUDA-10.1%2F10.2%2F11.0-green)

</div>

# What is nnDetection?
Simultaneous localisation and categorization of objects in medical images, also referred to as medical object detection, is of high clinical relevance because diagnostic decisions depend on rating of objects rather than e.g. pixels.
For this task, the cumbersome and iterative process of method configuration constitutes a major research bottleneck. 
Recently, nnU-Net has tackled this challenge for the task of image segmentation with great success.
Following nnU-Net’s agenda, in this work we systematize and automate the configuration process for medical object detection.
The resulting self-configuring method, nnDetection, adapts itself without any manual intervention to arbitrary medical detection problems while achieving results en par with or superior to the state-of-the-art.
We demonstrate the effectiveness of nnDetection on two public benchmarks, ADAM and LUNA16, and propose 10 further public data sets for a comprehensive evaluation of medical object detection methods.

**If you use nnDetection please cite our [paper](https://miccai2021.org/openaccess/paperlinks/2021/09/01/341-Paper1836.html)**:
```
Baumgartner M., Jäger P.F., Isensee F., Maier-Hein K.H. (2021) nnDetection: A Self-configuring Method for Medical Object Detection. In: de Bruijne M. et al. (eds) Medical Image Computing and Computer Assisted Intervention – MICCAI 2021. MICCAI 2021. Lecture Notes in Computer Science, vol 12905. Springer, Cham. https://doi.org/10.1007/978-3-030-87240-3_51
```
:tada: nnDetection was early accepted to  the International Conference on Medical Image Computing & Computer Assisted Intervention 2021 (MICCAI21) :tada:

# Installation
## Docker
The easiest way to get started with nnDetection is the provided is to build a Docker Container with the provided Dockerfile.

Please install docker and [nvidia-docker2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) before continuing.

All projects which are based on nnDetection assume that the base image was built with the following tagging scheme `nnDetection:[version]`.
To build a container (nnDetection Version 0.1) run the following command from the base directory:

```bash
docker build -t nndetection:0.1 --build-arg env_det_num_threads=6 --build-arg env_det_verbose=1 .
```

(`--build-arg env_det_num_threads=6` and `--build-arg env_det_verbose=1` are optional and are used to overwrite the provided default parameters)

The docker container expects data and models in its own `/opt/data` and `/opt/models` directories respectively.
The directories need to be mounted via docker `-v`. For simplicity and speed, the ENV variables `det_data` and `det_models` can be set in the host system to point to the desired directories. To run:

```bash
docker run --gpus all -v ${det_data}:/opt/data -v ${det_models}:/opt/models -it --shm-size=24gb nndetection:0.1 /bin/bash
```

Warning:
When running a training inside the container it is necessary to [increase the shared memory](https://stackoverflow.com/questions/30210362/how-to-increase-the-size-of-the-dev-shm-in-docker-container) (via --shm-size).

## Local
To create a working environment locally with conda, please run:
```
conda create --name nndet_venv python=3.8
conda activate nndet_venv
```

Now run the following commands to properly set it up:

```
git clone https://github.com/MIC-DKFZ/nnDetection.git
cd nnDetection

export CXX=$CONDA_PREFIX/bin/x86_64-conda_cos6-linux-gnu-c++
export CC=$CONDA_PREFIX/bin/x86_64-conda_cos6-linux-gnu-cc

conda install gxx_linux-64==9.3.0
conda install cuda -c nvidia/label/cuda-11.3.1
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

pip install -r requirements.txt  \
  && pip install hydra-core --upgrade --pre \
  && pip install git+https://github.com/mibaumgartner/pytorch_model_summary.git
FORCE_CUDA=1 pip install -v -e .
```

## Source

*Please note that nndetection requires Python 3.8+.*
*Please use PyTorch 1.X version for now and not 2.0*

1. Install CUDA (>10.1) and cudnn (make sure to select [compatible versions](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html)!)
2. [Optional] Depending on your GPU you might need to set `TORCH_CUDA_ARCH_LIST`, check [compute capabilities](https://developer.nvidia.com/cuda-gpus) here.
3. Install [torch](https://pytorch.org/) (make sure to match the pytorch and CUDA versions!) (requires pytorch >1.10+) and [torchvision](https://github.com/pytorch/vision)(make sure to match the versions!).
4. Clone nnDetection, `cd [path_to_repo]` and `pip install -e .`
5. Set environment variables (more info can be found below):
    - `det_data`: [required] Path to the source directory where all the data will be located
    - `det_models`: [required] Path to directory where all models will be saved
    - `OMP_NUM_THREADS=1` : [required] Needs to be set! Otherwise bad things will happen... Refer to batchgenerators documentation.
    - `det_num_threads`: [recommended] Number processes to use for augmentation (at least 6, default 12)
    - `det_verbose`: [optional] Can be used to deactivate progress bars (activated by default)
    - `MLFLOW_TRACKING_URI`: [optional] Specify the logging directory of mlflow. Refer to the [mlflow documentation](https://www.mlflow.org/docs/latest/tracking.html) for more information.

Note: nnDetection was developed on Linux => Windows is not supported.

<details close>
<summary>Test Installation</summary>
<br>
Run the following command in the terminal (!not! in pytorch root folder) to verify that the compilation of the C++/CUDA code was successfull:

```bash
python -c "import torch; import nndet._C; import nndet"
```

To test the whole installation please run the Toy Data set example.
</details>

<details close>
<summary>Maximising Training Speed</summary>
<br>
To get the best possible performance we recommend using CUDA 11.0+ with cuDNN 8.1.X+ and a (!)locally compiled version(!) of Pytorch 1.7+
</details>

# nnDetection
<div align="center">
    <img src=docs/source/nnDetectionFunctional.svg width="600px">
</div>

<details close>
<summary>nnDetection Module Overview</summary>
<br>
    <div align="center">
        <img src=docs/source/nnDetectionModule.svg width="600px">
    </div>

nnDetection uses multiple Registries to keep track of different modules and easily switch between them via the config files.

***Config Files***
nnDetection uses [Hydra](https://hydra.cc/) to dynamically configure and compose configurations.
The configuration files are located in `nndet.conf` and can be overwritten to customize the behavior of the pipeline.

***AUGMENTATION_REGISTRY***
The augmentation registry can be imported from `nndet.io.augmentation` and contains different augmentation configurations. Examples can be found in `nndet.io.augmentation.bg_aug`.

***DATALOADER_REGISTRY***
The dataloader registry contains different dataloader classes to customize the IO of nnDetection.
It can be imported from `nndet.io.datamodule` and examples can be found in `nndet.io.datamodule.bg_loader`.

***PLANNER_REGISTRY***
New plans can be registered via the planner registry which contains classes to define and perform different architecture and preprocessing schemes.
It can be imported from `nndet.planning.experiment` and examples can be found in `nndet.planning.experiment.v001`.

***MODULE_REGISTRY***
The module registry contains the core modules of nnDetection which inherits from the [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) Module.
It is the main module which is used for training and inference and contains all the necessary steps to build the final models.
It can be imported from `nndet.ptmodule` and examples can be found in `nndet.ptmodule.retinaunet`.

</details>

<details close>
<summary>nnDetection Functional Details</summary>
<br>
    <div align="center">
        <img src=docs/source/nnDetectionFunctionalDetails.svg width="600px">
    </div>
</details>

# Experiments & Data
The data sets used for our experiments are not hosted or maintained by us, please give credit to the authors of the data sets.
Some of the labels were corrected in data sets which we converted and can be downloaded (links can be found in the guides).
The `Experiments` section contains multiple guides which explain the preparation of the data sets via the provided scripts.

## Toy Data set
Running `nndet_example` will automatically generate an example data set with 3D squares and sqaures with holes which can be used to test the installation or experiment with prototype code (it is still necessary to run the other nndet commands to process/train/predict the data set).

```bash 
# create data to test installation/environment (10 train 10 test)
nndet_example

# create full data set for prototyping (1000 train 1000 test)
nndet_example --full [--num_processes]
```

The `full` problem is very easy and the final results should be near perfect.
After running the generation script follow the `Planning`, `Training` and `Inference` instructions below to construct the whole nnDetection pipeline.

## Guides
Work in progress

## Experiments
Besides the self-configuring method, nnDetection acts as a standard interface for many data sets.
We provide guides to prepare all data sets from our evaluation to the correct and make it easy to reproduce our resutls.
Furthermore, we provide pretrained models which can be used without investing large amounts of compute to rerun our experiments (see Section `Pretrained Models`).

<div align="center">

### Results
[nnDetection v0.1](./docs/results/nnDetectionV001.md)
</div>

<div align="center">

### Guides

| <!-- --> | <!-- --> | <!-- --> |
|:----------------------------------------------------------------:|:-------------------------------------------------:|:---------------------------------------:|
| [Task 003 Liver](./projects/Task001_Decathlon/README.md)          | [Task 011 Kits](./projects/Task011_Kits/README.md) | [Task 020 RibFrac](./projects/Task020_RibFrac/README.md)      |
| [Task 007 Pancreas](./projects/Task001_Decathlon/README.md)       | [Task 012 LIDC](./projects/Task012_LIDC/README.md) | [Task 021 ProstateX](./projects/Task021_ProstateX/README.md)  |
| [Task 008 Hepatic Vessel](./projects/Task001_Decathlon/README.md) | [Task 017 CADA](./projects/Task017_CADA/README.md) | [Task 025 LymphNodes](./projects/Task025_LymphNodes/README.md) |
| [Task 010 Colon](./projects/Task001_Decathlon/README.md)          | [Task 019 ADAM](./projects/Task019_ADAM/README.md) | [Task 016 Luna](./projects/Task016_Luna/README.md)         |

</div>

## Adding New Data sets
nnDetection relies on a standardized input format which is very similar to [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) and allows easy integration of new data sets.
More details about the format can be found below.

### Folders
All data sets should reside inside `Task[Number]_[Name]` folders inside the specified detection data folder (the path to this folder can be set via the `det_data` environment flag).
To avoid conflicts with our provided pretrained models we recommend to use task numbers starting from 100.
An overview is provided below ([Name] symbolise folders, `-` symbolise files, indents refer to substructures)

**Warning[!]: Please avoid any `.` inside file names/folder names/paths since it can influence how paths/names are splitted.**

```text
${det_data}
    [Task000_Example]
        - dataset.yaml # dataset.json works too
        [raw_splitted]
            [imagesTr]
                - case0000_0000.nii.gz # case0000 modality 0
                - case0000_0001.nii.gz # case0000 modality 1
                - case0001_0000.nii.gz # case0001 modality 0
                - case0000_0001.nii.gz # case0001 modality 1
            [labelsTr]
                - case0000.nii.gz # instance segmentation case0000
                - case0000.json # properties of case0000
                - case0001.nii.gz # instance segmentation case0001
                - case0001.json # properties of case0001
            [imagesTs] # optional, same structure as imagesTr
             ...
            [labelsTs] # optional, same structure as labelsTr
             ...
    [Task001_Example1]
        ...
```

### Data set Info
`dataset.yaml` or `dataset.json` provides general information about the data set:
Note: [Important] Classes and modalities start with index 0!
```yaml
task: Task000D3_Example

name: "Example" # [Optional]
dim: 3 # number of spatial dimensions of the data

# Note: need to use integer value which is defined below of target class!
target_class: 1 # [Optional] define class of interest for patient level evaluations
test_labels: True # manually splitted test set

labels: # classes of data set; need to start at 0
    "0": "Square"
    "1": "SquareHole"

modalities: # modalities of data set; need to start at 0
    "0": "CT"
```

### Image Format
nnDetection uses the same image format as nnU-Net.
Each case consists of at least one 3D nifty file with a single modality and are saved in the `images` folders.
If multiple modalities are available, each modality uses a separate file and the sequence number at the end of the name indicates the modality (these need to correspond to the numbers specified in the data set file and be consistent across the whole data set).

An example with two modalities could look like this:
```text
- case001_0000.nii.gz # Case ID: case001; Modality: 0
- case001_0001.nii.gz # Case ID: case001; Modality: 1

- case002_0000.nii.gz # Case ID: case002; Modality: 0
- case002_0001.nii.gz # Case ID: case002; Modality: 1
```

If multiple modalities are available, please check beforehand if they need to be registered and perform registration befor nnDetection preprocessing. nnDetection does (!)not(!) include automatic registration of multiple modalities.

### Label Format
Labels are encoded with two files per case: one nifty file which contains the instance segmentation and one json file which includes the "meta" information of each instance.
The nifty file should contain all annotated instances where each instance has a unique number and are in consecutive order (e.g. 0 ALWAYS refers to background, 1 refers to the first instance, 2 refers to the second instance ...)
`case[XXXX].json` label files need to provide the class of every instance in the segmentation. In this example the first isntance is assigned to class `0` and the second instance is assigned to class `1`:
```json
{
    "instances": {
        "1": 0,
        "2": 1
    }
}
```

Each label file needs a corresponding json file to define the classes.
We also wrote an [Detection Annotation Guide](https://www.notion.so/Object-Detection-Annotation-Guide-5318f090091c4e3db7e046a5990bd03c) which includes a dedicated section of the nnDetection format with additional visualizations :)

## Using nnDetection
The following paragrah provides an high level overview of the functionality of nnDetection and which commands are available.
A typical flow of commands would look like this:
```text
nndet_prep -> nndet_unpack -> nndet_train -> nndet_consolidate -> nndet_predict
```

Eachs of this commands is explained below and more detailt information can be obtained by running `nndet_[command] -h` in the terminal.

### Planning & Preprocessing
Before training the networks, nnDetection needs to preprocess and analyze the data.
The preprocessing stage normalizes and resamples the data while the analyzed properties are used to create a plan which will be used for configuring the training.
nnDetectionV0 requires a GPU with approximately the same amount of VRAM you are planning to use for training (we used a RTX2080TI; no monitor attached to it) to perform live estimation of the VRAM used by the network.
Future releases aim at improving this process...

```bash
nndet_prep [tasks] [-o / --overwrites] [-np / --num_processes] [-npp / --num_processes_preprocessing] [--full_check]

# Example
nndet_prep 000

# Script
# /scripts/preprocess.py - main()
```

`-o` option can be used to overwrite parameters for planning and preprocessing (refer to the config files to see all parameters). The number of processes used for cropping and analysis can be adjusted by using `-np` and the number of processes used for resampling can be set via `-npp`. The current values are fairly save if 64GB of RAM is available.
The `--full_check` will iterate over the data before starting any preprocessing and check correct formatting of the data and labels.
If any problems occur during preprocessing please run the full check to make sure that the format is correct.

After planning and preprocessing the resulting data folder structure should look like this:
```text
[Task000_Example]
    [raw_splitted]
    [raw_cropped] # only needed for different resampling strategies
        [imagesTr] # stores cropped image data; contains npz files
        [labelsTr] # stores labels
    [preprocessed]
        [analysis] # some plots to visualize properties of the underlying data set
        [properties] # sufficient for new plans
        [labelsTr] # labels in original format (original spacing)
        [labelsTs] # optional
        [Data identifier; e.g. D3V001_3d]
            [imagesTr] # preprocessed data
            [labelsTr] # preprocessed labels (resampled spacing)
        - {name of plan}.pkl e.g. D3V001_3d.pkl
```

Befor starting the training copy the data (Task Folder, data set info and preprocessed folder are needed) to a SSD (highly recommended) and unpack the image data with

```bash
nndet_unpack [path] [num_processes]

# Example (unpack example with 6 processes)
nndet_unpack ${det_data}/Task000D3_Example/preprocessed/D3V001_3d/imagesTr 6

# Script
# /scripts/utils.py - unpack()
```

### Training and Evaluation
After the planning and preprocessing stage is finished the training phase can be started.
The default setup of nnDetection is trained in a 5 fold cross-validation scheme.
First, check which plans were generated during planning by checking the preprocessing folder and look for the pickled plan files.
In most cases only the defaul plan will be generated (`D3V001_3d`) but there might be instances (e.g. Kits) where the low resolution plan will be generated too (`D3V001_3dlr1`).

```bash
nndet_train [task] [-o / --overwrites] [--sweep]

# Example (train default plan D3V001_3d and search best inference parameters)
nndet_train 000 --sweep

# Script
# /scripts/train.py - train()
```

Use `-o exp.fold=X` to overwrite the trained fold, this should be run for all folds `X = 0, 1, 2, 3, 4`!
The `--sweep` option tells nnDetection to look for the best hyparameters for inference by empirically evaluating them on the validation set.
Sweeping can also be performed later by running the following command:

```bash
nndet_sweep [task] [model] [fold]

# Example (sweep Task 000 of model RetinaUNetV001_D3V001_3d in fold 0)
nndet_sweep 000 RetinaUNetV001_D3V001_3d 0

# Script
# /experiments/train.py - sweep()
```

Evaluation can be invoked by the following command (requires access to the model and preprocessed data):
```bash
nndet_eval [task] [model] [fold] [--test] [--case] [--boxes] [--seg] [--instances] [--analyze_boxes]

# Example (evaluate and analyze box predictions of default model)
nndet_eval 000 RetinaUNetV001_D3V001_3d 0 --boxes --analyze_boxes

# Script
# /scripts/train.py - evaluate()

# Note: --test invokes evaluation of the test set
# Note: --seg, --instances are placeholders for future versions and not working yet
```

### Inference
After running all folds it is time to collect the models and creat a unified inference plan.
The following command will copy all the models and predictions from the folds. By adding the `sweep_` options, the empiricaly hyperparameter optimization across all folds can be started.
This will generate a unified plan for all models which will be used during inference.

```bash
nndet_consolidate [task] [model] [--overwrites] [--consolidate] [--num_folds] [--no_model] [--sweep_boxes] [--sweep_instances]

# Example
nndet_consolidate 000 RetinaUNetV001_D3V001_3d --sweep_boxes

# Script
# /scripts/consolidate.py - main()
```

For the final test set predictions simply select the best model according to the validation scores and run the prediction command below.
Data which is located in `raw_splitted/imagesTs` will be automatically preprocessed and predicted by running the following command:
```bash
nndet_predict [task] [model] [--fold] [--num_tta] [--no_preprocess] [--check] [-npp / --num_processes_preprocessing] [--force_args]

# Example
nndet_predict 000 RetinaUNetV001_D3V001_3d --fold -1

# Script
# /scripts/predict.py - main()
```

If a self-made test set was used, evaluation can be performed by invoking `nndet_eval` with `--test` as described above.

### Results
The final model directory will contain multiple subfolders with different information:
- `sweep`: contain information from the parameter sweeps and are only used for debugging purposes
- `sweep_predictions`: these contain prediction with additional ensembler state information which are used during the empirical parameter optimization. Since these save the model output in a fairly raw format they are bigger than the predictions seen during normal inference to avoid multiple model prediction runs during the parameter sweeps
- `[val/test]_predictions`: Contains the prediction of the validation/test set in the restored image space.
- `val_predictions_preprocessed`: This contains prediction in the preprocessed image space, i.e. the predictions from the resampled and cropped data. they are saved for debugging purposes.
- `[val/test]_results`: this folder contains the validation/test rsults computed by nnDetection. More information on the metrics can be found below.
- `val_results_preprocessed`: contains validation results inside the preprocessed image space are saved for debugging purposes
- `val_analysis[_preprocessed]` *experimental*: provide additional analysis information of the predictions. This feature is marked as expeirmental since it uses a simplified matching algorithm and should only be used to gain an intuition of potential improvements.

The following section contains some additional information regarding the metrics which are computed by nnDetection. They can be found in `[val/test]_results/results_boxes.json`:
- `AP_IoU_0.10_MaxDet_100`: is the main metric used for the evaluation in our paper. It is evaluated at an IoU threshold of `0.1` and `100` predictions per image. Note that this is a hard limit and if images contain much more instances this leads to wrong results.
- `mAP_IoU_0.10_0.50_0.05_MaxDet_100`: Is the typically found COCO mAP metric evaluated at multiple IoU values. *The IoU thresholds are different from those of the COCO evaluation to account for the generally lower IoU in 3D data*
- `[num]_AP_IoU_0.10_MaxDet_100`: AP metric computed per class
- `FROC_score_IoU_0.10` FROC score with default FPPI (1/8, 1/4, 1/2, 1, 2, 4, 8). Note (in contrast to the AP implementation): the multi-class case does not compute the metric per class but puts all predictions/gt into a single large pool (similar to AP_pool from https://arxiv.org/abs/2102.01066) and thus inter class calibration is important here. In most cases simply averaging the `[num]_FROC` scores manually to assign the same weight to each class should be prefered.
- case evaluation *experimental*: It is possible to run case evaluations with nnDetection but this is still experimental and undergoing additional testing and might be changed in the future.

## nnU-Net for Detection
Besides nnDetection we also include the scripts to prepare and evaluate nnU-Net in the context of obejct detection.
Both frameworks need to be configured correctly before running the scripts to assure correctness.
After preparing the data set in the nnDetection format (which is a superset of the nnU-Net format) it is possible to export it to nnU-Net via `scripts/nnunet/nnunet_export.py`. Since nnU-Net needs task ids without any additions it may be necessary to overwrite the task name via the `-nt` option for some dataets (e.g. `Task019FG_ADAM` needs to be renamed to `Task019_ADAM`).
Follow the usual nnU-Net preprocessing and training pipeline to generate the needed models.
Use the `--npz` option during training to save the predicted probabilities which are needed to generate the detection results.
After determining the best ensemble configuration from nnU-Net pass all paths to `scripts/nnunet/nnunet_export.py` which will ensemble and postprocess the predictions for object detection.
Per default the `nnU-Net Plus` scheme will be used which incorporates the empirical parameter optimization step.
Use `--simple` flag to switch to the `nnU-Net` basic configuration.

## Pretrained models
**Coming Soon**

# FAQ & Common Issues
<details close>
<summary>Installation & Initial Setup Errors</summary>
<br>

1. Error: Undefined CUDA symbols when importing `nndet._C` or other import related Errors from `nndet._C` or CUDA related ARCH errors
nnDetection includes additional CUDA code which needs to compiled upon installation and thus requires correct configuration of the CUDA dependencies.
Please double check CUDA version of your PC, pytorch, torchvision and nnDetection build.
This can be done by running `nndet_env` if the installation succeeded  or by running `python scripts/utils.py`.
An example output of the command is shown below:

```bash
----- PyTorch Information -----
PyTorch Version: 1.11.0+cu113
PyTorch Debug: False
PyTorch CUDA: 11.3
PyTorch Backend cudnn: 8200
PyTorch CUDA Arch List: ['sm_37', 'sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86']
PyTorch Current Device Capability: (7, 5)
PyTorch CUDA available: True

----- System Information -----
System NVCC: nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Sun_Aug_15_21:14:11_PDT_2021
Cuda compilation tools, release 11.4, V11.4.120
Build cuda_11.4.r11.4/compiler.30300941_0

System Arch List: None
System OMP_NUM_THREADS: 1
System CUDA_HOME is None: True
System CPU Count: 8
Python Version: 3.8.11 (default, Aug  3 2021, 15:09:35)
[GCC 7.5.0]

----- nnDetection Information -----
det_num_threads 6
det_data is set True
det_models is set True
```
Things to look out for:

Make sure that the versions of PyTorch CUDA and NVCC CUDA match (minor version mismatch as in this case, will work without error but could potentially introduce bugs.)

`OMP_NUM_THREADS` should always be set to 1 and `det_num_threads` should always be lower or equal `Systemm CPU Count`.

2. Error persists even after fixing the environment
Make sure to delete the `build` folder before rerunning the installation since it won't recompile the code otherwise.

3. Error: No kernel image is available for execution

You are probably executing the build on a machine with a GPU architecture which was not present/set during the build.

Please check [link](https://developer.nvidia.com/cuda-gpus) to find the correct SM architecture and set `TORCH_CUDA_ARCH_LIST`
approriately (e.g. check Dockefile for example).
As before make sure to delete the `build` folder when rerunning the installation process.

3. Please open an Issue and provide your environment as obtained by `nndet_env`.

</details>

<details close>
<summary>Training doesn't start or is stuck</summary>
<br>

1. Please run `nndet_env` and make sure `OMP_NUM_THREADS` is set to 1. No other values are supported here. To increase the number of workers used for IO and augmentation adjust `nndet_num_threads`.

2. Try running the training without multiprocessing as a sanity check: `nndet_train XXX -o augment_cfg.multiprocessing=False`. Don't use this for the full training, this is just one step of the debugging process.

3. Please open an Issue and provide your environment as obtained by `nndet_env` and report if the training without multiprocessing started correctly.

</details>

<details close>
<summary>(Slow) Training Speed</summary>
<br>

The training time of nnDetection should be roughly equal for most data sets: 2 days (1-2 hours per epoch) with mixed precision 3d speed up and 4 days without (this number refers to RTX 2080TI, newer GPUs can be significantly faster, on high end configuration training takes 1 day). It is highly recommended to use GPUs with Tensor Cores to enable fast mixed precision training for reasonable turnaround times. There can be several reasons for slow training:

1) PyTorch < 1.9 did not provide training speedup for mixed-precision 3d convs in their pip installable version and it was necessary to build it from source. (the docker build of nnDetection also provides the speedup). Newer versions like 1.10 and 1.11 provide the mixed precision speedup in their pip version (only tested with CUDA 11.X).


2) There is a bottleneck in the setup. This can be identified as follows:
    1) Check the GPU Util -> it should be high for most of the time if it isn't, there is either a CPU or IO bottleneck. If it is high it is the missing pytorch speed up.
    2) Check CPU util: if the CPU util is high (and the GPU util isn't) more cpu threads are needed for augmentation (can be adjusted via det_num_threads and depends on your CPU).
If GPU and CPU util are low, it is an IO bottleneck, it is quite hard to do anything about this (a typical SSD with ~500mb/s read speed ran fine for my experiments). If the CPU util is maxed out it is an CPU bottleneck: Adjust det_num_threads (similar to num workers in the normal pytorch dataloaders) for the available CPU resources (set this as high as possible but not more than available CPU threads) otherwise. Increasing the number of workers will increase the required RAM consumption -> make sure not to run out of memory there otherwise the training will be extreeemly slow and the workstation might crash.

Example for det_num_threads:
- CPUs with less cores but high clock speed: Needs a lower det_num_threads value. On an Intel i7 9700 (non k) det_num_threads=6 reaches 90+ % GPU usage.
- CPUs with many cores but lower clock speed: Needs a high det_num_threads value. In cluster environments det_num_threads=12 reaches ~80+% GPU usage.


</details>

<details close>
<summary>GPU requirements</summary>
<br>
nnDetection v0.1 was developed for GPUs with at least 11GB of VRAM (e.g. RTX2080TI, TITAN RTX).
All of our experiments were conducted with a RTX2080TI.
While the memory can be adjusted by manipulating the correct setting we recommend using the default values for now.
Future releases will refactor the planning stage to improve the VRAM estimation and add support for different memory budgets.
</details>


<details close>
<summary>Training with bounding boxes</summary>
<br>
The first release of nnDetection focuses on 3d medical images and Retina U-Net.
As a consequence training (specifically planning and augmentation) requrie segmentation annotations.
In many cases this limitation can be circumvented by converting the bounding boxes into segmentations.
</details>

<details close>
<summary>Mask RCNN and 2D Data sets</summary>
<br>
2D data sets and Mask R-CNN are not supported in the first release.
We hope to provide these in the future.
</details>

<details close>
<summary>Multi GPU Training</summary>
<br>
Multi GPU training is not officially supported yet.
Inference and the metric computation are not properly designed to support these usecases!
</details>

<details close>
<summary>Prebuild package</summary>
<br>
We are planning to provide prebuild wheels in the future but no prebuild wheels are available right now.
Please use the provided Dockerfile or the installation instructions to run nnDetection.
</details>

# Acknowledgements
nnDetection combines the information from multiple open source repositores we wish to acknoledge for their awesome work, please check them out!

## [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
nnU-Net is self-configuring method for semantic segmentation and many steps of nnDetection follow in the footsteps of nnU-Net.

## [Medical Detection Toolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
The Medical Detection Toolkit introduced the first codebase for 3D Object Detection and multiple tricks were transferred to nnDetection to assure optimal configuration for medical object detection.

## [Torchvision](https://github.com/pytorch/vision)
nnDetection tried to follow the interfaces of torchvision to make it easy to understand for everyone coming from the 2D (and video) detection scene. As a result we used based our implementations of some of the core modules of the torchvision implementation.

## Funding
Part of this work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – 410981386 and the Helmholtz Imaging Platform (HIP), a platform of the Helmholtz Incubator on Information and Data Science.
