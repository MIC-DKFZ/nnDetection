.. _user_guide-label:

User Guide
==========

TODOs
=====
- application limited to 3D
- Pointer to Projects and Plugins
- Improvements
   - select best model for evaluation
   - run inference on CPU (inference_kwargs.device=cpu)
   - run segmentation of RetinaU-Net

- Training Time / Training Speed / Benchmark?

Preparing Data Sets
*******************

Toy Data Set
------------

Running `nndet_example` will automatically generate an example data set with 3D squares and sqaures with holes which can be used to test the installation or experiment with prototype code (it is still necessary to run the other nndet commands to process/train/predict the data set).

.. code-block:: bash

    # create data to test installation/environment (10 train 10 test)
    nndet_example

    # create full data set for prototyping (1000 train 1000 test)
    nndet_example --full [--num_processes]

The `full` problem is very easy and the final results should be near perfect (even after very short training).
After running the generation script follow the `Planning`, `Training` and `Inference` instructions below to construct the whole nnDetection pipeline.

.. note::

    The toy data set does not require the full training, it is recommended to use the `toy` config for training it.
    The training command for the toy data set is provided below:

    .. code-block::
        
        # this is a very short training schedule specifically to speed up the toy data set and should not be used in any other circumstance!
        nndet_train 000 -o train=toy --sweep

Experiments
-----------

.. note::

    The data sets used for our experiments are not hosted or maintained by us, please give credit to the authors of the data sets.
    Some of the labels were corrected in data sets which we converted and can be downloaded (links can be found in the guides).
    The `Experiments` section contains multiple guides which explain the preparation of the data sets via the provided scripts.

Besides the self-configuring method, nnDetection acts as a standard interface for many data sets.
We provide guides to prepare all data sets from our evaluation to the correct and make it easy to reproduce our resutls.
Furthermore, we provide pretrained models which can be used without investing large amounts of compute to rerun our experiments (see Section `Pretrained Models`).

* Task 003 Liver: nndetection/projects/Task001_Decathlon
* Task 007 Pancreas: nndetection/projects/Task001_Decathlon
* Task 008 HepaticVessel: nndetection/projects/Task001_Decathlon
* Task 010 Colon: nndetection/projects/Task001_Decathlon
* Task 017 CADA: nndetection/projects/Task017_CADA
* Task 020 RibFrac: nndetection/projects/Task020_RibFrac
* Task 016 Luna: nndetection/projects/Task016_Luna

# TODO: finish V2 datasets
Additional data sets from nnDetection V2 (recommended):

* Task 035 KiTS21: nndetection/projects/Task_035_KiTS21
* Task 036 PICAI: nndetection/projects/Task_036_PICAI
* Task 037 ADAM TOF A: nndetection/projects/Task_037_ADAM_TOF_A
* Task 038 LIDC: nndetection/projects/Task017_CADA

Additional data sets from nnDetection V1:

* Task 011 Kits: nndetection/projects/Task011_Kits
* Task 019 ADAM: nndetection/projects/Task019_ADAM
* Task 021 ProstateX: nndetection/projects/Task021_ProstateX
* Task 012 LIDC: nndetection/projects/Task012_LIDC
* Task 025 LymphNodes: nndetection/projects/Task025_LymphNodes

Check the Projects tab for additional projects and results of nnDetection.

Adding New Data Sets
--------------------
nnDetection relies on a standardized input format which is very similar to `nnU-Net <https://github.com/MIC-DKFZ/nnUNet>`_ and allows easy integration of new data sets.
More details about the format can be found below.

Folders
~~~~~~~
All data sets should reside inside `Task[Number]_[Name]` folders inside the specified detection data folder (the path to this folder can be set via the `det_data` environment flag).
To avoid conflicts with our provided pretrained models we recommend to use task numbers starting from 100.
An overview is provided below ([Name] symbolise folders, `-` symbolise files, indents refer to substructures)
Note: Please avoid `.` inside file names since it can influence how paths/names are splitted.

.. code-block:: text

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


Data Set Info
~~~~~~~~~~~~~
`dataset.yaml` or `dataset.json` provides general information about the data set:
Note: [Important] Classes and modalities start with index 0!

.. code-block:: yaml

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


Image Format
~~~~~~~~~~~~

nnDetection uses the same image format as nnU-Net.
Each case consists of at least one 3D nifty file with a single modality and are saved in the `images` folders.
If multiple modalities are available, each modality uses a separate file and the sequence number at the end of the name indicates the modality (these need to correspond to the numbers specified in the data set file and be consistent across the whole data set).

An example with two modalities could look like this:

.. code-block:: text

    - case001_0000.nii.gz # Case ID: case001; Modality: 0
    - case001_0001.nii.gz # Case ID: case001; Modality: 1

    - case002_0000.nii.gz # Case ID: case002; Modality: 0
    - case002_0001.nii.gz # Case ID: case002; Modality: 1

If multiple modalities are available, please check beforehand if they need to be registered and perform registration befor nnDetection preprocessing. nnDetection does (!)not(!) include automatic registration of multiple modalities.

Label Format
~~~~~~~~~~~~

Labels are encoded with two files per case: one nifty file which contains the instance segmentation and one json file which includes the "meta" information of each instance.
The nifty file should contain all annotated instances where each instance has a unique number and are in consecutive order (e.g. 0 ALWAYS refers to background, 1 refers to the first instance, 2 refers to the second instance ...)
`case[XXXX].json` label files need to provide the class of every instance in the segmentation. In this example the first isntance is assigned to class `0` and the second instance is assigned to class `1`:

.. code-block:: json

    {
        "instances": {
            "1": 0,
            "2": 1
        }
    }


Each label file needs a corresponding json file to define the classes.

# TODO: all images need to have the same number of modalities
# TODO: images (all modalities) and corresponding label need to have the same size (number of pixels)
# TODO: modalities need to be registered (bias field correction?)


Using nnDetection
*****************

The following paragrah provides an high level overview of the functionality of nnDetection and which commands are available.
A typical flow of commands would look like this:

.. note::

    nndet_prep -> nndet_unpack -> nndet_cv_split -> nndet_train -> nndet_consolidate -> nndet_predict

Eachs of this commands is explained below and more detailt information can be obtained by running `nndet_[command] -h` in the terminal.


Planning & Preprocessing
------------------------

Before training the networks, nnDetection needs to preprocess and analyze the data.
The preprocessing stage normalizes and resamples the data while the analyzed properties are used to create a plan which will be used for configuring the training.
nnDetectionV0 requires a GPU with approximately the same amount of VRAM you are planning to use for training (we used a RTX2080TI; no monitor attached to it) to perform live estimation of the VRAM used by the network.
(Future releases aim at improving this process...)

.. code-block:: bash

    nndet_prep [tasks] [-o / --overwrites] [-np / --num_processes] [-npp / --num_processes_preprocessing] [--full_check]

    # Example
    nndet_prep 000

    # Script
    # /scripts/preprocess.py - main()

`-o` option can be used to overwrite parameters for planning and preprocessing (refer to the config files to see all parameters). The number of processes used for cropping and analysis can be adjusted by using `-np` and the number of processes used for resampling can be set via `-npp`. The current values are fairly save if 64GB of RAM is available.
The `--full_check` will iterate over the data before starting any preprocessing and check correct formatting of the data and labels.
If any problems occur during preprocessing please run the full check to make sure that the format is correct.

After planning and preprocessing the resulting data folder structure should look like this:

.. code-block:: text

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
            - {name of plan}.pkl # e.g. D3V001_3d.pkl

Befor starting a training copy the data (Task Folder, data set info and preprocessed folder are needed) to a SSD (highly recommended) and unpack the image data with

.. code-block:: bash

    nndet_unpack [path] [num_processes]

    # Example (unpack example with 6 processes)
    nndet_unpack ${det_data}/Task000D3_Example/preprocessed/D3V001_3d/imagesTr 6

    # Script
    # /scripts/utils.py - unpack()

Finally, it is necessary to create a data set split for the underlying data by running the following command:

.. code-block:: bash

    nndet_cv_split [task] [--num_folds] [--with_patients]

    # Example
    nndet_cv_split Task000D3_Example

    # Script
    # /scripts/utils.py - create_cv_split()

For more advanced options in the split functionality (like the '--with_patients' option) plese refer to the documentation of the function or create your own splitting function e.g. for hierarchical data.


Training and Evaluation
-----------------------

After the planning and preprocessing stage is finished the training phase can be started.
The default setup of nnDetection is trained in a 5 fold cross-validation scheme.
First, check which plans were generated during planning by checking the preprocessing folder and look for the pickled plan files.
In most cases only the defaul plan will be generated (`D3V001_3d`) but there might be instances (e.g. Kits) where the low resolution plan will be generated too (`D3V001_3dlr1`).

.. code-block:: bash

    nndet_train [task] [config_name] [fold] [-o / --overwrites] [--sweep] [--continue_training] [--log_net] [--log_aug]

    # Example (train default plan D3V001_3d and search best inference parameters)
    nndet_train 000 toy 0 --sweep

    # Script
    # /scripts/train.py - train()

`nndet_train` needs to be run for every fold separately, by default this means running it 5 times with `fold` varying between 0 and 4 (inclusive).
`--continue_training` can be activated to continue training from the last saved checkpoint.
The training time can vary between ~1 day (A100) to ~2 days (RTX2080TI) *per fold* (with correct mixed precision acceleration of 3D convolutions and no other bottlenecks, see FAQ section for common bottlenecks and how to diagnose them).
The `--sweep` option tells nnDetection to look for the best hyparameters for inference by empirically evaluating them on the validation set.
Sweeping can also be performed later by running the following command:

.. code-block:: bash

    nndet_sweep [task] [model] [fold]

    # Example (sweep Task 000 of model RetinaUNetV001_D3V001_3d in fold 0)
    nndet_sweep 000 RetinaUNetV001_D3V001_3d 0

    # Script
    # /experiments/train.py - sweep()


Evaluation can be invoked by the following command (requires access to the model and preprocessed data):

.. code-block:: bash

    nndet_eval [task] [model] [fold] [--test] [--case] [--boxes] [--analyze_boxes]

    # Example (evaluate and analyze box predictions of default model)
    nndet_eval 000 RetinaUNetV001_D3V001_3d 0 --boxes --analyze_boxes

    # Script
    # /scripts/train.py - evaluate()

    # Note: --test invokes evaluation of the test set

Inference
---------

After running all folds it is time to collect the models and creat a unified inference plan.
The following command will copy all the models and predictions from the folds. By adding the `sweep_` options, the empiricaly hyperparameter optimization across all folds can be started.
This will generate a unified plan for all models which will be used during inference.

.. code-block:: bash

    nndet_consolidate [task] [model] [--overwrites] [--consolidate] [--num_folds] [--no_model] [--sweep]

    # Example
    nndet_consolidate 000 RetinaUNetV001_D3V001_3d --sweep

    # Script
    # /scripts/consolidate.py - main()

For the final test set predictions simply select the best model according to the validation scores and run the prediction command below.
Data which is located in `raw_splitted/imagesTs` will be automatically preprocessed and predicted by running the following command:

.. code-block:: bash

    nndet_predict [task] [model] [--fold] [--num_tta] [--no_preprocess] [--check] [-npp / --num_processes_preprocessing] [--force_args]

    # Example
    nndet_predict 000 RetinaUNetV001_D3V001_3d -1

    # Script
    # /scripts/predict.py - main()

If a self-made test set was used, evaluation can be performed by invoking `nndet_eval` with `--test` as described above.

# TODO: udpate predict command to predict2
# TODO: pretrained models
# TODO: continue training
# TODO: move nnU-Net for detection into a separate project page

Results
-------

The final model directory will contain multiple subfolders with different information:

* `sweep`: contain information from the parameter sweeps and are only used for debugging purposes
* `sweep_predictions`: these contain prediction with additional ensembler state information which are used during the empirical parameter optimization. Since these save the model output in a fairly raw format they are bigger than the predictions seen during normal inference to avoid multiple model prediction runs during the parameter sweeps
* `[val/test]_predictions`: Contains the prediction of the validation/test set in the restored image space.
* `val_predictions_preprocessed`: This contains prediction in the preprocessed image space, i.e. the predictions from the resampled and cropped data. they are saved for debugging purposes.
* `[val/test]_results`: this folder contains the validation/test rsults computed by nnDetection. More information on the metrics can be found below.
* `val_results_preprocessed`: contains validation results inside the preprocessed image space are saved for debugging purposes

Evaluation
----------

The following section contains some additional information regarding the metrics which are computed by nnDetection. They can be found in `[val/test]_results/results_boxes.json`:

Most metric have an IoU attached to their evaluation, the value is usually part of the naming, e.g. `IoU_0.10` indicates an IoU threshold of `0.10`.
Some metrics are computed per class and thus per class values are also included for completeness e.g. `YY_AP_IoU_0.10` represents class `YY`.
Finally, some metrics are extended with additional analysis functions e.g. computed for a certain volume range which are indicated by additional letters (by default zVF where z various between categories, exact values can be found in the `results_curves` file)

* `AP_IoU_0.XX`: is the main metric used for the evaluation in our paper. Per default, the number of detections per image per class are limited to `400` but can be adjusted via `nndet_eval_max_detections_image_based`. 
* `mAP_IoU_0.XX_0.XX_0.XX`: Is the typically found COCO mAP metric evaluated at multiple IoU values. *The IoU thresholds are different from those of the COCO evaluation to account for the generally lower IoU in 3D data*.
* `FROCwp_IoU_0.10`: FROC computed at default FPPI values of (1/8, 1/4, 1/2, 1, 2, 4, 8), sensitivty at FPPI value is determined by last working point (not interpolated). This implementation pools all of the predictions and is *not* computed per class.
* `mc_FROCwp_IoU_0.10`: compute FROC per class and average across classes. This one should be used in most cases in multi class scenarios to stratify for the number of objects inside the classes.

.. warning::

    nnDetection provides some additional analysis files (located in the analysis folders) which are purely for qualitative analysis purposes and should never be used for quantitative evaluation!
    Since they are not part of the official functionality we do not provide extensive documentation nor support for this.

# TODO: visualisation of results
# TODO: format of predictions


Advanced Use Cases
******************

An advanced use case might require some minor coding which is not covered by the default functionality of nnDetection.
Nevertheless, some cases can occur frequently and are thus covered here. 

Detection Zoo
-------------

+--------------------------+------------------------++------------------------------------------------------------------------------+
| **Models**               | **Internal Inputs**    || **Command**                                                                  |
+--------------------------+------------------------++------------------------------------------------------------------------------+
+--------------------------+------------------------++------------------------------------------------------------------------------+
|| Retina U-Net V001       | BB + SS                || nndet_train [task] retinaunet_v001 [fold]                                    |
+--------------------------+------------------------++------------------------------------------------------------------------------+
+--------------------------+------------------------++------------------------------------------------------------------------------+
|| RetinaNet V002 HNM      | BB                     || nndet_train [task] retinaunet_hnm_v002 [fold] -o model=RetinaNetHNMV002      |
+--------------------------+------------------------++------------------------------------------------------------------------------+
|| RetinaNet V002 Focal    | BB                     || nndet_train [task] retinaunet_focal_v002 [fold] -o model=RetinaNetFocalV002  |
+--------------------------+------------------------++------------------------------------------------------------------------------+
|| Faster RCNN V002        | BB                     ||                                                                              |
+--------------------------+------------------------++------------------------------------------------------------------------------+
|| Retina U-Net V002 HNM   | BB (+ SS)              || nndet_train [task] retinaunet_hnm_v002 [fold]                                |
+--------------------------+------------------------++------------------------------------------------------------------------------+
|| Retina U-Net V002 Focal | BB (+ SS)              || nndet_train [task] retinaunet_focal_v002 [fold]                              |
+--------------------------+------------------------++------------------------------------------------------------------------------+
|| Box Mask RCNN V002      | BB + BI                ||                                                                              |
+--------------------------+------------------------++------------------------------------------------------------------------------+
|| Box Mask U-RCNN V002    | BB + BI (+ SS)         ||                                                                              |
+--------------------------+------------------------++------------------------------------------------------------------------------+
+--------------------------+------------------------++------------------------------------------------------------------------------+

Legend: BB = Bounding Boxes, BI = Binary Mask, SS = Semantic Segmentation (dervied from instance segmentation mask)


Evaluation Framework
--------------------

If the labels and predictions are present in the nnDetection format, evaluation of box predictions is as simple as:

.. code-block:: python

    from nndet.eval.registry import evaluate_box_dir

    classes = [YOUR CLASSES HERE]
    pred_dir = [YOUR PATH HERE]
    gt_dir = [YOUR PATH HERE]

    results = evaluate_box_dir(
        classes=classes,
        pred_dir=pred_dir,
        gt_dir=gt_dir,
    )

.. note::
    `nndet_eval_with_folders` provides a direct entrypoint to the above functionality.

The nnDetection format expects the ground truth labels to be saved in `npz` files with keys `boxes` and `classes`.
Predictions should be located in pkl files with keys `pred_boxes`, `pred_labels` and `pred_scores`.
All boxes need to be in the same coordinate system and follow the `ax0_min, ax1_min, ax0_max, ax1_max, ax2_min, ax2_max` format (ax denotes arbitrary axes).

Custom evaluation scripts can be esily created by passing the predictions and ground truth boxes to the evaluator.

.. code-block:: python

    from nndet.eval.det import BoxEvaluator

    classes = [YOUR CLASSES HERE]
    similarity_fn = [YOUR SIMILARITY FUNCTION e.g. box_iou_np]
    case_ids = [YOUR CASES]

    evaluator = BoxEvaluator.create(
        classes=classes,
        fast=False,
        verbose=True,
        similarity_fn=similarity_fn,
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
    return evaluator.finish_online_evaluation()


Custom Applications
-------------------

# TODO: custom split
# TODO: custom network -> refer to developer guide
# TODO: Running unit tests



FAQ & Common Issues
*******************

Installation & Initial Setup Errors
-----------------------------------

Error: Undefined CUDA symbols when importing `nndet._C` or other import related Errors from `nndet._C` or CUDA related ARCH errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

nnDetection includes additional CUDA code which needs to compiled upon installation and thus requires correct configuration of the CUDA dependencies.
Please double check CUDA version of your PC, pytorch, torchvision and nnDetection build.
This can be done by running `nndet_env` if the installation succeeded  or by running `python scripts/utils.py`.
An example output of the command is shown below:

.. note:: 

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

Things to look out for:

Make sure that the versions of PyTorch CUDA and NVCC CUDA match (minor version mismatch as in this case, will work without error but could potentially introduce bugs.)
`OMP_NUM_THREADS` should always be set to 1 and `det_num_threads` should always be lower or equal `Systemm CPU Count`.
Make sure to delete the `build` folder before rerunning the installation since it won't recompile the code otherwise.

Error: No kernel image is available for execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You are probably executing the build on a machine with a GPU architecture which was not present/set during the build.

Please check [link](https://developer.nvidia.com/cuda-gpus) to find the correct SM architecture and set `TORCH_CUDA_ARCH_LIST`
approriately (e.g. check Dockefile for example).
As before make sure to delete the `build` folder when rerunning the installation process.

Error still persists
~~~~~~~~~~~~~~~~~~~~

Please open an Issue and provide your environment as obtained by `nndet_env`.


Training doesn't start or is stuck
----------------------------------

* Please run `nndet_env` and make sure `OMP_NUM_THREADS` is set to 1. No other values are supported here. To increase the number of workers used for IO and augmentation adjust `nndet_num_threads`.
* Try running the training without multiprocessing as a sanity check: `nndet_train XXX -o augment_cfg.multiprocessing=False`. Don't use this for the full training, this is just one step of the debugging process.
* Please open an Issue and provide your environment as obtained by `nndet_env` and report if the training without multiprocessing started correctly.

GPU requirements
----------------

nnDetection v0.1 was developed for GPUs with at least 11GB of VRAM (e.g. RTX2080TI, TITAN RTX).
All of our experiments were conducted with a RTX2080TI.
While the memory can be adjusted by manipulating the correct setting we recommend using the default values for now.
Future releases will refactor the planning stage to improve the VRAM estimation and add support for different memory budgets.

Training with bounding boxes
----------------------------

The first release of nnDetection focuses on 3d medical images and Retina U-Net.
As a consequence training (specifically planning and augmentation) requrie segmentation annotations.
In many cases this limitation can be circumvented by converting the bounding boxes into segmentations.

Support for 2D Data Sets
------------------------
2D data sets are not supported since they are already amaying external repositories for these detection tasks, e.g. https://github.com/MIC-DKFZ/generalized_yolov5.

Multi GPU Training
------------------
Multi GPU training is not officially supported yet.
Inference and the metric computation are not properly designed to support these usecases!
