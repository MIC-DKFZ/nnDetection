User Guide
==========

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

:doc:`Task 003 Liver <./projects/Task001_Decathlon/README.md>`
:doc:`Task 007 Pancreas <./projects/Task001_Decathlon/README.md>`
:doc:`Task 008 Hepatic Vessel <./projects/Task001_Decathlon/README.md>`
:doc:`Task 010 Colon <./projects/Task001_Decathlon/README.md>`
:doc:`Task 017 CADA <./projects/Task017_CADA/README.md>`
:doc:`Task 020 RibFrac <./projects/Task020_RibFrac/README.md>`
:doc:`Task 016 Luna <./projects/Task016_Luna/README.md>`

# TODO: finish V2 datasets
Additional data sets from nnDetection V2 (recommended):
:doc:`Task 035 KiTS21 <./projects/Task_035_KiTS21/README.md>`
:doc:`Task 036 PICAI <./projects/Task_036_PICAI/README.md>`
:doc:`Task 037 ADAM TOF A <./projects/Task_037_ADAM_TOF_A/README.md>`
:doc:`Task 038 LIDC <./projects/Task017_CADA/README.md>`

Additional data sets from nnDetection V1:
:doc:`Task 011 Kits <./projects/Task011_Kits/README.md>`
:doc:`Task 019 ADAM <./projects/Task019_ADAM/README.md>`
:doc:`Task 021 ProstateX <./projects/Task021_ProstateX/README.md>`
:doc:`Task 012 LIDC <./projects/Task012_LIDC/README.md>`
:doc:`Task 025 LymphNodes <./projects/Task025_LymphNodes/README.md>`

Check the Projects tab for additional projects and results of nnDetection.

Adding New Data sets
--------------------
nnDetection relies on a standardized input format which is very similar to :doc:`nnU-Net <https://github.com/MIC-DKFZ/nnUNet>` and allows easy integration of new data sets.
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

# TODOs
# - all images need to have the same number of modalities
# - images (all modalities) and corresponding label need to have the same size (number of pixels)
# - modalities need to be registered (bias field correction?)


Using nnDetection
*****************

The following paragrah provides an high level overview of the functionality of nnDetection and which commands are available.
A typical flow of commands would look like this:

.. note::

    nndet_prep -> nndet_unpack -> nndet_train -> nndet_consolidate -> nndet_predict

Eachs of this commands is explained below and more detailt information can be obtained by running `nndet_[command] -h` in the terminal.


Planning & Preprocessing
------------------------

Before training the networks, nnDetection needs to preprocess and analyze the data.
The preprocessing stage normalizes and resamples the data while the analyzed properties are used to create a plan which will be used for configuring the training.
nnDetectionV0 requires a GPU with approximately the same amount of VRAM you are planning to use for training (we used a RTX2080TI; no monitor attached to it) to perform live estimation of the VRAM used by the network.
Future releases aim at improving this process...

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
        - {name of plan}.pkl e.g. D3V001_3d.pkl

Befor starting a training copy the data (Task Folder, data set info and preprocessed folder are needed) to a SSD (highly recommended) and unpack the image data with

.. code-block:: bash

    nndet_unpack [path] [num_processes]

    # Example (unpack example with 6 processes)
    nndet_unpack ${det_data}/Task000D3_Example/preprocessed/D3V001_3d/imagesTr 6

    # Script
    # /scripts/utils.py - unpack()

Finally, it is necessary to create a data set split for the underlying data by running the following command:

.. code-block:: bash

    nndet_cv_split [task] [] [--num_folds] [--with_patients]

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

    nndet_train [task] [-o / --overwrites] [--sweep]

    # Example (train default plan D3V001_3d and search best inference parameters)
    nndet_train 000 --sweep

    # Script
    # /scripts/train.py - train()


Use `-o exp.fold=X` to overwrite the trained fold, this should be run for all folds `X = 0, 1, 2, 3, 4`!
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

    nndet_eval [task] [model] [fold] [--test] [--case] [--boxes] [--seg] [--instances] [--analyze_boxes]

    # Example (evaluate and analyze box predictions of default model)
    nndet_eval 000 RetinaUNetV001_D3V001_3d 0 --boxes --analyze_boxes

    # Script
    # /scripts/train.py - evaluate()

    # Note: --test invokes evaluation of the test set
    # Note: --seg, --instances are placeholders for future versions and not working yet

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

# TODOs
# - pretrained models
# - continue training
# - move nnU-Net for detection into a separate project page

Results
-------

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

.. warning::

    nnDetection provides some additional analysis files (located in the analysis folders) which are purely for qualitative analysis purposes and should never be used for quantitative evaluation!


# TODOs
# - update FROC describtions
# - Evaluation and Analysis

Advanced Use Cases
******************

Detection Zoo
-------------

+--------------------------+------------------------+---------------------------+-------------------------++------------------------------------------------------------------------------+
| **Models**               | **Inputs**             | **Outputs**               | **Config**              || **Command**                                                                  |
+--------------------------+------------------------+---------------------------+-------------------------++------------------------------------------------------------------------------+
+--------------------------+------------------------+---------------------------+-------------------------++------------------------------------------------------------------------------+
|| Retina U-Net V001       | BB + SS                | BB                        | retinaunet_v001         || train=retinaunet_v001                                                        |
+--------------------------+------------------------+---------------------------+-------------------------++------------------------------------------------------------------------------+
+--------------------------+------------------------+---------------------------+-------------------------++------------------------------------------------------------------------------+
|| RetinaNet V002          | BB                     | BB                        | retinaunet_v002         || train=retinaunet_v002                                                        |
+--------------------------+------------------------+---------------------------+-------------------------++------------------------------------------------------------------------------+
|| Faster RCNN V002        | BB                     | BB                        |                         ||                                                                              |
+--------------------------+------------------------+---------------------------+-------------------------++------------------------------------------------------------------------------+
|| Retina U-Net V002       | BB (+ SS)              | BB                        | retinaunet_v002         || train=retinaunet_v002 module=RetinaNetV002                                   |
+--------------------------+------------------------+---------------------------+-------------------------++------------------------------------------------------------------------------+
|| Box Mask RCNN V002      | BB + BI                | BB                        |                         ||                                                                              |
+--------------------------+------------------------+---------------------------+-------------------------++------------------------------------------------------------------------------+
|| Box Mask U-RCNN V002    | BB + BI (+ SS)         | BB                        |                         ||                                                                              |
+--------------------------+------------------------+---------------------------+-------------------------++------------------------------------------------------------------------------+

Legend: BB = Bounding Boxes, BI = Binary Mask, SS = Semantic Segmentation (dervied from instance segmentation mask)

Trainning Different Versions of RetinaU-Net:
# TODO: focal loss training
