===============
Developer Guide
===============

These sections provide extended guides to implement custom features into the nnDetection framework.

Developer Flags
===============

nnDetection supports a broad range of configurations to provide extended information regarding models and training behavior.
These need to be specifically enable by the user for e.g. debugging purposes:

* `det_extended_logging`: Enable extended logging for some of the models (e.g. criterions for DETR).


Registries
==========

nnDetection uses multiple Registries to keep track of different modules which can be exchanged to customize different parts of the pipelines.
To register a new component, the registry needs to be imported inside the python file and the component needs to be wrapped by the decorator.
An example is shown below:

.. code:: python

   from nndet.ptmodule import MODULE_REGISTRY # import the registry

    # the compoenent will automatically register under the class name
   @MODULE_REGISTRY.register # register component
   class RetinaUNetV001(...):
      ...

The registry can than be accessed as any other dictionary to retrieve the class of the registered component:

.. code:: python

   from nndet.ptmodule import MODULE_REGISTRY # import the registry

   # get class from registry
   module_cls = MODULE_REGISTRY["RetinaUNetV001"]
   
   # initialize object
   module = module_cls(
      model_cfg=model_cfg,
      trainer_cfg=trainer_cfg,
      plan=plan,
   )

To see which modules are available from the command line (e.g. for overwriting keys in the config) the following command can be used:

.. code:: bash

   # nndet_print_reg [registry prefix]

   # e.g. list all modules
   nndet_print_reg module

   # e.g. list all augmentations
   nndet_print_reg augmentation


.. warning::
   To register a component the corresponding python files needs to be imported during runtime.
   While the nnDetection core repository does this for the provided base configurations automatically, external Plugins need to take of this manually.
   Please refer to the Plugins Guide for more information.

MODULE_REGISTRY
***************
The module registry contains the core modules of nnDetection which inherits from the `Pytorch Lightning <https://github.com/PyTorchLightning/pytorch-lightning>`_ Module.
It is the main module which is used for training and inference and contains all the necessary steps to build the final models.
It can be imported from `nndet.ptmodule` and examples can be found in `nndet.ptmodule.retinaunet`.

AUGMENTATION_REGISTRY
*********************
The augmentation registry can be imported from `nndet.io.augmentation` and contains different augmentation configurations. Examples can be found in `nndet.io.augmentation.bg_aug`.

DATALOADER_REGISTRY
*******************
The dataloader registry contains different dataloader classes to customize the IO of nnDetection.
It can be imported from `nndet.io.datamodule` and examples can be found in `nndet.io.datamodule.bg_loader`.

PLANNER_REGISTRY
****************
New plans can be registered via the planner registry which contains classes to define and perform different architecture and preprocessing schemes.
It can be imported from `nndet.planning.experiment` and examples can be found in `nndet.planning.experiment.v001`.

OPTIMIZER_REGISTRY
******************
Different optimizers can be registered in the optimizer registry and selected via the `trainer_cfg.opt_class`. Depending on the configured optimizer class different configuration options and keys are available / need to be configured in the `trainer_cfg`.


Overview
********
This section gives an overview of possible configuration and customization options. 


Custom Preprocessing
====================
The experiment planner defines the entire planning and preprocessing pipeline and is responsible for tying these components together.
They are retrieved from the planner registry and the primary entry point during planning is the `plan_experiment` function.
Individual parameters can be customized by overwriting the respective function e.g. `determine_dummy_2d_data_augmentation`, `determine_forward_backward_permutation`, `determine_target_spacing` and `trigger_low_res_model`.
The `create_architecture_planner` and `create_preprocessor` can be overwritten to implement other architecture planner (responsible for batch size, patch size, kernels etc.) and preprocessor classes (responsible for resampling, intensity normalisation etc.).
The `D3V002EstV1` planner can be used to perform VRAM esitmation on the current GPU like in nnDetection V1. V2 will perform estimation offline with a fixed set of heuristics to ensure reproducibility across GPUs and software versions.


Config Files
============

The config files of nnDetection are responsible for providing information for model configuration (fixed parameters), data loading, augmentation and training.
Training directories of nnDetection are composed of three part `{module name}_{plan name}_{exp tag}`. By changing the exp tag it is possible to create different training runs where hyperparameters are varied.
Each config consists of several parts which will be explained in the following:
Parts of the configs can be overwritten with the following structure `-o train/{XXX}_cfg@{key}_cfg={value}` e.g. `-o train/augment_cfg@augment_cfg=my_custom_aug`.

Augmentation
-------------

The augmentation part of the config file is responsible for defining the augmentation pipeline.
The `name` key is simply a short identifier of the augmentation config for easy lookup in the json file which will be saved for each training run.
The `transforms` key defines the augmentation transformations which will be executes in the python code. It will be retrieved through the augmentation registry.
The remaining parameters will depend on the selceted augmentation pipeline the implemented augmentation transformations.

Custom augmentation pipelines can be created by inserting new augmentations are declaring new pipelines in python.
Their configuration can than be changed via the config files. nnDetection also provides an interface to use augmentation from MONAI (see `MonaiTransform`).  

Data-Loading
------------

The `dataloader` key specified the intended dataloader class which will be retrieved from the dataloader registry.
The reaining parameters depend on the selected dataloader.

Each dataloader implementation is composed out of fours parts:

* the base moduel: this provides the general basis for the dataloader and is the access point from the outside
* the selection mixin: this mixin is responsible to select case ids and instance ids which shuld be sampled from the DATALOADER_REGISTRY
* the foreground mixin: given the case id and instance id, this mixin is responsible to load the data from the disk and crop the patch around the object
* the background mixin: this mixin is responsible to load the data from the disk and crop a patch, most implementations simply crop randomly.

A new dataloader can be created by mixing these four components. An example is shown below:

.. code:: python

   @DATALOADER_REGISTRY.register
   class DataLoader3DOffsetV2(
      RandomBGCrop3D, # define background cropping
      OffsetFGCrop3DV2, # define foreground cropping
      RandomSelectionMixin, # define selection strategy
      BaseDataLoader3D, # define base module
   ):
      ...

Trainer Config
--------------

The trainer config defines the learning rate, length of the training, metrics to observe and optimizer hyperparemters.
The `opt_class` key specifies the optimizer class which will be retrieved from the optimizer registry.
The remaining parameters are highly dependent on the selected optimizer but should be self explanatory.

Accelerator Config
------------------

A small configuration file containing the hardware resoruce and model optimizsation settings.
Sometime additional speed ups can be achieved by using the `gpu1_mixed16_bench` but it might not work on all datasets depending on the determined patch size and model configuration.
Multi-gpu support is not officially supported but can be performed by increasing the number of GPUs in lightning.
Please note, that the online validation won't compute metrics since the metrics will simply be averaged across GPUs and the inference (including final validation) do not support multi gpu setups.
These were never tested and there might be other aspects influencing the performance of the models (e.g. number of steps need to be scaled).
**Use multi-gpu at your own risk.**

Model Config
------------

The model config defined fixed parameters for the selected architecture.
The exact set of paraemeters will vary between models and need to be cross-referenced with the respective model parameters in the code or documentation.
The majority of parameters will be self-explentory, e.g. `loss_weight` defined the weight of the respective loss. 


Customized Models
=================

nnDetection uses `Pytorch Lightning` for training to provide a widely used, standardiced structure for its models.
Instead of using the lightning module directly, all modules in nnDetection are build on `LightningBaseModule` (`nndet.ptmodule.module`) which integrates additional procedures to setup transformations, the evaluation and the prediction pipeline.
A flow chart visualising the call procedure of nnDetection can be found below.

Each detection module in nnDetection should be a combination of the `LightningBaseModule` and multiple `Mixins` which are explained below.
By leveraging `Mixins` nnDetection can cover various input/output formats and provide models for: Bounding Box Detection + auxiliary task training, Instance Segmentation + auxiliary task training.
An example which builds a standard RetinaNet is shown below:

.. code:: python
    
    class RetinaNetModule(
        # nnDetection Base Module to integrate other mixins
        LightningBaseModule,
        # Convert the dataloader output to bounding boxes
        BoxesPrepareMixin,
        # Run Bounding Box evaluation during training
        BoxEvalMixin,
        # Use model structure of single stage detector
        SingleStageMixin,
        # Run the default Bounding Box Prediction and Sweep
        BoxPredictionMixin,
    ):
        # Cutomize SingleStageMixin attributes to switch between backbones,
        # necks, heads, sampling strategies, losse and much more
        ...


LightningBaseModule
~~~~~~~~~~~~~~~~~~~
The base module provides a standardized initialization procedure for all detection modules and is responsible for initializing the `Mixins`.
Every module should be composed of multiple `Mixins` which define the transformations to provide the ground truth format, model building structure, evaluation, optimizers and predictions pipeline.
More information about `Mixins` can be found below.
Furthermore, the base module configures some standardized callbacks like `EpochTimerCallback` and `Stochastic Weight Averaging (SWA)`.
`SWA` can be enabled by configuring the `swa_epochs` key in the config file.

Optimizers
~~~~~~~~~~
The optimizer class can be defind via the `trainer_cfg.opt_class` entry in the configs, please refer to the `OPTIMIZER_REGISTRY` for more info.

Model Mixins
~~~~~~~~~~~~
The `ModelMixin` (`nndet.ptmodule.mixins.model`) is responsible for building the model.
It provides the `from_config_plan` classmethod which will be called by the `LightningBaseModule` to initialize the model.
Depnding on the exact model type, different methods can be overwritten to customize the building behavior of the individual parts of the model.
Most changes can be made without overwriting methods though.
The `ModelMixin` introduces several class attributes which can be used to exchange modules, an example is provided below:

`SingleStageMixin`
 .. code:: python                         
                                          
    class SingleStageMixin(ModelMixin):   
        head_classifier_cls = ...         

`Binary Cross Entropy Loss`
 .. code:: python  
                          
    from nndet.nn.heads.classifier import BCECLassifier
                                 
    class SingleStageDetectorBCELoss(
        ...
        SingleStageMixin,
        ...
        ):      
        head_classifier_cls = BCECLassifier    

`Cross Entropy Loss`
 .. code:: python 
                           
    from nndet.nn.heads.classifier import CECLassifier
                                 
    class SingleStageDetectorCELoss(
        ...
        SingleStageMixin,
        ...
        ):      
        head_classifier_cls = CECLassifier         

Prepare Mixins
~~~~~~~~~~~~~~
`PrepareMixin`s are responsible for converting the output of the dataloader into the desired target format and can be xustomized by overwriting the `get_pre_transforms` method.
Sometimes it is necessary to add multiple `PrepareMixin` to create different ground truth formats, e.g. Retina U-Net requires bounding boxes and semantic segmentations.
In general there are three `PrepareMixin` Types which save the result in different keys:

* `BoxesPrepareMixin` saves the boxes in `boxes` and class in `classes`
* `SemanticPrepareMixin` save semantic segmentation into `target_seg`
* `SemanticFgPrepareMixin` save semantic segmentation (fg vs bg) into `target_seg`
* `BinaryMasksPrepareMixin` save binary masks into `target_binary_masks`

Eval Mixins
~~~~~~~~~~~
The `EvalMixin` defines the metrics which are tracked during the trainig.
It provides three important methods which can be used to customize the bahvior:

* `evaluation_init`: initilize the `Evaluator` (see `nndet.evaluator`) object
* `evaluation_step`: is called in every validation step and should cache intermediate results
* `evaluation_end`: is called at the end of the validation epoch to compute the final validation metrics.

Since most detection metrics are computed over the whole data set `evaluation_step` usually does not return intermediate metrics and `evaluation_end` will aggreagte the prediction and gt to compute the final set of metrics.

.. warning::
    While Distributed training can be used in nnDetection, the cached values inside the `Evaluator` are not synchronized between different workers.
    As a result, every worker will compute it's own set of metrics and these will be averaged over the workers.

Predict Mixins
~~~~~~~~~~~~~~
The prediction consists of two parts: `sweep` which is executed after the training to determine the best inference parameters and `get_predictor` which will create a predictor object to run inference.
These methods can be used to customize the sweeping and inference strategy of the module by exchanging the initialization of the `Sweeper` Object and `Predictor` / `Ensembler` objects.


Customized Losses
-----------------

There are four different loss categories in nnDetection:

- `regression`: theses losses are usually used for regression tasks (e.g. regression of bounding boxes) 
   and receive inputs in the form `[*, #dims * 2]` where `*` are arbitrary dimensions and `#dims` are
   the number of spatial dimenesions. Example shapes: RetinaNet `[N, #dims * 2]`, RCNN `[R, #dims * 2]`,
   DETR `[T, #dims * 2]` where `N` is the number of anchors, `R` is the number of RoIs, `T` number
   of matched boxes.
- `classification`: these losses are used for classification task and receive inputs in the form
   `[*, C]` where `*` are arbitrary dimensions and `C` is the number of *foreground* classes. The targets
   are encoded as numerical values. Note this is different to pytorch where the number
   of classes is usually located at the first dimension. Example shapes: RetinaNet `[N, C]`,
   RCNN `[R, C]`, DETR `[B, T, C]` where `N` is the number of anchors,
   `R` is the number of RoIs, `B` is the batch size, `T` number of boxes per image.
   In this setting, the targets are provided as numerical values where `0`
   is considered background and thus when creating the one hot encoding the 
   first channel (which is filled with 0s) is removed (so the maximal value in
   targets is `C+1` in this case). This is different from the segmentation losses
   where number of classes refers to the total number of clases (i.e. in this
   case the maximal value in targets should be equal to `C`)!
- `segmentation`: these losses are used for per-location classifiation of feature maps.
   They receive input in the form of `[B, C, *]`, where `B` is the batch size, 
   `C` is the number of classes and `*` are arbitrary dimensions.
- `mask`: these losses are used for per-location binary classification of feature maps.
   The input follows the same format at segmentation losses but the targets
   are already ont hot encoded, i.e. they have shape `[B, C, *]` , where `B` is
   the batch size, `C` is the number of classes and `*` are arbitrary dimensions.


Distribution Package
====================
To create a binary distribution package (wheel) one simply need to execute the following command:

.. code:: bash
   python setup.py bdist_wheel

On the other hand, to create a source distribution package (tarball) execute the following command:

.. code:: bash
   python setup.py sdist

The following docker command can be used to build the the distribution packages:

.. code:: bash
   docker run --rm --gpus all -v .:/opt/nndet --shm-size=48gb continuumio/miniconda3 /bin/bash -c "conda create --name venv python=3.10 -y && source activate base && conda activate venv && export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++ && export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc && pip3 install torch torchvision torchaudio && conda install cuda -c nvidia/label/cuda-$(python -c "import torch; print(torch.version.cuda)") -y && conda install gxx_linux-64 -y && cd /opt/nndet && rm -rf build nndet.egg-info && python setup.py bdist_wheel && python setup.py sdist"

After the execution, both the binary distribution package (`.whl`) or the source distribution (`.tar.gz`) can be then found in the `dist` directory.