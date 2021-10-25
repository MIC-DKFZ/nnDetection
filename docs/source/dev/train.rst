========
Training
========

Dataloading
===========
# TODO: write new augmentation pipelines
# TODO: write new dataloaders

Lightning Module
================

Working with Config and Hydra
-----------------------------
TBD

Building New nnDetection Models
-------------------------------

nnDetection uses `Pytorch Lightning` for training to provide a widely used, standardiced structure for its models.
Instead of using the lightning module directly, all modules in nnDetection are build on `LightningBaseModule` (`nndet.ptmodule.module`) which integrates additional procedures to setup transformations, the evaluation and the prediction pipeline.
A flow chart visualising the call procedure of nnDetection can be found below.

# TODO: flow chart

Each detection module in nnDetection should be a combination of the `LightningBaseModule` and multiple `Mixins` which are explained below.
By leveraging `Mixins` nnDetection can cover various input/output formats and provide models for: Bounding Box Detection + auxiliary task training, Instance Segmentation + auxiliary task training.
An example which builds a standard RetinaNet is shown below:

.. code:: python
    
    class RetinaNetModule(
        # Define the Optimzier class of this module, this needs to be listed first
        SGDDefaultMixin,
        # nnDetection Base Module to integrate other mixins
        LightningBaseModule,
        # Convert the dataloader output to bounding boxes
        BoxPrepareMixin,
        # Run Bounding Box evaluation furing training
        BoxEvalMixin,
        # Use model structure of single stage detection
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
                          
    from nndet.arch.heads.classifier import BCECLassifier
                                 
    class SingleStageDetectorBCELoss(
        ...
        SingleStageMixin,
        ...
        ):      
        head_classifier_cls = BCECLassifier    

`Cross Entropy Loss`
 .. code:: python 
                           
    from nndet.arch.heads.classifier import CECLassifier
                                 
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

- `BoxPrepareMixin` saves the boxes in `boxes` and class in `classes` #TODO
- `SemanticPrepareMixin` saves the result in `target_seg`
- `InstancePrepareMixin`: #TODO

Eval Mixins
~~~~~~~~~~~
The `EvalMixin` defines the metrics which are tracked during the trainig.
It provides three important methods which can be used to customize the bahvior:

- `evaluation_init`: initilize the `Evaluator` (see `nndet.evaluator`) object
- `evaluation_step`: is called in every validation step and should cache intermediate results
- `evaluation_end`: is called at the end of the validation epoch to compute the final validation metrics.

Since most detection metrics are computed over the whole data set `evaluation_step` usually does not return intermediate metrics and `evaluation_end` will aggreagte the prediction and gt to compute the final set of metrics.

.. warning::
    While Distributed training can be used in nnDetection, the cached values inside the `Evaluator` are not synchronized between different workers.
    As a result, every worker will compute it's own set of metrics and these will be averaged over the workers.

Predict Mixins
~~~~~~~~~~~~~~
The prediction consists of two parts: `sweep` which is executed after the training to determine the best inference parameters and `get_predictor` which will create a predictor object to run inference.
These methods can be used to customize the sweeping and inference strategy of the module by exchanging the initialization of the `Sweeper` Object and `Predictor` / `Ensembler` objects.

Evaluation
==========
