===============
Developer Guide
===============

TODO: interaction diagram of the different classes  (ptmodule = core)

Intro ... # TODO

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

Overview
********

.. image:: ../_static/nnDetectionModule.svg
   :width: 600
   :align: center
   :alt: nnDetection Module Overview

Config Files
============

- train new model with `exp.tag` key


Specialised Items
=================

.. toctree::
   :maxdepth: 4

   pre
   train
   predict
