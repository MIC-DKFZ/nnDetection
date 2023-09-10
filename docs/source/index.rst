.. nnDetection documentation master file, created by
   sphinx-quickstart on Wed Oct 20 16:22:19 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: ./_static/nnDetection.svg
   :width: 500
   :align: center
   :alt: nnDetection

|
|


What is nnDetection?
====================
Simultaneous localisation and categorization of objects in medical images, also referred to as medical object detection, is of high clinical relevance because diagnostic decisions depend on rating of objects rather than e.g. pixels.
For this task we have developed nnDetection which can be use in three different ways:


A self-configuring method for medical object detection
------------------------------------------------------
For this task, the cumbersome and iterative process of method configuration constitutes a major research bottleneck. 
Recently, nnU-Net has tackled this challenge for the task of image segmentation with great success.
Following nnU-Net’s agenda, in this work we systematize and automate the configuration process for medical object detection.
The resulting self-configuring method, nnDetection, adapts itself without any manual intervention to arbitrary medical detection problems while achieving results en par with or superior to the state-of-the-art.

.. .. image:: ./_static/nnDetectionFunctional.svg
..    :width: 600
..    :align: center
..    :alt: nnDetection functional overview

.. |

.. note::
   **If you used nnDetection for your project please cite the following publication(s):**
   
   Baumgartner M., Jäger P.F., Isensee F., Maier-Hein K.H. (2021) nnDetection: A Self-configuring Method for Medical Object Detection. In: de Bruijne M. et al. (eds) Medical Image Computing and Computer Assisted Intervention – MICCAI 2021. MICCAI 2021. Lecture Notes in Computer Science, vol 12905. Springer, Cham. https://doi.org/10.1007/978-3-030-87240-3_51


A medical object detection development framework
------------------------------------------------
nnDetection provides a rich ecosystem of detection models (e.g. Retina Net, Retina U-Net, Faster R-CNN+, Mask R-CNN), standardised access to a large number of datasets and a well tested training, inference and evaluation pipeline.
We have created various projects which use this framework as the primary development framework and achieve state-of-the-art results in clinical applications and challenges.

*Deep-learning based detection of vessel occlusions on CT-angiography in patients with suspected acute ischemic stroke*
.. note::
   Brugnara, Gianluca and Baumgartner, Michael and Scholze, Edwin D. et al. "Deep-learning based detection of vessel occlusions on CT-angiography in patients with suspected acute ischemic stroke." Nature Communications 14.1 (2023): 4938.

*Accurate Detection of Mediastinal Lesions with nnDetection*
Ranked third in the MELA2022 challenge where three out of five best performing solutions (inlcuding winning solution) were based on nnDetection.
.. note::
   Baumgartner, Michael, Peter M. Full, and Klaus H. Maier-Hein. "Accurate Detection of Mediastinal Lesions with nnDetection." MICCAI Challenge on Correction of Brainshift with Intra-Operative Ultrasound. Cham: Springer Nature Switzerland, 2022. 79-85.

*Retina U-Net for Aneurysm Detection in MR Images*
Ranked first in the detection track of the ADAM2020 challenge.
.. note::
   Baumgartner, Michael, et al. "Retina U-Net for aneurysm detection in MR images." Automatic Detection and SegMentation Challenge (ADAM) (2020).


A medical object detection toolbox
----------------------------------
Our repository contains code to evaluate 2D and 3D object detection tasks with a large number of metrics such as mAP, AP and FROC which can be easily integrated into existing code or used for evaluation.
Detailed guides to common and advanced use cases are provided in :ref:`_user_guide-label`.


Features
========

nnDetection can be used in two different ways:

1. As an out-of-the box detection baseline: nnDetection contains a self-configuring method which can be applied to new medical datasets without modifications.
In many applications, it can serve as a strong baseline without manual modifications.

2. As a medical object detection framework: While many features didn't make it into the final self-configuring pipeline, nnDetection comprises many additional options such as Static Backbone networks, a Detection Zoo and much more.
More information on the Detection Zoo can be found :ref:`here<Detection Zoo>` and the :ref:`developer guide<Developer Guide>` porivdes the best entrypoint for any further modifications.


Contents:
=========

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. toctree::
   :maxdepth: 2

   installation

.. toctree::
   :maxdepth: 2

   user_guide

.. toctree::
   :maxdepth: 3

   dev

.. toctree::
   :maxdepth: 2

   projects

.. toctree::
   :maxdepth: 2

   plugins

.. toctree::
   :maxdepth: 2

   api/index

FAQ
===
.. collapse:: Installation & Initial Setup Errors

   1. Error: Undefined CUDA symbols when importing `nndet._C` or other import related Errors from `nndet._C` or CUDA related ARCH errors
   nnDetection includes additional CUDA code which needs to compiled upon installation and thus requires correct configuration of the CUDA dependencies.
   Please double check CUDA version of your PC, pytorch, torchvision and nnDetection build.
   This can be done by running `nndet_env` if the installation succeeded  or by running `python scripts/utils.py`.
   Things to look out for: Make sure that the versions of PyTorch CUDA and NVCC CUDA match (minor version mismatch as in this case, will work without error but could potentially introduce bugs.)
   `OMP_NUM_THREADS` should always be set to 1 and `det_num_threads` should always be lower or equal `Systemm CPU Count`.
   An example output of the command is shown below:

   .. code::

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

   2. Error persists even after fixing the environment
   Make sure to delete the `build` folder before rerunning the installation since it won't recompile the code otherwise.

   3. Error: No kernel image is available for execution
   You are probably executing the build on a machine with a GPU architecture which was not present/set during the build.
   Please check `here <https://developer.nvidia.com/cuda-gpus>`_ to find the correct SM architecture and set `TORCH_CUDA_ARCH_LIST`
   approriately (e.g. check Dockefile for example).
   As before make sure to delete the `build` folder when rerunning the installation process.

   4. Open Issue:
   Please open an Issue and provide your environment as obtained by `nndet_env`.

.. collapse:: Training doesn't start or is stuck

   1. Please run `nndet_env` and make sure `OMP_NUM_THREADS` is set to 1. No other values are supported here. To increase the number of workers used for IO and augmentation adjust `nndet_num_threads`.
   2. Try running the training without multiprocessing as a sanity check: `nndet_train XXX -o augment_cfg.multiprocessing=False`. Don't use this for the full training, this is just one step of the debugging process.
   3. Please open an Issue and provide your environment as obtained by `nndet_env` and report if the training without multiprocessing started correctly.

.. collapse:: (Slow) Training Speed

   The training time of nnDetection should be roughly equal for most data sets: 2 days (1-2 hours per epoch) with mixed precision 3d speed up and 4 days without (this number refers to RTX 2080TI, newer GPUs can be significantly faster, on high end configuration training takes 1 day). It is highly recommended to use GPUs with Tensor Cores to enable fast mixed precision training for reasonable turnaround times. There can be several reasons for slow training:

   1. PyTorch < 1.9 did not provide training speedup for mixed-precision 3d convs in their pip installable version and it was necessary to build it from source. (the docker build of nnDetection also provides the speedup). Newer versions like 1.10 and 1.11 provide the mixed precision speedup in their pip version (only tested with CUDA 11.X).


   2. There is a bottleneck in the setup. This can be identified as follows:

      a. Check the GPU Util -> it should be high for most of the time if it isn't, there is either a CPU or IO bottleneck. If it is high it is the missing pytorch speed up.

      b. Check CPU util: if the CPU util is high (and the GPU util isn't) more cpu threads are needed for augmentation (can be adjusted via det_num_threads and depends on your CPU).

   If GPU and CPU util are low, it is an IO bottleneck, it is quite hard to do anything about this (a typical SSD with ~500mb/s read speed ran fine for my experiments). If the CPU util is maxed out it is an CPU bottleneck: Adjust det_num_threads (similar to num workers in the normal pytorch dataloaders) for the available CPU resources (set this as high as possible but not more than available CPU threads) otherwise. Increasing the number of workers will increase the required RAM consumption -> make sure not to run out of memory there otherwise the training will be extreeemly slow and the workstation might crash.

   Examples for det_num_threads:

   - CPUs with less cores but high clock speed: Needs a lower det_num_threads value. On an Intel i7 9700 (non k) det_num_threads=6 reaches 90+ % GPU usage.

   - CPUs with many cores but lower clock speed: Needs a high det_num_threads value. In cluster environments det_num_threads=12 reaches ~80+% GPU usage.

.. collapse:: GPU requirements

   nnDetection v0.1 was developed for GPUs with at least 11GB of VRAM (e.g. RTX2080TI, TITAN RTX).
   All of our experiments were conducted with a RTX2080TI.
   While the memory can be adjusted by manipulating the correct setting we recommend using the default values for now.
   Future releases will refactor the planning stage to improve the VRAM estimation and add support for different memory budgets.

.. collapse:: Training with bounding boxes

   The first release of nnDetection focuses on 3d medical images and Retina U-Net.
   As a consequence training (specifically planning and augmentation) requrie segmentation annotations.
   In many cases this limitation can be circumvented by converting the bounding boxes into segmentations.


.. collapse:: 2D data sets

   2D data sets are not supported but other great 2D frameworks exist which can be adapted for medical use cases like torchvision, detectron2, mmdetection and YOLOv5.

.. collapse:: Multi GPU Training

   Multi GPU training is not officially supported.
   Inference and the metric computation are not properly designed to support these usecases!


Acknowledgements
================
nnDetection incorporates the information from multiple open source repositores which we wish to acknoledge for their awesome work, please check them out!

`nnU-Net <https://github.com/MIC-DKFZ/nnUNet>`_
-----------------------------------------------

nnU-Net is self-configuring method for semantic segmentation and many steps of nnDetection follow in the footsteps of nnU-Net.

`Medical Detection Toolkit <https://github.com/MIC-DKFZ/medicaldetectiontoolkit>`_
----------------------------------------------------------------------------------

The Medical Detection Toolkit introduced the first codebase for 3D Object Detection and multiple tricks were transferred to nnDetection to assure optimal configuration for medical object detection.

`Torchvision <https://github.com/pytorch/vision>`_
--------------------------------------------------

nnDetection tried to follow the interfaces of torchvision to make it easy to understand for everyone coming from the 2D (and video) detection scene. As a result we used based our implementations of some of the core modules of the torchvision implementation.

Funding
=======
Part of this work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – 410981386 and the Helmholtz Imaging Platform (HIP), a platform of the Helmholtz Incubator on Information and Data Science.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


TODOs
=====
- application limited to 3D
- Pointer to Projects and Plugins
- Improvements
   - select best model for evaluation
   - run inference on CPU (inference_kwargs.device=cpu)
   - run segmentation of RetinaU-Net
