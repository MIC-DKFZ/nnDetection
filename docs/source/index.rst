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
For this task, the cumbersome and iterative process of method configuration constitutes a major research bottleneck. 
Recently, nnU-Net has tackled this challenge for the task of image segmentation with great success.
Following nnU-Net’s agenda, in this work we systematize and automate the configuration process for medical object detection.
The resulting self-configuring method, nnDetection, adapts itself without any manual intervention to arbitrary medical detection problems while achieving results en par with or superior to the state-of-the-art.

.. image:: ./_static/nnDetectionFunctional.svg
   :width: 600
   :align: center
   :alt: nnDetection functional overview

|

.. note::
   **If you used nnDetection for your project please cite the following publication(s):**
   
   Baumgartner M., Jäger P.F., Isensee F., Maier-Hein K.H. (2021) nnDetection: A Self-configuring Method for Medical Object Detection. In: de Bruijne M. et al. (eds) Medical Image Computing and Computer Assisted Intervention – MICCAI 2021. MICCAI 2021. Lecture Notes in Computer Science, vol 12905. Springer, Cham. https://doi.org/10.1007/978-3-030-87240-3_51


Features
========
While nnDetection consists of many different configurations of each module to allow for free customization by providing high modularity, some model are provided via standardized configs and can be used out-of-the-box.

+--------------------------+--------------------------------+----------------------------------+-----------------------+
| Models                   | Trainig Signals                | Prediction Output                | Config                |
+--------------------------+--------------------------------+----------------------------------+-----------------------+
+--------------------------+--------------------------------+----------------------------------+-----------------------+
| Box Detection            |                                | Boxes                            |                       |
+--------------------------+--------------------------------+----------------------------------+-----------------------+
|| Retina U-Net V001       | BB + SS                        | Boxes                            |                       |
|| RetinaNet V002          | BB                             | Boxes                            |                       |
|| Retina U-Net V002       | BB + SS                        | Boxes                            |                       |
+--------------------------+--------------------------------+----------------------------------+-----------------------+
|| Faster RCNN V002        | BB                             | Boxes                            |                       |
|| Box Mask RCNN V002      | BB + BI                        | Boxes                            |                       |
|| Box Mask U-RCNN V002    | BB + BI + SS                   | Boxes                            |                       |
+--------------------------+--------------------------------+----------------------------------+-----------------------+



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

   dev/index

.. toctree::
   :maxdepth: 2

   projects

.. toctree::
   :maxdepth: 2

   plugins

.. toctree::
   :maxdepth: 2

   api/index

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
