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
For this task (detecting objects in 3D medical images) we have developed nnDetection which can be use in three different ways:


A self-configuring method for medical object detection
------------------------------------------------------
Following nnU-Net’s agenda, in this work we systematize and automate the configuration process for medical object detection.
The resulting self-configuring method, nnDetection, adapts itself without any manual intervention to arbitrary medical detection problems while achieving results en par with or superior to the state-of-the-art.

.. .. image:: ./_static/nnDetectionFunctional.svg
..    :width: 600
..    :align: center
..    :alt: nnDetection functional overview

.. |

.. notes::

   **If you used nnDetection for your project please cite the following publication(s):**
   
   Baumgartner M., Jäger P.F., Isensee F., Maier-Hein K.H. (2021) nnDetection: A Self-configuring Method for Medical Object Detection. In: de Bruijne M. et al. (eds) Medical Image Computing and Computer Assisted Intervention – MICCAI 2021. MICCAI 2021. Lecture Notes in Computer Science, vol 12905. Springer, Cham. https://doi.org/10.1007/978-3-030-87240-3_51


A medical object detection development framework
------------------------------------------------
nnDetection provides a rich ecosystem of detection models (e.g. Retina Net, Retina U-Net, Faster R-CNN+, Mask R-CNN), standardised access to a large number of datasets and a well tested training, inference and evaluation pipeline.
We have created various projects which use this framework as the primary development framework and achieve state-of-the-art results in clinical applications and challenges.


*Deep-learning based detection of vessel occlusions on CT-angiography in patients with suspected acute ischemic stroke*

.. notes::

   Brugnara, Gianluca and Baumgartner, Michael and Scholze, Edwin D. et al. "Deep-learning based detection of vessel occlusions on CT-angiography in patients with suspected acute ischemic stroke." Nature Communications 14.1 (2023): 4938.

*Accurate Detection of Mediastinal Lesions with nnDetection*
Ranked third in the MELA2022 challenge where three out of five best performing solutions (inlcuding winning solution) were based on nnDetection.

.. notes::

   Baumgartner, Michael, Peter M. Full, and Klaus H. Maier-Hein. "Accurate Detection of Mediastinal Lesions with nnDetection." MICCAI Challenge on Correction of Brainshift with Intra-Operative Ultrasound. Cham: Springer Nature Switzerland, 2022. 79-85.

*Retina U-Net for Aneurysm Detection in MR Images*
Ranked first in the detection track of the ADAM2020 challenge.

.. notes::

   Baumgartner, Michael, et al. "Retina U-Net for aneurysm detection in MR images." Automatic Detection and SegMentation Challenge (ADAM) (2020).


A medical object detection toolbox
----------------------------------
Our repository contains code to evaluate 2D and 3D object detection tasks with a large number of metrics such as mAP, AP and FROC which can be easily integrated into existing code or used for evaluation.
Detailed guides to common and advanced use cases are provided in :ref:`_user_guide`.
#TODO #FIXME: label not working


Features
========

nnDetection can be used in two different ways:

1. As an out-of-the box detection baseline: nnDetection contains a self-configuring method which can be applied to new medical datasets without modifications.
In many applications, it can serve as a strong baseline without manual modifications.

2. As a medical object detection framework: While many features didn't make it into the final self-configuring pipeline, nnDetection comprises many additional options such as Static Backbone networks, a Detection Zoo and much more.
More information on the Detection Zoo can be found :ref:`here<Detection Zoo>` and the :ref:`developer guide<Developer Guide>` provides the best entrypoint for any further modifications.

#TODO #FIXME: ref not working


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
