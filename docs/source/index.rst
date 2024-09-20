.. nnDetection documentation master file, created by
   sphinx-quickstart on Wed Oct 20 16:22:19 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: ./_static/logos_nndet.png
   :width: 900
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
The resulting self-configuring method, nnDetection, adapts itself without minimal intervention to arbitrary medical (volumetric) detection problems while achieving results en par with or superior to the state-of-the-art.

nnDetection t(w)o Ensemble (nnDetection2E), systematises the design of single-stage, multi-stage and
set-prediction based object detection methods in a unified framework. nnDetection2E outperforms all baseline methods on a new pool of nine generalization datasets. Additionally, it surpasses all existing specialized solutions on two public benchmarking datasets.


**If you use nnDetection(2E) please cite our papers:**

[1] `nnDetection2E: Systematic Design of Medical Object Detection <todo>`_

.. tip::
   TODO

[2] `nnDetection: Self-configuring Medical Object Detection <https://miccai2021.org/openaccess/paperlinks/2021/09/01/341-Paper1836.html>`_

.. tip::
   Baumgartner M., Jäger P.F., Isensee F., Maier-Hein K.H. (2021)

   nnDetection: A Self-configuring Method for Medical Object Detection.
   
   https://doi.org/10.1007/978-3-030-87240-3_51


.. image:: ./_static/main_qualitative.jpg
   :width: 900
   :align: center
   :alt: Qualitative Results of nnDetection2E


Documentation
=============

|:boom:| Welcome to the online Documentation of nnDetection, we provide an extensive information for: |:boom:|

* Installation (Source install & Docker) -> see Installation Tab
* Using nnDetection as a self-configuring Object Detection Method -> see User Guide Tab
* Integrating new functionality when using our framework as a research platform -> see Developer Guide Tab
* Using the evaluation framework of nnDetection to compute metrics for Object Detection problems -> see User Guide Tab
* Guides to prepare 21 different medical detection datasets -> see User Guide Tab


News
====

nnDetection2E is now publicly available
---------------------------------------

|:computer:| The code for nnDetection t(w)o ensemble is not publicly available. It systematises the design of single-stage, multi-stage and set-prediction based object detection methods in a unified framework. nnDetection2E outperforms all baseline methods on a new pool of nine generalization datasets. Additionally, it surpasses all existing specialized solutions on two public benchmarking datasets.


[Related Project] Deep-learning based Detection of Vessel Occlusions was Accepted at Nature Communications
----------------------------------------------------------------------------------------------------------

|:page_facing_up:| Fast and accurate detection and vessel occlusions in CTA images is an important clinical task but was previously tackled with many hand crafted solutions. In our study, we present a detection based approach which provides provides great resutls without relying on expensive pre-processing or anatomical limitations. Intereted in our findings? Check out our `Paper <https://www.nature.com/articles/s41467-023-40564-8>`_ . 

.. tip::
   Brugnara, G., Baumgartner, M., Scholze, E. D., Deike-Hofmann, K., Kades, K., Scherer, J., ... & Vollmuth, P. (2023).

   Deep-learning based detection of vessel occlusions on CT-angiography in patients with suspected acute ischemic stroke. 
   
   Nature Communications, 14(1), 4938.

nnDetection Ranked Second in the TDSC-ABUS23 Challenge
-------------------------------------------------------------------------------

|:sparkles:| Our method based on nnDetection ranked second in the detection track of the `TDSC-ABUS23 Challenge <https://tdsc-abus2023.grand-challenge.org/TDSC-ABUS2023/>`_ .


DETR Pilot Project was Accepted at BVM23 as Oral presentation
-----------------------------------------------------------------------------------

|:page_facing_up:| Our pilot project to investigate the feasibility of DEtection TRansformers (DETR) for medical object detection was accepted to BVM23 as an oral presentation. It ranked third for best scientific contribution. Intereted in our findings? Check out our `Paper <https://arxiv.org/abs/2306.15472>`_ .

.. tip::
   Ickler, M. K., Baumgartner, M., Roy, S., Wald, T., & Maier-Hein, K. H. (2023, June).
   
   Taming Detection Transformers for Medical Object Detection.
   
   In BVM Workshop (pp. 183-188). Wiesbaden: Springer Fachmedien Wiesbaden.


nnDetection Ranked Third in the Mediastinal Detection Challenge 
---------------------------------------------------------------
|:sparkles:| Our method based on nnDetection ranked third in the detection track of the `Mediastinal Lesion Analysis Challenge 2022 <https://mela.grand-challenge.org/>`_ .

.. tip::
   Baumgartner, M., Full, P.M., Maier-Hein, K.H. (2023).
   
   Accurate Detection of Mediastinal Lesions with nnDetection.
   
   In: Xiao, Y., Yang, G., Song, S. (eds) Lesion Segmentation in Surgical and Diagnostic Applications. CuRIOUS KiPA MELA 2022 2022 2022. Lecture Notes in Computer Science, vol 13648. Springer, Cham.
   
   https://doi.org/10.1007/978-3-031-27324-7_10


nnDetection was Accepted at MICCAI21
------------------------------------

|:page_facing_up:| nnDetection was early accepted to  the International Conference on Medical Image Computing & Computer Assisted Intervention 2021 (MICCAI21)

.. tip::
   Baumgartner M., Jäger P.F., Isensee F., Maier-Hein K.H. (2021)
   
   nnDetection: A Self-configuring Method for Medical Object Detection.
   
   https://doi.org/10.1007/978-3-030-87240-3_51


nnDetection Prototype ranked first in ADAM2020 challenge
--------------------------------------------------------

|:sparkles:| Our prototype version of nnDetection ranked first in the detection task of the ADAM 2020 challenge. Check out the following publication for moer info:

.. tip::
   Baumgartner, Michael, et al.
   
   Retina U-Net for aneurysm detection in MR images.
   
   Automatic Detection and SegMentation Challenge (ADAM) (2020).


Contents
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

   plugins

.. toctree::
   :maxdepth: 2

   archive

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


`transoar <https://github.com/bwittmann/transoar>`_
---------------------------------------------------

3D Deformable Attention for Deformable DETR was integration from transoar, and was extremely helpful. We are grateful for the open source release of this code.

DETR
----

DETR components from multiple repositores were adapted for 3D use, we would like to thank the authors for their great work and open sourcing their code under nice licenses.

* `DETR <https://github.com/facebookresearch/detr>`_
* `Conditional DETR <https://github.com/Atten4Vis/ConditionalDETR>`_
* `Deformable DETR <https://github.com/fundamentalvision/Deformable-DETR>`_
* `detrex: Benchmarking Detection Transformers <https://github.com/IDEA-Research/detrex>`_


Funding
=======
Part of this work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – 410981386 and the Helmholtz Imaging Platform (HIP), a platform of the Helmholtz Incubator on Information and Data Science.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
