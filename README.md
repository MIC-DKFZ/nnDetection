<div align="center">

<img src=docs/source/_static/logos_nndet.png width="600px">

![Version](https://img.shields.io/badge/nnDetection-v2.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-orange)

</div>

# What is nnDetection?
Simultaneous localisation and categorization of objects in medical images, also referred to as medical object detection, is of high clinical relevance because diagnostic decisions depend on rating of objects rather than e.g. pixels.
For this task, the cumbersome and iterative process of method configuration constitutes a major research bottleneck. 
Recently, nnU-Net has tackled this challenge for the task of image segmentation with great success.
Following nnU-Net’s agenda, in this work we systematize and automate the configuration process for medical object detection.
The resulting self-configuring method, nnDetection, adapts itself without minimal intervention to arbitrary medical (volumetric) detection problems while achieving results en par with or superior to the state-of-the-art.

nnDetection t(w)o Ensemble (nnDetection2E), systematises the design of single-stage, multi-stage and
set-prediction based object detection methods in a unified framework. nnDetection2E outperforms all baseline methods on a new pool of nine generalization datasets. Additionally, it surpasses all existing specialized solutions on two public benchmarking datasets.

**If you use nnDetection(2E) please cite our papers:**

[1] [nnDetection2E: A Self-Configuring Ensemble for
Generalized Medical Object Detection]()
```
TODO
```

[2] [nnDetection: A Self-configuring Method for
Medical Object Detection](https://miccai2021.org/openaccess/paperlinks/2021/09/01/341-Paper1836.html)


    Baumgartner M., Jäger P.F., Isensee F., Maier-Hein K.H. (2021)
    nnDetection: A Self-configuring Method for Medical Object Detection.
    https://doi.org/10.1007/978-3-030-87240-3_51


**Installation & Self-configuring Application & Research Platform & Evaluation Framework**

<div align="center">

:boom::exclamation:Please refer to our [online documentation]():exclamation::boom:

</div>

We provide an extensive documentation for:
- Installation (Source install & Docker)
- Using nnDetection as a self-configuring Object Detection Method
- Integrating new functionality when using our framework as a research platform
- Using the evaluation framework of nnDetection to compute metrics for Object Detection problems
- Guides to prepare 21 different medical detection datasets

<div align="center">
<img src=docs/source/_static/main_qualitative.jpg width="900px">
</div>

# News

### nnDetection2E is now publicly available
:computer: The code for nnDetection t(w)o ensemble is not publicly available. It systematises the design of single-stage, multi-stage and set-prediction based object detection methods in a unified framework. nnDetection2E outperforms all baseline methods on a new pool of nine generalization datasets. Additionally, it surpasses all existing specialized solutions on two public benchmarking datasets.

### [Related Project] Deep-learning based Detection of Vessel Occlusions was Accepted at Nature Communications
:page_facing_up::tada: Fast and accurate detection and vessel occlusions in CTA images is an important clinical task but was previously tackled with many hand crafted solutions. In our study, we present a detection based approach which provides provides great resutls without relying on expensive pre-processing or anatomical limitations. Intereted in our findings? Check out our [paper](https://www.nature.com/articles/s41467-023-40564-8). 


### [Related Project] DETR Pilot Project was Accepted at BVM23 as Oral presentation
:page_facing_up::tada: Our pilot project to investigate the feasibility of DEtection TRansformers (DETR) for medical object detection was accepted to BVM23 as an oral presentation. It ranked third for best scientific contribution. Intereted in our findings? Check out our [paper](https://arxiv.org/abs/2306.15472).

### nnDetection was Accepted at MICCAI21
:page_facing_up::tada: nnDetection was early accepted to  the International Conference on Medical Image Computing & Computer Assisted Intervention 2021 (MICCAI21)

# Acknowledgements
nnDetection combines the information from multiple open source repositores we wish to acknoledge for their awesome work, please check them out!

## [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
nnU-Net is self-configuring method for semantic segmentation and many steps of nnDetection follow in the footsteps of nnU-Net.

## [Medical Detection Toolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
The Medical Detection Toolkit introduced the first codebase for 3D Object Detection and multiple tricks were transferred to nnDetection to assure optimal configuration for medical object detection.

## [Torchvision](https://github.com/pytorch/vision)
nnDetection tried to follow the interfaces of torchvision to make it easy to understand for everyone coming from the 2D (and video) detection scene. As a result we used based our implementations of some of the core modules of the torchvision implementation.


## License
This project is licensed under multiple licenses, please refer to the `LICENSES` directory for an overview of the licenses. 


## Funding
Part of this work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – 410981386 and the Helmholtz Imaging Platform (HIP), a platform of the Helmholtz Incubator on Information and Data Science.
