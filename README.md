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

[1] [nnDetection: A Self-configuring Method for Volumetric 3D Object Detection]()
```
Baumgartner M., Kovacs B., Ickler M.K., Jäger P.F., Isensee F., Ulrich C., Wald T., Holzschuh J.C., Ghosh P., for the ALFA study, Maier-Hein K.H.
nnDetection: A Self-configuring Method for Volumetric 3D Object Detection.
Nature Methods (in press, 2026).
```

[2] [nnDetection: A Self-configuring Method for
Medical Object Detection](https://miccai2021.org/openaccess/paperlinks/2021/09/01/341-Paper1836.html)


    Baumgartner M., Jäger P.F., Isensee F., Maier-Hein K.H. (2021)
    nnDetection: A Self-configuring Method for Medical Object Detection.
    https://doi.org/10.1007/978-3-030-87240-3_51


**Installation & Self-configuring Application & Research Platform & Evaluation Framework**

<div align="center">

:boom: :exclamation: Please refer to our [online documentation](docs/source/index.rst) :exclamation: :boom:

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

# Acknowledgements
nnDetection combines the information from multiple open source repositores we wish to acknoledge for their awesome work, please check them out!

### [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
nnU-Net is self-configuring method for semantic segmentation and many steps of nnDetection follow in the footsteps of nnU-Net.

### [Medical Detection Toolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
The Medical Detection Toolkit introduced the first codebase for 3D Object Detection and multiple tricks were transferred to nnDetection to assure optimal configuration for medical object detection.

### [Torchvision](https://github.com/pytorch/vision)
nnDetection tried to follow the interfaces of torchvision to make it easy to understand for everyone coming from the 2D (and video) detection scene. As a result we used based our implementations of some of the core modules of the torchvision implementation.


### [transoar](https://github.com/bwittmann/transoar)
3D Deformable Attention for Deformable DETR was integration from transoar, and was extremely helpful. We are grateful for the open source release of this code.

### DETR

DETR components from multiple repositores were adapted for 3D use, we would like to thank the authors for their great work and open sourcing their code under nice licenses.

- [DETR](https://github.com/facebookresearch/detr)
- [Conditional DETR](https://github.com/Atten4Vis/ConditionalDETR)
- [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)
- [detrex: Benchmarking Detection Transformers](https://github.com/IDEA-Research/detrex)


## Funding
Part of this work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – 410981386 and the Helmholtz Imaging Platform (HIP), a platform of the Helmholtz Incubator on Information and Data Science.


# License
This project is licensed under multiple licenses, please refer to the `LICENSES` directory for an overview of the licenses.

# Copyright
Copyright German Cancer Research Center (DKFZ) and contributors.
