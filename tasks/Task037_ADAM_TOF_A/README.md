# ADAM TOF Aneurysm
**Disclaimer**: We are not the host of the data.
Please make sure to read the requirements and usage policies of the data and **give credit to the authors of the dataset**!

Please read the information from the homepage carefully and follow the rules and instructions provided by the original authors when using the data.
- Homepage: http://adam.isi.uu.nl/
- Subtask: Task 1

```
Timmins, Kimberley M., et al. "Comparing methods of detecting and segmenting unruptured intracranial aneurysms on TOF-MRAS: the ADAM challenge." Neuroimage 238 (2021): 118216.
```

## Setup

Note: This will prepare a different version of the ADAM dataset for training and evalation. In contrast to the nnDetection V1 verison
this will only inlcude the TOF MRA image and only untreated & unruptured aneurysms will be considered objects. 

0. Follow the installation instructions of nnDetection and create a data directory name `Task037_ADAM_TOF_A`.
1. Follow the instructions and usage policies to download the data and place the data into `Task037_ADAM_TOF_A / raw / ADAM_release_subjs`
2. Run `python prepare.py` in `projects / Task037_ADAM_TOF_A / scripts` of the nnDetection repository.

The data is now converted to the correct format and the instructions from the nnDetection README can be used to train the networks.
