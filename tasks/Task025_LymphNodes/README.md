# LymphNodes
**Disclaimer**: We are not the host of the data.
Please make sure to read the requirements and usage policies of the data and **give credit to the authors of the dataset**!

Please read the information from the homepage carefully and follow the rules and instructions provided by the original authors when using the data.
- Homepage: https://wiki.cancerimagingarchive.net/display/Public/CT+Lymph+Nodes
- Masks: we used the masks provided by the same page

## Setup
0. Follow the installation instructions of nnDetection and create a data directory name `Task025_LymphNodes`.
1. Down the data and labels and place the data into `Task025_LymphNodes / raw / CT Lymph Nodes` and the labels into `Task025_LymphNodes / raw / MED_ABD_LYMPH_MASKS`
2. Run `python prepare.py` in `projects / Task025_LymphNodes / scripts` of the nnDetection repository.

The data is now converted to the correct format and the instructions from the nnDetection README can be used to train the networks.
