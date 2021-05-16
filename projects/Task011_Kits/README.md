# Kits
**Disclaimer**: We are not the host of the data.
Please make sure to read the requirements and usage policies of the data and **give credit to the authors of the dataset**!

Please read the information from the homepage carefully and follow the rules and instructions provided by the original authors when using the data.
- Homepage: https://kits19.grand-challenge.org/data/

## Setup
0. Follow the installation instructions of nnDetection and create a data directory name `Task011_Kits`.
1. Follow the instructions and usage policies to download the data and place all the folders which contain the data and labels for each case into `Task011_Kits / raw`
2. Run `python prepare.py` in `projects / Task011_Kits / scripts` of the nnDetection repository.
3. Run `nndet_seg2det 011` to convert the semantic segmentation labels to instance segmentations.
4. Run ... to download and replace the manually corrected labels.

The data is now converted to the correct format and the instructions from the nnDetection README can be used to train the networks.
