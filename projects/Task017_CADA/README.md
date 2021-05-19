# CADA
**Disclaimer**: We are not the host of the data.
Please make sure to read the requirements and usage policies of the data and **give credit to the authors of the dataset**!

Please read the information from the homepage carefully and follow the rules and instructions provided by the original authors when using the data.
- Homepage: https://cada.grand-challenge.org/Introduction/
- Subtask: Task 1 aneurysm detection

## Setup
0. Follow the installation instructions of nnDetection and create a data directory name `Task017_CADA`.
1. Follow the instructions and usage policies to download the data and place the data and labels at the following locations: data -> `Task017_CADA / raw / train_dataset` and labels -> `Task017_CADA / raw / train_mask_images`
2. Run `python prepare.py` in `projects / Task017_CADA / scripts` of the nnDetection repository.

The data is now converted to the correct format and the instructions from the nnDetection README can be used to train the networks.
