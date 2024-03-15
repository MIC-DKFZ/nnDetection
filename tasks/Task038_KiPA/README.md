## KiPA22
**Disclaimer**: We are not the host of the data.
Please make sure to read the requirements and usage policies of the data and **give credit to the authors of the dataset**!

Please read the information from the homepage carefully and follow the rules and instructions provided by the original authors when using the data.
- Homepage: https://kipa22.grand-challenge.org/


## Preparation

0. Follow the installation instructions of nnDetection and create a data directory name `Task038_KiPA`.
1. Download the dataset via the official website and place it into a directory called `raw` inside the task directory.
2. The final folder structure should look like this:

- Task038_KiPA
    - raw
        - train
            - image
                - 0.nii.gz
                - ...
            - label
                - 0.nii.gz
                - ...

4. Execute `python prepare.py` in the `nndet / tasks / Task038_KiPA / scripts` directory
5. Run `nndet_seg2det 038` to convert the semantic segmentation into instanes (all images will have a single instance except 2 which have 2 instances)

The data is now saved in the correct format. Please folow the instructions from the nnDetection README can be used to train the networks.
