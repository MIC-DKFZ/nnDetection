## ABUS
**Disclaimer**: We are not the host of the data.
Please make sure to read the requirements and usage policies of the data and **give credit to the authors of the dataset**!

Please read the information from the homepage carefully and follow the rules and instructions provided by the original authors when using the data.
- Homepage: https://tdsc-abus2023.grand-challenge.org/


## Preparation

0. Follow the installation instructions of nnDetection and create a data directory name `Task049_ABUS`.
1. Download the dataset via the official website and place it into a directory called `raw` inside the task directory.
2. The final folder structure should look like this:

- Task049_ABUS
    - raw
        - data
            - DATA_000.nrrd
            - DATA_001.nrrd
            - ...
        - MASK
            - MASK_000.nrrd
            - MASK_001.nrrd
            - ...

4. Execute `python prepare.py` in the `nndet / tasks / Task049_ABUS / scripts` directory

(we do not need to run postprocessing since ABUS has only a single instance annotated per case)

The data is now saved in the correct format. Please folow the instructions from the nnDetection README can be used to train the networks.
