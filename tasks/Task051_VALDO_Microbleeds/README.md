## VALDO Microbleeds
**Disclaimer**: We are not the host of the data.
Please make sure to read the requirements and usage policies of the data and **give credit to the authors of the dataset**!

Please read the information from the homepage carefully and follow the rules and instructions provided by the original authors when using the data.
- Homepage: https://valdo.grand-challenge.org/


## Preparation

0. Follow the installation instructions of nnDetection and create a data directory name `Task051_VALDO_Microbleeds`.
1. Download the dataset via the official website and place it into a directory called `raw` inside the task directory.
2. The final folder structure should look like this:

- Task051_VALDO_Microbleeds
    - Task2
        - sub-101   
            - sub101_space-T2S_CMB.nii.gz
            - sub101_space-T2S_desc-masked_T1.nii.gz
            - sub101_space-T2S_desc-masked_T2.nii.gz
            - sub101_space-T2S_desc-masked_T2S.nii.gz
        - sub-102   
            - sub102_space-T2S_CMB.nii.gz
            - sub102_space-T2S_desc-masked_T1.nii.gz
            - sub102_space-T2S_desc-masked_T2.nii.gz
            - sub102_space-T2S_desc-masked_T2S.nii.gz
        - ...

3. Execute `python prepare.py` in the `nndet / tasks / Task051_VALDO_Microbleeds / scripts` directory
4. Run `nndet_seg2det 051` to convert the semantic segmentation into instanes
5. Run `nndet_test_data_split 051 --size 0.3 --stratify` to split a test set

The data is now saved in the correct format. Please folow the instructions from the nnDetection README can be used to train the networks.
