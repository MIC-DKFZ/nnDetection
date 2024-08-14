## Panorama
**Disclaimer**: We are not the host of the data.
Please make sure to read the requirements and usage policies of the data and **give credit to the authors of the dataset**!

Please read the information from the homepage carefully and follow the rules and instructions provided by the original authors when using the data.
- Homepage: https://www.synapse.org/Synapse:syn51156910/wiki/627000

Note: We only use a subset of the provided panorama data to avoid data leakage and high quality annotations: we exclude all cases of the MSD Pancreas dataset and cases which only have automatic labels. This happens automatically when the images get prepared by the script.

## Preparation

0. Follow the installation instructions of nnDetection and create a data directory name `Task057_BraTSMets`.
1. Download the dataset via the official website (train, train additional and UCSF) and place it into a directory called `raw` inside the task directory.
2. The final folder structure should look like this:

- Task057_BraTSMets
    - raw
        - ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData
        - ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData_Additional
        - UCSF_BrainMetastases_v1.3
            - TableS1_UCSF_BrainMetastases_SubjectInfo.xlsx
            - UCSF_BrainMetastases_TRAIN

4. Execute `python prepare.py` in the `nndet / tasks / Task057_BraTSMets / scripts` directory. This requires the installation of `cc3d`! 
5. Continue with the normal nnDetection command order.
