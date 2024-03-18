## Pancreas Cysts
**Disclaimer**: We are not the host of the data.
Please make sure to read the requirements and usage policies of the data and **give credit to the authors of the dataset**!

Please read the information from the homepage carefully and follow the rules and instructions provided by the original authors when using the data.
- Homepage: https://www.mdpi.com/2075-4418/11/5/901
- Data: https://zenodo.org/records/4621057

```
Abel, Lorraine, et al. "Automated detection of pancreatic cystic lesions on CT using deep learning." Diagnostics 11.5 (2021): 901.
```

## Preparation

0. Follow the installation instructions of nnDetection and create a data directory name `Task041_PancreasCysts`.
1. Download the dataset via the official website and place it into a directory called `raw` inside the task directory.
2. The final folder structure should look like this:

- Task041_PancreasCysts
    - raw
        - Zenodo_upload
            - data
                - 10000
                    - ct.nii.gz
                    - cyst_mask_grountruth.nii.gz
                    - ..
                - 10001
                    - ...
                - ...

4. Execute `python prepare.py` in the `nndet / tasks / Task041_PancreasCysts / scripts` directory
5. Run `nndet_seg2det 041` to convert the semantic segmentation into instanes

The data is now saved in the correct format. Please folow the instructions from the nnDetection README can be used to train the networks.
