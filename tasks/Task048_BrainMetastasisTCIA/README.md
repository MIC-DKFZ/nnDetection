## Brain Metastasis
**Disclaimer**: We are not the host of the data.
Please make sure to read the requirements and usage policies of the data and **give credit to the authors of the dataset**!

Please read the information from the homepage carefully and follow the rules and instructions provided by the original authors when using the data.
- Homepage: https://www.cancerimagingarchive.net/collection/pretreat-metstobrain-masks/


## Preparation

0. Follow the installation instructions of nnDetection and create a data directory name `Task048_BrainMetastasisTCIA`.
1. Download the dataset via the official website and place it into a directory called `raw` inside the task directory.
2. The final folder structure should look like this:

- Task048_BrainMetastasisTCIA
    - raw
        - Pretreat-MetsToBrain-Masks
            - BraTS-MET-00086-000
                - BraTS-MET-00086-000-seg.nii.gz
                - BraTS-MET-00086-000-t1c.nii.gz
                - BraTS-MET-00086-000-t1n.nii.gz
                - BraTS-MET-00086-000-t2f.nii.gz
                - BraTS-MET-00086-000-t2w.nii.gz
            - BraTS-MET-00089-000
            - ...

4. Execute `python prepare.py` in the `nndet / tasks / Task048_BrainMetastasisTCIA / scripts` directory
5. Run `nndet_seg2det 048` to convert the semantic segmentation into instances
6. Run `nndet_test_data_split 048 --size 0.3 --stratify` to split a test set

The data is now saved in the correct format. Please folow the instructions from the nnDetection README can be used to train the networks.
