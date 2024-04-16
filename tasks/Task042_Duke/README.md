## Breast MRI Primary Lesion detection
**Disclaimer**: We are not the host of the data.
Please make sure to read the requirements and usage policies of the data and **give credit to the authors of the dataset**!

Please read the information from the homepage carefully and follow the rules and instructions provided by the original authors when using the data.
- Paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6134102/
- Project: https://sites.duke.edu/mazurowski/resources/breast-cancer-mri-dataset/
- Data: https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/


## Preparation

0. Follow the installation instructions of nnDetection and create a data directory name `Task042_Duke`.
1. Install `pydicom` and `openpyxl` which are required to read dcm and xlsx files (tested with pydicom==2.4.4 and openpyxl==3.1.2 when writing this, both packages can be pip installed)
2. Download the dataset via the official website and place it into a directory called `raw` inside the task directory.
3. The final folder structure should look like this:

- Task042_Duke
    - Duke-Breast-Cancer-MRI
        - Breast_MRI_001
            - [subfolder]
                - [sequences]
                    - 1-001.dcm
                    - ...
        - Breast_MRI_002
        - ...
    - Annotation_Boxes.xlsx
    - Breast-Cancer-MRI-filepath_filename-mapping.xlsx
    

4. Execute `export det_num_threads=4 && python prepare.py` in the `nndet / tasks / Task042_Duke / scripts` directory. Adjust `det_num_threads` according to your CPU and RAM availability (RAM will likely be the major bottleneck since the images as quite large). Remember to reset `det_num_threads` to its original afterwards (usually higher for training networks).

#TODO: split test?

Note:
Some cases were filtered due to missing `post_3` sequence.

```python
exclude_cases = [
    "Breast_MRI_103",
    "Breast_MRI_164",
    "Breast_MRI_253",
    "Breast_MRI_282",
    "Breast_MRI_700",
    "Breast_MRI_728",
    "Breast_MRI_801",
    "Breast_MRI_893",
]
```

The data is now saved in the correct format. Please folow the instructions from the nnDetection README can be used to train the networks.
