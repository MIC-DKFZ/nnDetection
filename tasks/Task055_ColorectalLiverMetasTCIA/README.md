## Colorectal-Liver-Metastases TCIA
**Disclaimer**: We are not the host of the data.
Please make sure to read the requirements and usage policies of the data and **give credit to the authors of the dataset**!

Please read the information from the homepage carefully and follow the rules and instructions provided by the original authors when using the data.
- Homepage: https://www.cancerimagingarchive.net/collection/colorectal-liver-metastases/


## Preparation

0. Follow the installation instructions of nnDetection and create a data directory name `Task055_ColorectalLiverMetasTCIA`.
1. Download the dataset via the official website and place it into a directory called `raw` inside the task directory.
2. The final folder structure should look like this:

- Task055_ColorectalLiverMetasTCIA
    - raw
        - Colorectal-Liver-Metastases
            - image
                - CRLM-CT-1001
                    - [subfolder]
                        - [folder with 'Segmentation' in name and containing the segmentation]
                            - 1-1.dcm
                        - [folder containing the data]
                            - 1-001.dcm
                            - ..
                - CRLM-CT-1002
                - ... 


4. Execute `python prepare.py` in the `nndet / tasks / Task055_ColorectalLiverMetasTCIA / scripts` directory

Note:
Some cases were filtered due to overlapping labels. After manual inspection of a subset, some lesion annotations covered multiple lesions which were already covered by another annotation or were duplicates of the same lesion. The case ids are:

```python
exclude_cases = [
    # overlapping
    "CRLM-CT-1020",
    "CRLM-CT-1026",
    "CRLM-CT-1027",
    "CRLM-CT-1031",
    "CRLM-CT-1037",
    "CRLM-CT-1049",
    "CRLM-CT-1053",
    "CRLM-CT-1057",
    "CRLM-CT-1070",
    "CRLM-CT-1078",
    "CRLM-CT-1080",
    "CRLM-CT-1081",
    "CRLM-CT-1083",
    "CRLM-CT-1088",
    "CRLM-CT-1112",
    "CRLM-CT-1122",
    "CRLM-CT-1127",
    "CRLM-CT-1133",
    "CRLM-CT-1139",
    "CRLM-CT-1145",
    "CRLM-CT-1155",
    "CRLM-CT-1168",
    "CRLM-CT-1173",
    "CRLM-CT-1186",
    "CRLM-CT-1190",
    # missing data (at least with my download :) )
    "CRLM-CT-1183",
]
```

The data is now saved in the correct format. Please folow the instructions from the nnDetection README can be used to train the networks.
