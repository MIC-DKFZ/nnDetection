## TCIA Lymph Nodes AQIIDCNM
**Disclaimer**: We are not the host of the data.
Please make sure to read the requirements and usage policies of the data and **give credit to the authors of the dataset**!

Please read the information from the homepage carefully and follow the rules and instructions provided by the original authors when using the data.
- Homepage: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=19726546

```
Roth, H. R., Lu, L., Seff, A., Cherry, K. M., Hoffman, J., Wang, S., Liu, J., Turkbey, E., & Summers, R. M. (2014). A New 2.5D Representation for Lymph Node Detection Using Random Sets of Deep Convolutional Neural Network Observations. In Medical Image Computing and Computer-Assisted Intervention – MICCAI 2014 (pp. 520–527). Springer International Publishing. https://doi.org/10.1007/978-3-319-10404-1_65
```

```
A Seff, L Lu, A Barbu, H Roth, HC Shin, RM Summers. Leveraging Mid-Level Semantic Boundary Cues for Automated Lymph Node Detection. Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015, 53-61 (http://link.springer.com/chapter/10.1007/978-3-319-24571-3_7)
```

## Preparation
Note: the preparation is the same as for Task 025 but the code was cleaned up :)

0. Follow the installation instructions of nnDetection and create a data directory name `Task046_TCIAMedLymphNodesAQIIDCNM`.
1. Download the TCIA Lymph Node data set via the official website and place it into a directory called `raw` inside the task directory.
2. Download the lymph node masks and place it into the `raw` directory as well.
3. The final folder structure should look like this:

- Task046_TCIAMedLymphNodesAQIIDCNM
    - raw
        - TCIA_CT_Lymph_Nodes_[some date]
            - CT Lymph Nodes
                - ABD_XXXXX
                - ...
                - MED_XXXXX
                - ...
        - MED_ABD_LYMPH_MASKS

4. Execute `python prepare.py` in the `nndet / tasks / Task046_TCIAMedLymphNodesAQIIDCNM / scripts` directory -> this will create two tasks, one for mediastinal lesions and one for abdominal lesions

The data is now converted to the correct format and the instructions from the nnDetection README can be used to train the networks.
