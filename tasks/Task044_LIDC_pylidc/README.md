## LIDC
**Disclaimer**: We are not the host of the data.
Please make sure to read the requirements and usage policies of the data and **give credit to the authors of the dataset**!

Please read the information from the homepage carefully and follow the rules and instructions provided by the original authors when using the data.
- Homepage: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254

## PyLIDC preparations

1. Create a new env (tested with python 3.8)
2. Install pylidc `pip install git+https://github.com/mibaumgartner/pylidc.git` (as of writing this the original repo does not work with recent numpy versions due to `np.int` usage. If that is fixed you can also use `pip install pylidc`)
3. Configure pylidc according to the official documentation
4. Create folder `Task044_LIDC_pylidc` and download tcia data into the folder.

The folder structure should now look like this:
- Task044_LIDC_pylidc
    - TCIA_LIDC-IDRI_20200921
        - LIDC-IDRI
            - LIDC-IDRI-XXXX
            - ...

5. Go into the task directory of nndetection `Task044_LIDC_pylidc`.
6. Execute `python prepare.py` to run the preparation of the binary data set, run `python prepare.py --malignant` to prepare the two class problem.
7. Create split `nndet_cv_split 044 --with_patients` and/or `nndet_cv_split 045 --with_patients`
8. Continue with nnDetection.


## Manual Grouping Information
- lidc0055: Annotation id `588` was added to two groups
- lidc0092: Annoation id `844` and `845` were combined into a single lesion 
- lidc0204: Annotation id `1720`, `1721` and `1722` were combined into a single lesion
- lidc0252: Annotation id `1984` and `1985` were combined into a single lesion
- lidc0366: Annotation id `2642` was added to two groups
- lidc0404: Annotation id `2928` was added to two groups
- lidc0608: Annotation id `4312` was combined into a single lesion
- lidc0815: Annotations merged into one lesion
- lidc0863: Annotations merged into one lesion
- lidc0865: Annotations merged into one lesion
- lidc0942: Annotation added to two groups
- lidc332_1: Annotation id `2398` added to two groups
