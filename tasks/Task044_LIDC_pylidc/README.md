## LIDC
**Disclaimer**: We are not the host of the data.
Please make sure to read the requirements and usage policies of the data and **give credit to the authors of the dataset**!

Please read the information from the homepage carefully and follow the rules and instructions provided by the original authors when using the data.
- Homepage: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254

```
Armato SG 3rd, McLennan G, Bidaut L, McNitt-Gray MF, Meyer CR, Reeves AP, Zhao B, Aberle DR, Henschke CI, Hoffman EA, Kazerooni EA, MacMahon H, Van Beeke EJ, Yankelevitz D, Biancardi AM, Bland PH, Brown MS, Engelmann RM, Laderach GE, Max D, Pais RC, Qing DP, Roberts RY, Smith AR, Starkey A, Batrah P, Caligiuri P, Farooqi A, Gladish GW, Jude CM, Munden RF, Petkovska I, Quint LE, Schwartz LH, Sundaram B, Dodd LE, Fenimore C, Gur D, Petrick N, Freymann J, Kirby J, Hughes B, Casteele AV, Gupte S, Sallamm M, Heath MD, Kuhn MH, Dharaiya E, Burns R, Fryd DS, Salganicoff M, Anand V, Shreter U, Vastagh S, Croft BY.  The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A completed reference database of lung nodules on CT scans. Medical Physics, 38: 915--931, 2011. DOI: https://doi.org/10.1118/1.3528204
```

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

The data is now converted to the correct format and the instructions from the nnDetection README can be used to train the networks.


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
