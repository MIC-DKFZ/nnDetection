# LIDC
**Disclaimer**: We are not the host of the data.
Please make sure to read the requirements and usage policies of the data and **give credit to the authors of the dataset**!

Please read the information from the homepage carefully and follow the rules and instructions provided by the original authors when using the data.
- Homepage: https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI

## Setup MIC LIDC Data preprocessing
0. Follow https://github.com/MIC-DKFZ/LIDC-IDRI-processing to convert the LIDC data into a simpler format. 
1. Follow the installation instructions of nnDetection and create a data directory name `Task012_LIDC`.
2. Place the `data_nrrd` folder and `characteristics.csv` into `Task012_LIDC / raw`
3. Run `python prepare_mic.py` in `projects / Task012_LIDC / scripts` of the nnDetection repository.
4. Copy the `splits_final.pkl` from `projects / Task012_LIDC` into the preprocessed folder of the data (if the preprocessing wasn't run until now, it is nesseary to manually create the folder)

The data is now converted to the correct format and the instructions from the nnDetection README can be used to train the networks.

## Setup PyLIDC
**Coming Soon**
