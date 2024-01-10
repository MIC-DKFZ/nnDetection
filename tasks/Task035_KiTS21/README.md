# KiTS21
**Disclaimer**: We are not the host of the data.
Please make sure to read the requirements and usage policies of the data and **give credit to the authors of the dataset**!

Please read the information from the homepage carefully and follow the rules and instructions provided by the original authors when using the data.You can find all information about KiTS21 at https://kits21.kits-challenge.org/.

## Setup
1. Clone the git repository and run the download script.
2. Create a folder `${det_data}/Task035_KiTS21/raw` (note the upper and lower case letters) and place the data folder in there. The final folder structure should look like this

```
${data_data}
    Task035_KiTS35
        raw
            data
                case_00001
                case_00002
                ...
```

3. [Might not apply to newer versions of the KiTS annotations]: Delete Instance 3 Tumor from case case_00205/segmentations
4. Run the `prepare.py` in the KiTS21 Projects folder of nnDetection.

The data is now converted to the correct format and the instructions from the nnDetection README can be used to train the networks.
