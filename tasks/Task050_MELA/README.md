## KiPA22
**Disclaimer**: We are not the host of the data.
Please make sure to read the requirements and usage policies of the data and **give credit to the authors of the dataset**!

Please read the information from the homepage carefully and follow the rules and instructions provided by the original authors when using the data.
- Homepage: https://mela.grand-challenge.org/


## Preparation

0. Follow the installation instructions of nnDetection and create a data directory name `Task050_MELA`.
1. Download the dataset via the official website and place it into a directory called `raw` inside the task directory. Training images go into `imagesTr` and the official validation set is used as the test set in `imagesTs`.
2. The final folder structure should look like this:

```
{nndet_data}
    - Task050_MELA
        - raw
            - imagesTr
                - mela_0001.nii.gz
            - imagesTs
                - mela_0771.nii.gz
        - mela_train_val_annotations.csv
```

4. Execute `python prepare.py` in the `nndet / tasks / Task050_MELA / scripts` directory

The data is now saved in the correct format. Please folow the instructions from the nnDetection README can be used to train the networks.

## Notes
Since not all sizes are cleanly divisable by 2 there is ~1 pixel uncertainty in back and forth conversion of the bounding box annotations wrt. to the orignal csv file.
