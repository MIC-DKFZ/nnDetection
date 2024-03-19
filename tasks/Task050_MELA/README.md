## Data Preparation

- Create `Task030_Mela/raw_splitted/imagesTr` and place the downloaded data into there.

The result should look like this:

```
{nndet_data}
    - Task030_Mela
        - mela_train_val_annotations.csv
        - raw_splitted
             - imagesTr
                 - mela_0001.nii.gz
                 - ...
```

- Run the prepare script from this folder `python prepare.py`


## Notes
Since not all sizes are cleanly divisable by 2 there is ~1 pixel uncertainty in back and forth conversion.
