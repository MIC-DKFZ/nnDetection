## LNDb
**Disclaimer**: We are not the host of the data.
Please make sure to read the requirements and usage policies of the data and **give credit to the authors of the dataset**!

Please read the information from the homepage carefully and follow the rules and instructions provided by the original authors when using the data.
- Homepage: https://lndb.grand-challenge.org/


## Preparation

0. Follow the installation instructions of nnDetection and create a data directory name `Task054_LNDb`.
1. Download the dataset via the official website and place it into a directory called `raw` inside the task directory.
2. Place the images into a `data` folder and the segmentations into a `masks` folder, `trainNodules_gt.csv` should be located in `raw` as well.
3. The final folder structure should look like this:

- Task054_LNDb
    - raw
        - trainNodules_gt.csv
        - data
            - LNDb-0001.mhd
            - LNDb-0001.raw
            - LNDb-0002.mhd
            - LNDb-0002.raw
            - ...
        - masks
            - LNDb-0001_rad1.mhd
            - LNDb-0001_rad1.raw
            - LNDb-0001_rad2.mhd
            - LNDb-0001_rad2.raw
            - LNDb-0001_rad3.mhd
            - LNDb-0001_rad3.raw
            - ...

4. Execute `python prepare.py` in the `nndet / tasks / Task054_LNDb / scripts` directory

The data is now saved in the correct format. Please folow the instructions from the nnDetection README can be used to train the networks.
