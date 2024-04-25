## PN9
**Disclaimer**: We are not the host of the data.
Please make sure to read the requirements and usage policies of the data and **give credit to the authors of the dataset**!

Please read the information from the homepage carefully and follow the rules and instructions provided by the original authors when using the data.
- Homepage: https://ieeexplore.ieee.org/document/9373930

(Note: we needed to request the data from the authors)

## Preparation

0. Follow the installation instructions of nnDetection and create a data directory name `Task053_PN9`.
1. Download the dataset via the official website and place it into a directory called `raw` inside the task directory.
2. The final folder structure should look like this:

- Task053_PN9
    - raw
        - train
            - image
                - 00001_zoom.npy
                - ...
            - test
                - 00000_zoom.npy
                - ...
            - train_anno.csv
            - test_anno.csv
            - train.txt
            - val.txt
            - test.txt

4. Execute `python prepare.py` in the `nndet / tasks / Task053_PN9 / scripts` directory. Adjust `det_num_threads` according to your CPU and RAM availability (RAM will likely be the major bottleneck since the images as quite large). Remember to reset `det_num_threads` to its original afterwards (usually higher for training networks).

IMPORTANT: the prepare script automatically created multiple split files which follow the published PN9 splits (single split & official val), you need to specify the approriate split file when training the networks via `-o +io_cfg.splits=[split file name]`!

The data is now saved in the correct format. Please folow the instructions from the nnDetection README can be used to train the networks.
