# RibFrac
**Disclaimer**: We are not the host of the data.
Please make sure to read the requirements and usage policies of the data and **give credit to the authors of the dataset**!

Please read the information from the homepage carefully and follow the rules and instructions provided by the original authors when using the data.
- Homepage: https://ribfrac.grand-challenge.org/
- Subtask: Task 1

## Setup
0. Follow the installation instructions of nnDetection and create a data directory name `Task020FG_RibFrac`. We added FG to the ID to indicate that we don't distinguish the different classes. (even if you prepare the data set with classes, the data needs to be placed inside that directory)
1. Follow the instructions and usage policies to download the data and copy the data/labels/csv files to the following locations:
data -> `Task020FG_RibFrac / raw / imagesTr`; labels -> `Task020FG_RibFrac / raw / labelsTr`; csv files -> `Task020FG_RibFrac / raw`
2. Run `python prepare.py` in `projects / Task020FG_RibFrac / scripts` of the nnDetection repository.

Note: If no manual split is created, nnDetection will create a random 5Fold split which we used for results.

The data is now converted to the correct format and the instructions from the nnDetection README can be used to train the networks.
