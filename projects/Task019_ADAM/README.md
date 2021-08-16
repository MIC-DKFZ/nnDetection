# ADAM
**Disclaimer**: We are not the host of the data.
Please make sure to read the requirements and usage policies of the data and **give credit to the authors of the dataset**!

Please read the information from the homepage carefully and follow the rules and instructions provided by the original authors when using the data.
- Homepage: http://adam.isi.uu.nl/
- Subtask: Task 1

## Setup
0. Follow the installation instructions of nnDetection and create a data directory name `Task019FG_ADAM`. We added FG to the ID to indicate that unruptered and ruptured aneurysms are treated as one i.e. we are running a foreground vs background detection without distinguishing the classes.
1. Follow the instructions and usage policies to download the data and place the data into `Task019FG_ADAM / raw / ADAM_release_subjs`
2. Run `python prepare.py` in `projects / Task019_ADAM / scripts` of the nnDetection repository.
3. Run `python split.py` in `projects / Task019_ADAM / scripts` of the nnDetection repository.
4. [Info]: The provided instructions will automatically create a patient stratified random split. We used a random split for our challenge submission. By renaming the provided split file in the `preprocessed` folders, nnDetection will automatically create a random split.

The data is now converted to the correct format and the instructions from the nnDetection README can be used to train the networks.

## Submission
The submission folder contains the scripts used for our leaderboard submissions.
Before building the Docker Image the model directory needs to be copied to the the submission folder to be detected by the docker context. Make sure to adapt the name of the nndetection base container to the name you installed it with (/ the current nndetection version).
Before submitting make sure to run a test prediction on the training set to double check it :)
