## MRA Aneurysms
**Disclaimer**: We are not the host of the data.
Please make sure to read the requirements and usage policies of the data and **give credit to the authors of the dataset**!

Please read the information from the homepage carefully and follow the rules and instructions provided by the original authors when using the data.
- Paper: https://link.springer.com/article/10.1007/s12021-022-09597-0
- Github: https://github.com/connectomicslab/Aneurysm_Detection


## Preparation

0. Follow the installation instructions of nnDetection and create a data directory name `Task052_MRAAneurysms`.
1. Download the dataset via the official website and place it into a directory called `raw` inside the task directory.
2. Down the github repository into the raw folder via `git clone https://github.com/connectomicslab/Aneurysm_Detection.git`
3. The final folder structure should look like this:

- Task052_MRAAneurysms
    - raw
        - Aneurysm_Detection
        - derivates
            - manual_masks
                - sub-XXX
        - sub-XXX
        - sub-YYY
        - sub-ZZZ

4. Execute `export det_num_threads=4 && python prepare.py` in the `nndet / tasks / Task052_MRAAneurysms / scripts` directory. Adjust `det_num_threads` according to your CPU and RAM availability (RAM will likely be the major bottleneck since the images are quite large). Remember to reset `det_num_threads` to its original afterwards (usually higher for training networks).

The data is now saved in the correct format. Please folow the instructions from the nnDetection README can be used to train the networks (except the cross-validation split!). The provided script creates a custom cross-validation split to account for the manually annotated cases, do NOT use nndetection to overwrite the created split file from here!  
