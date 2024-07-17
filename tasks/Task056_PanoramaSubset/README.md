## Panorama
**Disclaimer**: We are not the host of the data.
Please make sure to read the requirements and usage policies of the data and **give credit to the authors of the dataset**!

Please read the information from the homepage carefully and follow the rules and instructions provided by the original authors when using the data.
- Homepage: https://panorama.grand-challenge.org/

Note: We only use a subset of the provided panorama data to avoid data leakage and high quality annotations: we exclude all cases of the MSD Pancreas dataset and cases which only have automatic labels. This happens automatically when the images get prepared by the script.

## Preparation

0. Follow the installation instructions of nnDetection and create a data directory name `Task056_PanoramaSubset`.
1. Download the dataset via the official website and place it into a directory called `raw` inside the task directory.
2. Download the panorama data into the raw folder.
3. The final folder structure should look like this:

- Task056_PanoramaSubset
    - raw
        - imagesTr
        - labelsTr
        - panorama_labels
            - automatic_labels
            - manual_labels
            - clinical_information.xlsx

4. Execute `python prepare.py` in the `nndet / tasks / Task056_PanoramaSubset / scripts` directory
5. Run `python convert_seg2_det_panorama.py 056` to convert the semantic segmentations into instance segmentations via connected components. 
