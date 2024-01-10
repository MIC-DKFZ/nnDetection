# ProstateX
**Disclaimer**: We are not the host of the data.
Please make sure to read the requirements and usage policies of the data and **give credit to the authors of the dataset**!

Please read the information from the homepage carefully and follow the rules and instructions provided by the original authors when using the data.
- Data: https://wiki.cancerimagingarchive.net/display/Public/SPIE-AAPM-NCI+PROSTATEx+Challenges
- Masks: https://github.com/rcuocolo/PROSTATEx_masks

## Setup
0. Follow the installation instructions of nnDetection and create a data directory name `Task021_ProstateX`.
1. Download the data and labels and place them in the following structure:

```text
{det_data}
    Task021_ProstateX
        raw
            ktrains
            ProstateX
            ProstateX-TrainingLesionInformationv2
            rcuocolo-PROSTATEx_masks-e344452
```

We used the masks from the git hash e3444521e70cd5e8d405f4e9a6bc08312df8afe7 for our experiments.
For training only the T2 masks and T2,ADC and bVal high were used for training (no KTrains).
If you intend to use the Ktrains sequence, simply add it to the `dataset.json` file, the data is already prepared by the script.

2. Run `python prepare.py` in `projects / Task021_ProstateX / scripts` of the nnDetection repository.

The data is now converted to the correct format and the instructions from the nnDetection README can be used to train the networks.

Note: Since ProstateX only contains a fairly small number of clinically significant lesions and we used a 30% test split, we observed a fairly high variance in the performance of our runs.
