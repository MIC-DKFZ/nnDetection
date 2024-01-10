# PICAI
**Disclaimer**: We are not the host of the data.
Please make sure to read the requirements and usage policies of the data and **give credit to the authors of the dataset**!

Please read the information from the homepage carefully and follow the rules and instructions provided by the original authors when using the data.
- Homepage: https://pi-cai.grand-challenge.org/

Note: in contrast to the original PICAI challenge we use run this an object detection problem and not as an instance segmentation.

## Setup Data
Large parts of this installation guide are based on the orignal instructions found at: https://github.com/DIAGNijmegen/picai_baseline . If you consider this helpful please consider leaving them a star on their github repo :) 

Installed version during our experiments:
```text
picai-baseline                0.8.2
picai-eval                    1.4.4
picai-prep                    2.1.2
```

0. Follow the installation instructions of nnDetection and create a data directory name `Task036_PICAI`.
1. Download the data and place it the images in `$det_data/Task036_PICAI/raw/input/images`.
2. Download the labels and place them in `$det_data/Task036_PICAI/raw/input/labels`.
3. Create a directory `workdir` in `$det_data/Task036_PICAI/raw`

The result should look like this:
```text
$det_data/Task036_PICAI/raw/
    - input
        - images
            - ... PICAI folders with data, in total 1476 patient folders
        - labels
            - ... pical repo with labels
    - workdir
```

4. Install picai baseline (and potentially check out necessary version)

```bash
git clone https://github.com/DIAGNijmegen/picai_baseline
pip install -e ./picai_baseline
```

5. Convert data into nnDetection format

```bash
python picai_baseline/src/picai_baseline/prepare_data.py --workdir $det_data/Task036_PICAI/raw/workdir --inputdir $det_data/Task036_PICAI/raw/input --task Task036_PICAI --labelsdir labels
```

```bash
python -m picai_prep nnunet2nndet \
    --input $det_data/Task036_PICAI/raw/workdir/nnUNet_raw_data/Task036_PICAI \
    --output $det_data/Task036_PICAI
```

6. Create nnDetection split

```bash
python -m picai_baseline.splits.picai_nnunet --output "${det_data}/Task036_PICAI/preprocessed"
```

```bash
python scripts/prepare.py
```

The data is now converted to the correct format and the instructions from the nnDetection README can be used to train the networks.
