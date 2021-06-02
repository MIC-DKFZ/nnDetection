# Decathlon
**Disclaimer**: We are not the host of the data.
Please make sure to read the requirements and usage policies of the data and **give credit to the authors of the dataset**!

Please read the information from the homepage carefully and follow the rules and instructions provided by the original authors when using the data.
- Homepage: http://medicaldecathlon.com/

## Setup
0. Follow the installation instructions of nnDetection and create the data directories for the intended tasks, e.g. `Task003_Liver`.
1. Follow the instructions and usage policies to download the data and place the images, labels and dataset.json files inside the raw folder of the respective tasks, e.g. imagesTr -> `Task003_Liver / raw / imagesTr`, labelsTr -> `Task003_Liver / raw / labelsTr` and dataset.json -> `Task003_Liver / raw / dataset.json`
2. Run `python prepare.py [tasks]` in `projects / Task001_Decathlion / scripts` of the nnDetection repository, e.g. to prepare all tasks: `python prepare.py Task003_Liver Task007_Pancreas Task008_HepaticVessel Task010_Colon`
3. Download labels from [here](https://zenodo.org/record/4876497#.YLSudzYzYeY) and replace `labelsTr` / `labelsTs` in the splitted folder with the downloaded ones.

The data is now converted to the correct format and the instructions from the nnDetection README can be used to train the networks.
