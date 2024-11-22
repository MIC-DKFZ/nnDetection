# Luna16
**Disclaimer**: We are not the host of the data.
Please make sure to read the requirements and usage policies of the data and **give credit to the authors of the dataset**!

Please read the information from the homepage carefully and follow the rules and instructions provided by the original authors when using the data.
- Homepage: https://luna16.grand-challenge.org/Home/

## Setup
0. Follow the installation instructions of nnDetection and create a data directory name `Task016_Luna`.
1. Follow the instructions and usage policies to download the data and place all the subsets into `Task016_Luna / raw`
2. Run `python prepare.py` in `projects / Task016_Luna / scripts` of the nnDetection repository.

The data is now converted to the correct format and the instructions from the nnDetection README can be used to train the networks.

## Evaluation 9-0-1 Split

Notes:
- since Luna is a 10 Fold cross validation, all 10 folds need to be run
- all runs should be run with the `--sweep` option and consolidation should be performed via the `--num_folds 10 --no_model -c copy` since we are not planning to predict a separate test set.

1. Run `python prepare_eval_cpm.py [model_name]` to convert the predictions to the Luna format.
Note: The script needs access to the raw_splitted images.
2. Download and run the luna evaluation script.

## Evaluation 8-1-1 Split

- since Luna is a 10 Fold cross validation, all 10 folds need to be run
- training needs to be executed with the `-o +io_cfg.splits=splits_final811` to select the correct split
- prediction needs to be performed via the `nndet_predict_test_split` of nnDetection.

An example script to train and predict Fold 0 is shown below.
``` bash
nndet_train 016 def_detr_v002 0 -o +io_cfg.splits=splits_final811 exp.tag=_split811 --sweep
nndet_predict_test_split 016 BoxDeformableDETRV002_D3V002_3d_split811 0 --load_models last
```

1. Run `python prepare_eval_cpm_811.py [model_name]` to convert the predictions to the Luna format.
Note: The script needs access to the raw_splitted images.
2. Download and run the luna evaluation script.
