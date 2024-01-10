#!/bin/bash
. /activate

# bring adam file into internal structure
cp /input/pre/struct_aligned.nii.gz ${det_data}/${ENVTASK}/raw_splitted/imagesTs/case_0000.nii.gz
cp /input/pre/TOF.nii.gz ${det_data}/${ENVTASK}/raw_splitted/imagesTs/case_0001.nii.gz

nndet_predict ${ENVTASK} ${ENVMODEL} -f -1
cp ${det_models}/${ENVTASK}/${ENVMODEL}/${ENVFOLD}/test_predictions/* ${det_results}

python convert.py ${det_results}
cp /opt/results/result.txt /output
