# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Union
from batchgenerators.utilities.file_and_folder_operations import join, isdir, isfile, load_json, subfiles, save_json

from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
nnUNet_raw=os.environ.get('nnUNet_raw')
nnUNet_preprocessed=os.environ.get('nnUNet_preprocessed')
nnUNet_results=os.environ.get('nnUNet_results')


def move_plans(source_plan, target_plan):

    #load both plans
    #copy parts from MT to nndetection plan:
    MT_plan=load_json(source_plan)
    nndet_plan=load_json(target_plan)

    print(MT_plan.keys())
    nndet_plan['architecture']['backbone_architecture_class_name']=MT_plan['configurations']['3d_fullres']['architecture']['network_class_name']

    keys_to_remove=['strides', 'kernel_sizes', 'features_per_stage']
    nndet_plan['architecture']['backbone_arch_init_kwargs'] = MT_plan['arch_kwargs'] #but not all the parts

    for key in keys_to_remove:
        if key in nndet_plan['architecture']['backbone_arch_init_kwargs']:
            del nndet_plan['architecture']['backbone_arch_init_kwargs'][key]

    nndet_plan['architecture']['conv_kernels']=MT_plan['kernel_sizes']
    nndet_plan['architecture']['out_channels']=MT_plan['features_per_stage']
    nndet_plan['architecture']['strides']=MT_plan['strides']
    nndet_plan['architecture']['backbone_arch_init_kwargs_req_import']
    nndet_plan['in_channels']=len(MT_plan['foreground_intensity_properties_per_channel'].keys())
    nndet_plan['patch_size']=[128,128,128]

    save_json(nndet_plan, f'ResEnc{target_plan}')


if __name__=='__main__':
    source_plan='/home/k979n/cluster-data_all/c306h/nnUNetV2/nnUNet_preprocessed/Dataset900_MT_resenc_katharina_zscore/nnUNetResEncUNetLPlansIso1x1x1.json'
    target_plan='/home/k979n/cluster-data/nndet_data/Task007_Pancreas/preprocessed/StaticPlannerTotalSeg_spacing_1_ps_128_3d.json'
    move_plans(source_plan, target_plan)
