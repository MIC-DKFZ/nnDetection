// Modifications licensed under:
// SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
// SPDX-License-Identifier: Apache-2.0

// Parts of this code are from transoar licensed under
// SPDX-FileCopyrightText: 2022, Bastian Wittmann
// SPDX-License-Identifier: Apache-2.0
//
// Parts of this code are from detrex licensed under
// SPDX-FileCopyrightText: 2022, The IDEA Authors
// SPDX-License-Identifier: Apache-2.0
//
// Parts of this code are from Deformable-DETR licensed under
// SPDX-FileCopyrightText: 2020, SenseTime
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

at::Tensor
ms_deform_attn_cpu_forward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step)
{
    AT_ERROR("Not implement on cpu");
}

std::vector<at::Tensor>
ms_deform_attn_cpu_backward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int im2col_step)
{
    AT_ERROR("Not implement on cpu");
}
