// Parts of this code are from torchvision licensed under
// SPDX-FileCopyrightText: 2016 Soumith Chintala
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#define CUDA_1D_KERNEL_LOOP(i, n)                                \
  for (int i = (blockIdx.x * blockDim.x) + threadIdx.x; i < (n); \
       i += (blockDim.x * gridDim.x))
