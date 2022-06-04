// Modifications licensed under:
// SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
// SPDX-License-Identifier: Apache-2.0
//
// Parts of this code are from torchvision licensed under
// SPDX-FileCopyrightText: 2016 Soumith Chintala
// SPDX-License-Identifier: BSD-3-Clause


#include <torch/extension.h>
#include "cpu/nms.cpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "NMS C++ and/or CUDA");
}
