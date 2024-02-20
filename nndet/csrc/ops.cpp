// Modifications licensed under:
// SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
// SPDX-License-Identifier: Apache-2.0
//
// Parts of this code are from torchvision licensed under
// SPDX-FileCopyrightText: 2016 Soumith Chintala
// SPDX-License-Identifier: BSD-3-Clause


#include <torch/extension.h>
#include "cpu/nms.cpp"
#include "cpu/roi_align.cpp"
#include "cuda/ms_deform_attn.h"

#ifdef WITH_CUDA
extern int get_cudart_version();
#endif

std::string get_cuda_version() {
#ifdef WITH_CUDA
  std::ostringstream oss;

  // copied from
  // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/detail/CUDAHooks.cpp#L231
  auto printCudaStyleVersion = [&](int v) {
    oss << (v / 1000) << "." << (v / 10 % 100);
    if (v % 10 != 0) {
      oss << "." << (v % 10);
    }
  };
  printCudaStyleVersion(get_cudart_version());
  return oss.str();
#else
  return std::string("not available");
#endif
}

// similar to
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Version.cpp
std::string get_compiler_version() {
  std::ostringstream ss;
#if defined(__GNUC__)
#ifndef __clang__
  { ss << "GCC " << __GNUC__ << "." << __GNUC_MINOR__; }
#endif
#endif

#if defined(__clang_major__)
  {
    ss << "clang " << __clang_major__ << "." << __clang_minor__ << "."
       << __clang_patchlevel__;
  }
#endif

#if defined(_MSC_VER)
  { ss << "MSVC " << _MSC_FULL_VER; }
#endif
  return ss.str();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ms_deform_attn_forward", &ms_deform_attn_forward, "ms_deform_attn_forward");
  m.def("ms_deform_attn_backward", &ms_deform_attn_backward, "ms_deform_attn_backward");
  m.def("nms", &nms, "NMS C++ and/or CUDA");
  m.def("roi_align", &roi_align, "ROIAlign 3D in c++ and/or cuda");
}
