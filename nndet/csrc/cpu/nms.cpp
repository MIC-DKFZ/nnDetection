// Modifications licensed under:
// SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
// SPDX-License-Identifier: Apache-2.0
//
// Parts of this code are from torchvision licensed under
// SPDX-FileCopyrightText: 2016 Soumith Chintala
// SPDX-License-Identifier: BSD-3-Clause

/*  adopted from
    https://github.com/pytorch/vision/blob/master/torchvision/csrc/nms.h on Nov 15 2019
    no cpu support, but could be added with this interface.
*/
//#include "cpu/vision_cpu.h"
#include <torch/types.h>

at::Tensor nms_cuda(const at::Tensor& dets, const at::Tensor& scores, float iou_threshold);

at::Tensor nms(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const double iou_threshold) {
  if (dets.device().is_cuda()) {

    if (dets.numel() == 0) {
      //at::cuda::CUDAGuard device_guard(dets.device());
      return at::empty({0}, dets.options().dtype(at::kLong));
    }
    return nms_cuda(dets, scores, iou_threshold);

  }
  AT_ERROR("Not compiled with CPU support");
  //at::Tensor result = nms_cpu(dets, scores, iou_threshold);
  //return result;
}
