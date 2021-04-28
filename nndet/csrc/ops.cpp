#include <torch/extension.h>
#include "cpu/nms.cpp"
#include "cpu/roi_align.cpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "NMS C++ and/or CUDA");
  m.def("roi_align", &roi_align, "ROIAlign 3D in c++ and/or cuda");
}
