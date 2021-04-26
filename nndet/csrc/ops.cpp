#include <torch/extension.h>
#include "cpu/nms.cpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "NMS C++ and/or CUDA");
}
