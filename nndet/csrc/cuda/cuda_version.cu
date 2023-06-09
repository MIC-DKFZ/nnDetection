/*! Modifications licensed under:
SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
SPDX-License-Identifier: Apache-2.0

Parts of this code are from detrex licensed under
SPDX-FileCopyrightText: 2022, The IDEA Authors
SPDX-License-Identifier: Apache-2.0 */

#include <cuda_runtime_api.h>

int get_cudart_version() {
  return CUDART_VERSION;
}
