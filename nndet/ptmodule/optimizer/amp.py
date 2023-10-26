# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
from pytorch_lightning.plugins.precision import MixedPrecisionPlugin


class ExposedNativeMixedPrecisionPlugin(MixedPrecisionPlugin):
    def __init__(
        self,
        precision: str,
        device: str,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        init_scale: Optional[float] = None,
    ) -> None:
        if "16-mixed" not in precision:
            raise ValueError(f"Precision {precision} is not supported.")

        if scaler is None and precision == 16:
            scaler = torch.cuda.amp.GradScaler(init_scale=init_scale)

        super().__init__(
            precision=precision,
            device=device,
            scaler=scaler,
        )
