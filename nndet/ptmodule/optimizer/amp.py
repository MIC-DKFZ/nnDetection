from typing import Optional, Union

import torch
from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin


class ExposedNativeMixedPrecisionPlugin(NativeMixedPrecisionPlugin):
    def __init__(
        self,
        precision: Union[str, int],
        device: str,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        init_scale: Optional[float] = None,
    ) -> None:
        if scaler is None and precision == 16:
            scaler = torch.cuda.amp.GradScaler(init_scale=init_scale)

        super().__init__(
            precision=precision,
            device=device,
            scaler=scaler,
        )
