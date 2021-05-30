import copy
import os
from pathlib import Path
from typing import Any, List, Optional, Union
from nndet.utils.tensor import to_device

import torch

from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule


class EMAWeightsCB(Callback):
    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        beta: float = 0.9998,
        ema_eval: bool = True,
        dirpath: Optional[os.PathLike] = None,
):
        """
        Callback to compute exponential moving average of weights

        Args:
            device: device to save shadow weights on. Defaults to None.
            beta: Beta of EMA. Defaults to 0.9998.
            ema_eval: Replace current model weights with EMA weights during
                evaluation. Defaults to True.
            dirpath: if povided, save EMA weights here
                (only contains the statedict!)

        Notes:
            This is only a prototype.
            Multi GPU and ema_eval are not supported.
        """
        self.ema: Optional[EMAWeights] = None
        self.device = device
        self.beta = beta
        self.ema_eval = ema_eval
        if self.ema_eval:
            raise NotImplementedError("Evaluation with EMA weights is not implemented yet.")
        self.dirpath = Path(dirpath) if dirpath is not None else dirpath

    def on_train_start(self, trainer, pl_module):
        # init EMA
        self.ema = EMAWeights(
            module=pl_module,
            device=self.device,
            beta=self.beta,
        )

    def on_train_batch_end(self,
                           trainer,
                           pl_module,
                           outputs,
                           batch,
                           batch_idx,
                           dataloader_idx,
                           ):
        # on_train_batch_end -> update weights
        self.ema.add(pl_module)

    # def on_validation_epoch_start(self,
    #                               trainer,
    #                               pl_module: LightningModule,
    #                               ) -> None:
    #     # on_validation_epoch_start -> optionally replace with shadowed weights
    #     if self.ema_eval:
    #         pass
    #     return super().on_validation_epoch_start(trainer, pl_module)

    # def on_validation_epoch_end(self,
    #                             trainer,
    #                             pl_module: LightningModule,
    #                             outputs: List[Any],
    #                             ) -> None:
    #     # on_validation_epoch_end -> optionally replace with current weights
    #     if self.ema_eval:
    #         pass
    #     return super().on_validation_epoch_end(trainer, pl_module, outputs)

    def on_train_end(self, trainer, pl_module: LightningModule) -> None:
        # on_train_end -> save ema weights
        if self.dirpath is not None:
            torch.save({"state_dict": self.ema.get_state_dict()}, str(self.dirpath / "model_ema.ckpt"))
        return super().on_train_end(trainer, pl_module)


class EMAWeights:
    def __init__(
        self,
        module: torch.nn.Module,
        device: Optional[Union[str, torch.device]] = None,
        beta: float = 0.9998,
):
        """
        Shadow model parameters keep track of an exponential moving average of
        the model weights

        Args:
            module: module to shadow parameters of
            device: Move shadowed parameters to this device.
                If None, the device is not changed. Defaults to None.
            beta: Beta of moving average. Defaults to 0.9998.
        """
        self.module_state_dict = copy.deepcopy(module.state_dict())
        self.device = torch.device(device) if isinstance(device, str) else device

        if self.device is not None:
            self.module_state_dict = to_device(self.module_state_dict, device=self.device, detach=True)

        self.beta = beta

    def add(self, module: torch.nn.Module) -> None:
        """
        Add new weights to EMA module

        Args:
            module: module with updated weights
        """
        for (nw, weight), (nu, udpate) in  zip(self.module_state_dict.items(), module.state_dict().items()):
            assert nw == nu
            _update = udpate.detach().clone()

            if self.device is not None:
                _update = _update.to(self.device)

            self.module_state_dict[nw] = self.beta * weight + (1 - self.beta) * _update

    def get_state_dict(self) -> dict:
        """
        Return state dict of shadowed parameters

        Returns:
            dict: state dict of EMA weights
        """
        return self.module_state_dict


class EMA:
    def __init__(self, beta: float = 0.9, bias_correction: bool = True):
        """
        Exponentially weighted moving average
        new_cache = beta * cache + (1 - beta) * new_val
        Approximatley averages (1 - beta)^(-1) values

        Args:
            beta: weights for averaging
            bias_correction: applies bias correction
        """
        self.beta = beta
        self.cache: float = 0.

        self.bias_correction = bias_correction
        self.t = 0

    def add(self, val):
        """
        Add new value

        Args:
            val: new value to add
        """
        self.cache = self.beta * self.cache + (1 - self.beta) * float(val)
        self.t = min(self.t + 1, 100000)  # prevent overflow

    def get(self) -> float:
        """
        Retrive vurrent value

        Returns:
            float: current EMA
        """
        if self.bias_correction and self.t > 0:
            return (self.cache / (1 - pow(self.beta, self.t)))
        else:
            return self.cache
