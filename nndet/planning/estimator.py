"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import gc
import subprocess as sp
import math
import copy

from abc import ABC, abstractmethod
from functools import partial, reduce
from typing import Sequence, Union, Callable, Tuple
from contextlib import contextmanager
from loguru import logger

from nndet.arch.abstract import AbstractModel

"""
This is just a first prototype to estimate VRAM consumption for different GPUs
I hope to update this soon.
"""

def b2mb(x): return x / (2**20)
def mb2b(x): return x * (2**20)


# remove 11mb from target memory to have a little wiggle room
# (sometimes that amount was blocked on my GPU even though nothing was running)
ARCHS = {
    "RTX2080TI": 11523260416 - int(mb2b(11))
}

# this is just an esitmation ... probably depend on the cuda version too
CUDA_CONTEXT = {
    "none": 0,
    "RTX2080TI": int(mb2b(910))
}


class MemoryEstimator(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = None

    @abstractmethod
    def estimate(self, *args, **kwargs):
        raise NotImplementedError


class MemoryEstimatorDetection(MemoryEstimator):
    def __init__(self,
                 target_mem: Union[float, str] = "RTX2080TI",
                 gpu_id: int = 0,
                 context: Union[float, str] = "RTX2080TI",
                 offset: int = mb2b(768),
                 batch_size: int = 1,
                 mixed_precision: bool = True):
        """
        Estimate memory needed for training a specific network

        Args:
            target_mem: memory of target card (can be higher than
                currently used card). Defaults to "RTX2080TI".
            gpu_id: GPU id to use for estimation. Defaults to 0.
            context: Memory which is reserved for cuda context. Depends on
                CUDA version and GPU. Defaults to "RTX2080TI".
            offset: Additional safety offset because memory consuption
                can fluctuate a bit during training. Defaults to 1024mb.
            batch_size: batch size to use for estimation. Defaults to 1.
        """
        super().__init__()
        if isinstance(context, str):
            self.context = CUDA_CONTEXT[context]
        else:
            self.context = context
        
        self.offset = offset
        self.block_mem_tensor = None

        if isinstance(target_mem, str):
            self.target_mem = ARCHS[target_mem]
        else:
            self.target_mem = target_mem
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        self.mixed_precision = mixed_precision

    def create_offset_tensor_on_GPU(self) -> torch.Tensor:
        device = f"cuda:{self.gpu_id}"
        tensor_mem = torch.rand(1, dtype=float, requires_grad=False, device=device).element_size()
        return torch.rand(math.ceil(self.offset / tensor_mem), dtype=float,
                          requires_grad=False, device=device)

    def estimate(self,
                 min_shape: Sequence[int],
                 target_shape: Sequence[int],
                 network: AbstractModel,
                 optimizer_cls: Callable = torch.optim.Adam,
                 in_channels: int = None,
                 num_instances: int = 1,
                 ) -> Tuple[int, bool]:
        if in_channels is not None:
            min_shape = [in_channels, *min_shape]
            target_shape = [in_channels, *target_shape]

        # all_mem - reserved_mem[misc + context] + context
        available_mem = torch.cuda.get_device_properties(self.gpu_id).total_memory - \
            smi_memory_allocated(self.gpu_id) + self.context
        logger.info(
            f"Found available gpu memory: {available_mem} bytes / {b2mb(available_mem)} mb "
            f"and estimating for {self.target_mem} bytes / {b2mb(self.target_mem)}")

        # if available_mem >= self.target_mem:
        res = self._estimate_mem_available(
            min_shape=min_shape,
            target_shape=target_shape,
            network=copy.deepcopy(network),
            optimizer_cls=optimizer_cls,
            num_instances=num_instances,
            )
        # else:
        #     res = self._estimate_mem_not_available(
        #         min_shape=min_shape, target_shape=target_shape,
        #         network=network, optimizer_cls=optimizer_cls,
        #         num_instances=num_instances,
        # )
        del self.block_mem_tensor
        self.block_mem_tensor = None
        torch.cuda.empty_cache()
        gc.collect()
        return res

    def _estimate_mem_available(self,
                                min_shape: Sequence[int],
                                target_shape: Sequence[int],
                                network: AbstractModel,
                                optimizer_cls: Callable = torch.optim.Adam,
                                num_instances: int = 1,
                                ) -> Tuple[int, bool]:
        logger.info("Estimating in memory.")
        fixed, dynamic = self.measure(shape=target_shape,
                                      network=network,
                                      optimizer_cls=optimizer_cls,
                                      num_instances=num_instances,
                                      )
        estimated_mem = fixed + dynamic
        return estimated_mem, estimated_mem < self.target_mem

    def _estimate_mem_not_available(self,
                                    min_shape: Sequence[int],
                                    target_shape: Sequence[int],
                                    network: AbstractModel,
                                    optimizer_cls: Callable = torch.optim.Adam,
                                    num_instances: int = 1,
                                    ) -> Tuple[int, bool]:
        raise NotImplementedError("!!!!!This needs more refinement!!!!")
        logger.info("Extrapolating memory consumption.")
        assert all([t >= m for t, m in zip(target_shape, min_shape)])
        fixed_mem, dyn_mem = self.measure(shape=min_shape,
                                          network=network,
                                          optimizer_cls=optimizer_cls,
                                          num_instances=num_instances,
                                          )
        ratios = [t / m for t, m in zip(target_shape, min_shape)]
        scale = reduce((lambda x, y: x * y), ratios)
        estimated_dyn_mem = dyn_mem * scale
        estimated_mem = estimated_dyn_mem + fixed_mem
        if self.context is not None:
            estimated_mem += self.context
        return estimated_mem, estimated_mem < self.target_mem

    def measure(self,
                shape: Sequence[int],
                network: AbstractModel,
                optimizer_cls: Callable = torch.optim.Adam,
                num_instances: int = 1,
                ):
        device = torch.device("cuda", self.gpu_id)
        logger.info(f"Estimating on {device} with shape {shape} and "
                    f"batch size {self.batch_size} and num_instances {num_instances}")
        loss = None
        opt = None
        inp = None
        block_tensor = None
        try:
            with cudnn_deterministic():
                torch.cuda.reset_peak_memory_stats()
                network = network.to(device)
                # torch.cuda.memory_allocated
                empty_mem = torch.cuda.memory_reserved()
                scaler = torch.cuda.amp.GradScaler()
                opt = optimizer_cls(network.parameters())

                boxes = [[0, 0, 2, 2]]
                if len(shape) == 4:  # in_channels + spatial dims
                    boxes[0].extend((0, 2))

                block_tensor = self.create_offset_tensor_on_GPU().to(device=device)
                import time
                time.sleep(1)

                for _ in range(10):
                    opt.zero_grad()
                    inp = {"images": torch.rand((self.batch_size, *shape), device=device, dtype=torch.float),
                           "targets": {
                               "target_boxes": [torch.tensor(
                                   boxes, device=device, dtype=torch.float).repeat(num_instances, 1)
                                   for _ in range(self.batch_size)],
                               "target_classes": [torch.tensor(
                                   [0] * num_instances, device=device, dtype=torch.float)
                                   for _ in range(self.batch_size)],
                               "target_seg": torch.zeros(
                                   (self.batch_size, *shape[1:]), device=device, dtype=torch.float),
                           }}
                    fixed_mem = torch.cuda.memory_reserved()
                    with torch.cuda.amp.autocast():
                        loss_dict, _ = network.train_step(
                            images=inp["images"],
                            targets=inp["targets"],
                            evaluation=False,
                            batch_num=0,
                        )
                        loss = sum(loss_dict.values())
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                dyn_mem = torch.cuda.memory_reserved()
        except Exception as e:
            logger.info(f"Caught error (If out of memory error do not worry): {e}")
            empty_mem = 0
            fixed_mem = float('Inf')
            dyn_mem = float('Inf')
        finally:
            del loss
            del opt
            del inp
            del block_tensor

        network.cpu()
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"Measured: {b2mb(empty_mem)} mb empty, "
                    f"{b2mb(fixed_mem)} mb fixed, "
                    f"{b2mb(dyn_mem)} mb dynamic")
        return fixed_mem - empty_mem, dyn_mem - fixed_mem


def num_gpus():
    """
    Number of GPUs independent of visible devices
    """
    return str(sp.check_output(["nvidia-smi", "-L"])).count('UUID')


def smi_memory_allocated(gpu_id: int = 0) -> int:
    """
    Read memory consumption from nvidia smi
    
    Returns:
        int: measured GPU memory in bytes
    """
    reading = int(sp.check_output(
        ['nvidia-smi', '--query-gpu=memory.used',
         '--format=csv,nounits,noheader'], encoding='utf-8').split('\n')[gpu_id])
    return mb2b(reading)


class Tracemalloc():
    def __init__(self, measure_fn):
        super().__init__()
        self.measure_fn = measure_fn
    
    def __enter__(self):
        self.begin = self.measure_fn()
        return self

    def __exit__(self, *exc):
        self.end  = self.measure_fn()
        self.used   = self.end - self.begin
        logger.info(f"Measured {self.used} byte GPU mem consumption")


class TorchTracemalloc(Tracemalloc):
    def __init__(self, gpu_id: int = None):
        if gpu_id is not None:
            fn = partial(torch.cuda.memory_reserved, device=gpu_id)
        else:
            fn = torch.cuda.memory_reserved
        super().__init__(measure_fn=fn)

    def __enter__(self):
        super().__enter__()
        torch.cuda.reset_peak_memory_stats()  # reset the peak to zero
        return self

    def __exit__(self, *exc):
        super().__exit__()
        self.peak = torch.cuda.max_memory_allocated()
        self.peaked = self.peak - self.begin
        logger.info(f"Measured peak {self.used} byte GPU mem consumption")


class SmiTracemalloc(Tracemalloc):
    def __init__(self, gpu_id: int = None):
        if gpu_id is not None:
            fn = partial(smi_memory_allocated, gpu_id=gpu_id)
        else:
            fn = smi_memory_allocated
        super().__init__(measure_fn=fn)


@contextmanager
def cudnn_deterministic():
    old_value = torch.backends.cudnn.deterministic
    old_value_benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    try:
        yield None
    finally:
        torch.backends.cudnn.deterministic = old_value
        torch.backends.cudnn.benchmark = old_value_benchmark
