# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import copy
import gc
import math
import subprocess as sp
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial, reduce
from typing import Callable, Sequence, Tuple, Union

import numpy as np
import torch
from loguru import logger

from nndet.core.abstract import AbstractDetector
from nndet.utils.format import to_nd_tuple


def bit2mb(x):
    return (x / 8) / (2**20)  # noqa: E704


def b2mb(x):
    return x / (2**20)  # noqa: E704


def mb2b(x):
    return x * (2**20)  # noqa: E704


# remove 11mb from target memory to have a little wiggle room
# (sometimes that amount was blocked on my GPU even though nothing was running)
ARCHS = {"RTX2080TI": 11523260416 - int(mb2b(11))}

# this is just an esitmation ... probably depend on the cuda version too
CUDA_CONTEXT = {"none": 0, "RTX2080TI": int(mb2b(910))}


class MemoryEstimator(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = None

    @abstractmethod
    def estimate(self, *args, **kwargs):
        raise NotImplementedError


class NoGPUMemoryEstimator(MemoryEstimator):
    def __init__(
        self,
        target_mem_mb: int,
        batch_size: int,
        buffer_mb: int = 910,
    ):
        """
        Estimate if model will fit into VRAM

        Args:
            target_mem_mb: memory of target GPU in mb
            batch_size: batch size during training
            buffer: additional vram buffer to account for uncertainty
                of estimate in mb
        """
        super().__init__()
        self.target_mem_mb = target_mem_mb
        self.batch_size = batch_size
        self.buffer_mb = buffer_mb
        self.cuda_context_mb = 910
        self.base_type = 16

        # heuristics to get close to estimates from nnDet V1
        # this slightly underestimates the memory copared to V1 for some cases
        # due to improved memory management of PyTorch & CUDA optimizations
        # it should still remain below the memory budget
        # self.encoder_heuristic = 1.2
        # self.decoder_heuristic = 1.0
        # self.det_head_heuristic = 2.6
        # self.seg_heuristic = 1.0
        # self.iou_matrix_heuristic = 1.81
        # self.heuristic_factor = 0.5
        # self.param_factor = 1.0

        # self.encoder_heuristic = 2 # conv + act + norm
        # self.decoder_heuristic = 2 # conv
        # self.det_head_heuristic = 6.0 # conv + act + norm
        # self.seg_heuristic = 1.0
        # self.iou_matrix_heuristic = 5
        # # self.param_factor = 3.0  # model + grad + optim state
        # # self.heuristic_factor = 2.05

        # heuristics to get close to estimates from nnDet V1
        self.encoder_heuristic = 2.5  # conv + act + norm
        self.decoder_heuristic = 2  # conv
        self.det_head_heuristic = 5.0  # conv + act + norm
        self.seg_heuristic = 1.0
        self.iou_matrix_heuristic = 7
        self.param_factor = 3.0  # model + grad + optim state
        self.heuristic_factor = 2.0

    def _estimate_feature_voxels(
        self,
        model_cfg: dict,
        plan_arch: dict,
        patch_size: Sequence[int],
        in_channels: int = None,
        num_instances: int = 1,
    ):
        # we assume a plain RetinaU-Net for estimation
        # other architectures will consume more or less VRAM and need
        # to be adjusted manually

        num_levels = len(plan_arch["conv_kernels"])
        rel_strides = plan_arch["strides"]
        decoder_levels = plan_arch["decoder_levels"]
        start_channels = plan_arch["start_channels"]
        max_channels = plan_arch["max_channels"]
        num_classes = plan_arch["classifier_classes"]
        fpn_channels = plan_arch["fpn_channels"]
        num_anchors = 27

        first_decoder_level = min(decoder_levels)

        # determine sizes of feature maps
        feature_map_sizes = [patch_size]
        encoder_channels = [start_channels]
        decoder_channels = [int(fpn_channels * (2 ** (-1 * first_decoder_level)))]
        _current_shape = patch_size
        for level_idx, level_stride in enumerate(rel_strides, start=1):
            nd_stride = to_nd_tuple(level_stride, len(patch_size))
            assert len(nd_stride) == len(patch_size)
            _current_shape = [cs / s for cs, s in zip(_current_shape, nd_stride)]
            feature_map_sizes.append(_current_shape)

            encoder_channels.append(min(start_channels * (2**level_idx), max_channels))
            decoder_channels.append(min(int(fpn_channels * (2 ** (level_idx - first_decoder_level))), fpn_channels))

        # sanity checks
        assert len(feature_map_sizes) == num_levels
        assert len(feature_map_sizes) == len(encoder_channels)
        assert len(feature_map_sizes) == len(decoder_channels)

        # compute voxels in input
        input_voxels = np.prod(patch_size, dtype=np.int64) * in_channels

        # compute voxels in encoder
        encoder_voxel_ops = self.encoder_heuristic * np.sum(
            [
                np.prod(fm_size, dtype=np.int64) * fm_channels
                for fm_size, fm_channels in zip(feature_map_sizes, encoder_channels)
            ],
            dtype=np.int64,
        )
        # compute voxels in decoder
        decoder_voxel_ops = self.decoder_heuristic * np.sum(
            [
                np.prod(fm_size, dtype=np.int64) * fm_channels
                for fm_size, fm_channels in zip(feature_map_sizes, decoder_channels)
            ],
            dtype=np.int64,
        )

        # compute voxels in detection head
        det_feature_map_sizes = [feature_map_sizes[dl] for dl in decoder_levels]
        det_head_voxel_ops = self.det_head_heuristic * np.sum(
            [np.prod(fm, dtype=np.int64) * fpn_channels for fm in det_feature_map_sizes],
            dtype=np.int64,
        )
        det_head_cls_voxel_ops = np.sum(
            [num_classes * np.prod(fm, dtype=np.int64) for fm in det_feature_map_sizes],
            dtype=np.int64,
        )
        det_head_box_voxel_ops = np.sum(
            [num_anchors * np.prod(fm, dtype=np.int64) for fm in det_feature_map_sizes],
            dtype=np.int64,
        )

        # compute voxels in segmentation head
        seg_voxels = self.seg_heuristic * np.prod(patch_size, dtype=np.int64)  # * num_classes

        # compute elements of IoU matrix
        iou_matrix_ops = self.iou_matrix_heuristic * det_head_box_voxel_ops * num_instances

        final_estimate = (
            input_voxels
            + encoder_voxel_ops
            + decoder_voxel_ops
            + det_head_voxel_ops
            + det_head_cls_voxel_ops
            + det_head_box_voxel_ops
            + seg_voxels
            + iou_matrix_ops
        )

        logger.info(
            f"++++++ Estimated no scale::"
            f"enc {bit2mb(encoder_voxel_ops * self.base_type / self.encoder_heuristic)} "
            f"dec {bit2mb(decoder_voxel_ops * self.base_type) / self.decoder_heuristic} "
            f"det {bit2mb(det_head_voxel_ops * self.base_type) / self.det_head_heuristic} "
            f"cls {bit2mb(det_head_cls_voxel_ops * self.base_type)} "
            f"box {bit2mb(det_head_box_voxel_ops * self.base_type)} "
            f"seg {bit2mb(seg_voxels * self.base_type) / self.seg_heuristic} "
            f"iou {bit2mb(iou_matrix_ops * self.base_type) / self.iou_matrix_heuristic} ni {num_instances}++++++"
        )
        logger.info(
            f"++++++ Estimated:: {bit2mb(final_estimate * self.base_type)} "
            f"enc {bit2mb(encoder_voxel_ops * self.base_type)} "
            f"dec {bit2mb(decoder_voxel_ops * self.base_type)} "
            f"det {bit2mb(det_head_voxel_ops * self.base_type)} "
            f"cls {bit2mb(det_head_cls_voxel_ops * self.base_type)} "
            f"box {bit2mb(det_head_box_voxel_ops * self.base_type)} "
            f"seg {bit2mb(seg_voxels * self.base_type)} "
            f"iou {bit2mb(iou_matrix_ops * self.base_type)} ni {num_instances}++++++"
        )
        return final_estimate

    def estimate(
        self,
        target_shape: Sequence[int],
        model_cfg: dict,
        plan_arch: dict,
        network: AbstractDetector,
        optimizer_cls: Callable = torch.optim.Adam,
        in_channels: int = None,
        num_instances: int = 1,
        **kwargs,
    ) -> Tuple[int, bool]:
        params = np.sum([np.prod(n.shape, dtype=np.int64) for n in network.parameters()])
        feature_voxels = self._estimate_feature_voxels(
            model_cfg=model_cfg,
            plan_arch=plan_arch,
            patch_size=target_shape,
            in_channels=in_channels,
            num_instances=num_instances,
        )
        param_mb = bit2mb(self.param_factor * params * self.base_type)
        voxel_mb = self.heuristic_factor * self.batch_size * bit2mb(feature_voxels * self.base_type)
        full_estimate = param_mb + voxel_mb + self.cuda_context_mb
        logger.info(f"++++++ Final estimate {full_estimate} for path size {target_shape} ++++++")
        # from IPython import embed; embed();
        return full_estimate, full_estimate <= self.target_mem_mb


class MemoryEstimatorDetection(MemoryEstimator):
    def __init__(
        self,
        target_mem: Union[float, str] = "RTX2080TI",
        gpu_id: int = 0,
        context: Union[float, str] = "RTX2080TI",
        offset: int = mb2b(768),
        batch_size: int = 1,
        mixed_precision: bool = True,
    ):
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
        return torch.rand(
            math.ceil(self.offset / tensor_mem),
            dtype=float,
            requires_grad=False,
            device=device,
        )

    def estimate(
        self,
        min_shape: Sequence[int],
        target_shape: Sequence[int],
        network: AbstractDetector,
        optimizer_cls: Callable = torch.optim.Adam,
        in_channels: int = None,
        num_instances: int = 1,
        **kwargs,
    ) -> Tuple[int, bool]:
        if in_channels is not None:
            min_shape = [in_channels, *min_shape]
            target_shape = [in_channels, *target_shape]

        # all_mem - reserved_mem[misc + context] + context
        available_mem = (
            torch.cuda.get_device_properties(self.gpu_id).total_memory
            - smi_memory_allocated(self.gpu_id)
            + self.context
        )
        logger.info(
            f"Found available gpu memory: {available_mem} bytes / {b2mb(available_mem)} mb "
            f"and estimating for {self.target_mem} bytes / {b2mb(self.target_mem)}"
        )

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

    def _estimate_mem_available(
        self,
        min_shape: Sequence[int],
        target_shape: Sequence[int],
        network: AbstractDetector,
        optimizer_cls: Callable = torch.optim.Adam,
        num_instances: int = 1,
    ) -> Tuple[int, bool]:
        logger.info("Estimating in memory.")
        fixed, dynamic = self.measure(
            shape=target_shape,
            network=network,
            optimizer_cls=optimizer_cls,
            num_instances=num_instances,
        )
        estimated_mem = fixed + dynamic
        return estimated_mem, estimated_mem < self.target_mem

    def _estimate_mem_not_available(
        self,
        min_shape: Sequence[int],
        target_shape: Sequence[int],
        network: AbstractDetector,
        optimizer_cls: Callable = torch.optim.Adam,
        num_instances: int = 1,
    ) -> Tuple[int, bool]:
        raise NotImplementedError("!!!!!This needs more refinement!!!!")
        logger.info("Extrapolating memory consumption.")
        assert all([t >= m for t, m in zip(target_shape, min_shape)])
        fixed_mem, dyn_mem = self.measure(
            shape=min_shape,
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

    def measure(
        self,
        shape: Sequence[int],
        network: AbstractDetector,
        optimizer_cls: Callable = torch.optim.Adam,
        num_instances: int = 1,
    ):
        device = torch.device("cuda", self.gpu_id)
        logger.info(
            f"Estimating on {device} with shape {shape} and "
            f"batch size {self.batch_size} and num_instances {num_instances}"
        )
        try:
            loss = None
            opt = None
            inp = None
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
                    inp = {
                        "images": torch.rand((self.batch_size, *shape), device=device, dtype=torch.float),
                        "targets": {
                            "target_boxes": [
                                torch.tensor(boxes, device=device, dtype=torch.float).repeat(num_instances, 1)
                                for _ in range(self.batch_size)
                            ],
                            "target_classes": [
                                torch.tensor(
                                    [0] * num_instances,
                                    device=device,
                                    dtype=torch.float,
                                )
                                for _ in range(self.batch_size)
                            ],
                            "target_seg": torch.zeros(
                                (self.batch_size, *shape[1:]),
                                device=device,
                                dtype=torch.float,
                            ),
                        },
                    }
                    fixed_mem = torch.cuda.memory_reserved()
                    with torch.cuda.amp.autocast():
                        loss_dict = network.train_step(
                            images=inp["images"],
                            targets=inp["targets"],
                            batch_num=0,
                        )
                        loss = sum(loss_dict.values())
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                dyn_mem = torch.cuda.memory_reserved()
        except (RuntimeError,) as e:
            logger.info(f"Caught error (If out of memory error do not worry): {e}")
            empty_mem = 0
            fixed_mem = float("Inf")
            dyn_mem = float("Inf")
        finally:
            del loss

        del opt
        del inp
        del block_tensor

        network.cpu()
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(
            f"Measured: {b2mb(empty_mem)} mb empty, " f"{b2mb(fixed_mem)} mb fixed, " f"{b2mb(dyn_mem)} mb dynamic"
        )
        return fixed_mem - empty_mem, dyn_mem - fixed_mem


def num_gpus():
    """
    Number of GPUs independent of visible devices
    """
    return str(sp.check_output(["nvidia-smi", "-L"])).count("UUID")


def smi_memory_allocated(gpu_id: int = 0) -> int:
    """
    Read memory consumption from nvidia smi

    Returns:
        int: measured GPU memory in bytes
    """
    reading = int(
        sp.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            encoding="utf-8",
        ).split("\n")[gpu_id]
    )
    return mb2b(reading)


class Tracemalloc:
    def __init__(self, measure_fn):
        super().__init__()
        self.measure_fn = measure_fn

    def __enter__(self):
        self.begin = self.measure_fn()
        return self

    def __exit__(self, *exc):
        self.end = self.measure_fn()
        self.used = self.end - self.begin
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
