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

import math
from typing import List, Union, Sequence

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from loguru import logger


def linear_warm_up(
    iteration: int,
    initial_lr: float,
    num_iterations: int,
    final_lr: float,
    ) -> float:
    """
    Linear learning rate warm up

    Args:
        iteration: current iteration
        initial_lr: initial learning rate for poly lr
        num_iterations: total number of iterations for of warmup
        final_lr: final learning rate of warmup

    Returns:
        float: learning rate
    """
    assert final_lr > initial_lr
    if iteration >= num_iterations:
        logger.warning(f"WarmUp was stepped too often, {iteration} "
                f"but only {num_iterations} were expected!")
    
    return initial_lr + (final_lr - initial_lr) * (float(iteration) / float(num_iterations))


def poly_lr(
    iteration: int,
    initial_lr: float,
    num_iterations: int,
    gamma: float,
    ) -> float:
    """
    initial_lr * (1 - epoch / max_epochs) ** gamma

    Adapted from
    https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/learning_rate/poly_lr.py
    https://arxiv.org/abs/1904.08128

    Args:
        iteration: current iteration
        initial_lr: initial learning rate for poly lr
        num_iterations: total number of iterations of poly lr
        gamma: gamma value

    Returns:
        float: learning rate
    """
    if iteration >= num_iterations:
        logger.warning(f"PolyLR was stepped too often, {iteration} "
                f"but only {num_iterations} were expected! "
                f"Using {num_iterations - 1} for lr computation.")
        iteration = num_iterations - 1
    return initial_lr * (1 - iteration / float(num_iterations)) ** gamma


def cyclic_linear_lr(
    iteration: int,
    num_iterations_cycle: int,
    initial_lr: float,
    final_lr: float,
    ) -> float:
    """
    Linearly cycle learning rate

    Args:
        iteration: current iteration
        num_iterations_cycle: number of iterations per cycle
        initial_lr: learning rate to start cycle
        final_lr: learning rate to end cycle

    Returns:
        float: learning rate
    """
    cycle_iteration = int(iteration) % num_iterations_cycle
    lr_multiplier = 1 - (cycle_iteration / float(num_iterations_cycle))
    return initial_lr + (final_lr - initial_lr) * lr_multiplier


def cosine_annealing_lr(
    iteration: int,
    num_iterations: int,
    initial_lr: float,
    final_lr: float,
):
    """
    Cosine annealing NO restarts

    Args:
        iteration: current iteration
        num_iterations: total number of iterations of coine lr
        initial_lr: learning rate to start
        final_lr: learning rate to end

    Returns:
        float: learning rate
    """
    return final_lr + 0.5 * (initial_lr - final_lr) * (1 + \
        math.cos(math.pi * float(iteration) / float(num_iterations)))


class LinearWarmupPolyLR(_LRScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 warm_iterations: int,
                 warm_lr: Union[float, Sequence[float]],
                 poly_gamma: float,
                 num_iterations: int,
                 last_epoch: int = -1,
                 ) -> None:
        """
        Linear Warm Up LR -> Poly LR -> Cycle LR

        Args:
            optimizer: optimizer for lr scheduling
            warm_iterations: number of warmup iterations
            warm_lr: initial learning rate of warm up
            poly_gamma: gamma of poly lr
            num_iterations: total number of iterations (including warmup)
            last_epoch: The index of the last epoch. Defaults to -1.
        """
        self.num_iterations = num_iterations
        # warmup
        self.warm_iterations = warm_iterations

        if not isinstance(warm_lr, list) and not isinstance(warm_lr, tuple):
            self.warm_lr = [warm_lr] * len(optimizer.param_groups)
        else:
            if len(warm_lr) != len(optimizer.param_groups):
                raise ValueError("Expected {} warm_lr, but got {}".format(
                    len(optimizer.param_groups), len(warm_lr)))
            self.warm_lr = [warm_lr]

        # poly lr
        self.poly_iterations = self.num_iterations - self.warm_iterations
        self.poly_gamma = poly_gamma
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute current learning rate for each param group
        """
        if self.last_epoch < self.warm_iterations:
            # warm up period
            lrs = [linear_warm_up(
                iteration=self._step_count,
                initial_lr=self.warm_lr[idx],
                num_iterations=self.warm_iterations,
                final_lr=base_lr,
                ) for idx, base_lr in enumerate(self.base_lrs)]
        else:
            # poly lr phase
            lrs = [poly_lr(
                iteration=self._step_count - self.warm_iterations,
                initial_lr=base_lr,
                num_iterations=self.poly_iterations,
                gamma=self.poly_gamma,
            ) for idx, base_lr in enumerate(self.base_lrs)]
        return lrs


class CycleLinear(_LRScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 cycle_num_iterations: int,
                 cycle_initial_lr: Union[float, Sequence[float]],
                 cycle_final_lr:Union[float, Sequence[float]],
                 last_epoch: int = -1,
                 ) -> None:
        """
        Cyclic learning rates with linear decay

        Args:
            optimizer: optimizer for lr scheduling
            cycle_num_iterations: number of iterations per cycle
            cycle_initial_lr: initial learning rate of cycle
            cycle_final_lr: final learning rate of cycle
            last_epoch: The index of the last epoch. Defaults to -1.
        """
        # cycle linear lr
        self.cycle_num_iterations = cycle_num_iterations

        if not isinstance(cycle_initial_lr, list) and not isinstance(cycle_initial_lr, tuple):
            self.cycle_initial_lr = [cycle_initial_lr] * len(optimizer.param_groups)
        else:
            if len(cycle_initial_lr) != len(optimizer.param_groups):
                raise ValueError("Expected {} cycle_initial_lr, but got {}".format(
                    len(optimizer.param_groups), len(cycle_initial_lr)))
            self.cycle_initial_lr = [cycle_initial_lr]

        if not isinstance(cycle_final_lr, list) and not isinstance(cycle_final_lr, tuple):
            self.cycle_final_lr = [cycle_final_lr] * len(optimizer.param_groups)
        else:
            if len(cycle_final_lr) != len(optimizer.param_groups):
                raise ValueError("Expected {} cycle_final_lr, but got {}".format(
                    len(optimizer.param_groups), len(cycle_final_lr)))
            self.cycle_final_lr = [cycle_final_lr]
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute current learning rate for each param group
        """
        lrs = [cyclic_linear_lr(
            iteration=max(self._step_count - 1, 0), # init steps once
            num_iterations_cycle=self.cycle_num_iterations,
            initial_lr=self.cycle_initial_lr[idx],
            final_lr=self.cycle_final_lr[idx],
        ) for idx, base_lr in enumerate(self.base_lrs)]
        return lrs


class WarmUpExponential(_LRScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 beta2: float,
                 last_epoch: int = -1,
                 ):
        """
        Expoenential learning rate warmup
        warmup_lr = base_lr * 1 - exp(- (1 - beta2) * t)
        for 2 * (1 - beta2)^(-1) iterations
        `On the adequacy of untuned warmup for adaptive optimization`
        https://arxiv.org/abs/1910.04209

        Args:
            optimizer: optimizer to schedule lr from (best used with Adam,
                AdamW)
            beta2: second beta param of Adam optimizer.
            last_epoch: The index of the last epoch. Defaults to -1.
        """
        self.iterations = int(2. * (1. / (1. - beta2)))
        self.beta2 = beta2
        logger.info(f"Running exponential warmup for {self.iterations} iterations")
        self.finished = False

        super().__init__(optimizer=optimizer, last_epoch=last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute current learning rate for each param group
        """
        # last epoch is automatically handled by parent class
        return [base_lr * (1 - math.exp(- (1 - self.beta2) * self.last_epoch))
                for base_lr in zip(self.base_lrs)]
