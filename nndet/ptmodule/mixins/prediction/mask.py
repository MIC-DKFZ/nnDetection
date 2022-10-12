# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Type

from nndet.inference.ensembler.base import BaseEnsembler
from nndet.inference.ensembler.mask import MaskViaBoxesSelectiveEnsembler
from nndet.inference.sweeper import MaskSweeper, Sweeper
from nndet.ptmodule.mixins.prediction.base import PredictionMixin


class MaskViaBoxPredictionMixin(PredictionMixin):
    @classmethod
    def requires_box_eval(cls) -> bool:
        return True

    @classmethod
    def requires_mask_eval(cls) -> bool:
        return True

    @classmethod
    def requires_case_eval(cls) -> bool:
        return True

    @classmethod
    def get_ensembler_cls(cls, dim: int) -> Type[BaseEnsembler]:
        """
        Returns:
            Type[BaseEnsembler]: return class of ensembler to use for this
                class
        """
        if dim == 3:
            return MaskViaBoxesSelectiveEnsembler
        else:
            raise ValueError(f"Dim {dim} not supported in get_ensembler_cls.")

    @classmethod
    def get_sweeper_cls(cls) -> Type[Sweeper]:
        """
        Returns:
            Type[Sweeper]: return class of sweeper to use for this class
        """
        return MaskSweeper
