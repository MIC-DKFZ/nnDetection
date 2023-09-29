# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from typing import Dict

from nndet.eval import AbstractEvaluator


class EvalMixin(ABC):
    """
    This mixin module defines the operation modes of the network.
    It provides the transformation to prepare the ground truth and input
    for the networks and defines the evaluations to perforn.
    """

    evaluators: Dict = {}  # needs to be overwritten in subclass

    def evaluation_init(self, plan: dict) -> Dict[str, AbstractEvaluator]:
        """
        Initialize evaluation. Needs to be called before
        ::method::`evaluation_step` and ::method::`evaluation_end`.

        Notes:
        make sure to call the super classes here!
        """
        return {}

    def evaluation_step(
        self,
        predictions: dict,
        targets: dict,
    ) -> None:
        """
        Evaluate a validation batch

        Args:
        predictions: dict with predictions.
        Exact keys depend on the module class
        targets: dict with ground truth.
        Exact keys depend on the module class.

        Notes:
        make sure to call the super classes here!
        """
        pass  # end parent calls

    def evaluation_end(self) -> Dict[str, float]:
        """
        Compute validation metrics of epoch

        .. code-block::

        General pipeline should look something like this:
        # collect other scores
        scores = super().evaluation_end()

        # compute own scores
        own_scores = ...

        # add own scores
        metric_scores.update(own_scores)

        Returns:
        Dict[str, float]: computed metrics

        Notes:
        make sure to call the super classes here!
        """
        # logger.info("--- Online Evaluation ---")
        return {}
