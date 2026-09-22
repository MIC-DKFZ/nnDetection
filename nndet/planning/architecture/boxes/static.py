# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from loguru import logger

from nndet.planning.architecture.boxes.base import BaseBoxesPlanner
from nndet.planning.architecture.boxes.c002 import BoxC002


class StaticArchitecturePlanner(BoxC002):
    @abstractmethod
    def get_static_paramters(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Statically define training and architecture paramters

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: dictionary containing
                (1) training parameters and (2) architecture parameters.

                Training paramters need to include:

                    ``"patch_size"``: List[int]

                    ``"batch_size"`` int

                Architecture parameteres need to include:

                    ``"conv_kernels"``

                    ``"strides"``

                    ``"decoder_levels"``

                    ``"start_channels"``
                        can be set in config or in plan

                    ``"fpn_channels"``
                        can be set in config or in plan

                    ``"head_channels"``
                        can be set in config or in plan

                    ``"max_channels"``
                        can be set in config or in plan
        """
        raise NotImplementedError()

    def plan(
        self,
        target_spacing_transposed: Sequence[float],
        median_shape_transposed: Sequence[float],
        transpose_forward: Sequence[int],
        mode: str = "3d",
    ) -> dict:
        """
        Plan network architecture, anchors, patch size and batch size

        Args:
            target_spacing_transposed: spacing after data is transposed and resampled
            median_shape_transposed: median shape after data is
                transposed and resampled
            transpose_forward: new ordering of axes for forward pass
            mode: mode to use for planning ('3d' | '2d')

        Returns:
            dict: training and architecture information

        See Also:
            :method:`_plan_architecture`, :method:`_plan_anchors`
        """
        if mode != "3d":
            raise NotImplementedError(f"Only 3d mode supported, got {mode}")

        BaseBoxesPlanner.plan(
            self,
            target_spacing_transposed=target_spacing_transposed,
            median_shape_transposed=median_shape_transposed,
            transpose_forward=transpose_forward,
            mode=mode,
        )
        training_parameters, architecture_parameters = self.get_static_paramters()

        # parse parameters from `process_properties` functions; generally these can be kept as is
        architecture_parameters["dim"] = self.architecture_kwargs["dim"]
        architecture_parameters["in_channels"] = self.architecture_kwargs["in_channels"]
        architecture_parameters["seg_classes"] = self.architecture_kwargs["seg_classes"]
        architecture_parameters["classifier_classes"] = self.architecture_kwargs["classifier_classes"]

        # these params are used during anchor planning -> plan anchors
        self.architecture_kwargs["strides"] = architecture_parameters["strides"]
        self.architecture_kwargs["decoder_levels"] = architecture_parameters["decoder_levels"]
        anchor_parameters = self._plan_anchors(
            target_spacing_transposed=target_spacing_transposed,
            transpose_forward=transpose_forward,
        )

        plan = {
            **training_parameters,
            "architecture": {**architecture_parameters, "arch_name": "MultiDet"},
            "anchors": {**anchor_parameters},

        }

        num_instances_per_patch_est = self._estimate_num_instances_per_patch(
            patch_size=plan["patch_size"],
            target_spacing_transposed=target_spacing_transposed,
            transpose_forward=transpose_forward,
        )
        est_instances_patch = {
            "min": int(min(num_instances_per_patch_est)),
            "max": int(max(num_instances_per_patch_est)),
            "mean": int(np.mean(num_instances_per_patch_est)),
            "median": int(np.median(num_instances_per_patch_est)),
            "perc95": int(np.percentile(num_instances_per_patch_est, 95)),
        }
        plan["architecture"]["est_instances_patch"] = est_instances_patch

        num_instances_per_image = self._get_num_instances_per_img()
        instances_img = {
            "min": int(min(num_instances_per_image)),
            "max": int(max(num_instances_per_image)),
            "mean": int(np.mean(num_instances_per_image)),
            "median": int(np.median(num_instances_per_image)),
            "perc95": int(np.percentile(num_instances_per_image, 95)),
        }
        plan["architecture"]["instances_img"] = instances_img

        logger.info(f"Using architecture plan: \n{plan}")
        return plan

    def _get_num_instances_per_img(
        self,
    ) -> List[int]:
        num_instances = []
        for boxes in self.all_boxes:
            if isinstance(boxes, list):
                num_instances.append(0)
            elif boxes.size > 0:
                num_instances.append(boxes.shape[0])
            else:
                num_instances.append(0)
        return num_instances
