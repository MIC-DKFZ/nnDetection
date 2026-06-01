# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import List, Sequence

import numpy as np

from nndet.planning.architecture.boxes.base import BoxC001
from nndet.planning.architecture.boxes.c002 import BoxC002


class BoxV002(BoxC002):
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
        if not mode.startswith("3d"):
            raise NotImplementedError("Only 3d planning is supported")
        plan = BoxC001.plan(
            self,
            transpose_forward=transpose_forward,
            target_spacing_transposed=target_spacing_transposed,
            median_shape_transposed=median_shape_transposed,
        )
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
