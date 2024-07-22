# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from os import PathLike
from typing import Dict, List, Sequence

import numpy as np
from loguru import logger

import nndet.core.ops_np as ops_np
from nndet.planning.architecture.boxes import BoxC002
from nndet.planning.architecture.boxes.utils import concatenate_property_boxes
from nndet.planning.estimator import MemoryEstimatorDetection
from nndet.planning.experiment import PLANNER_REGISTRY
from nndet.planning.experiment.base import AbstractPlanner
from nndet.preprocessing.preprocessor import DynDTypePreprocessor, GenericPreprocessor
from nndet.ptmodule import MODULE_REGISTRY
from nndet.utils.config import load_plan_from_dir


@PLANNER_REGISTRY.register
class D3V001(AbstractPlanner):
    def plan_experiment(
        self,
        model_name: str,
        model_cfg: Dict,
    ) -> List[str]:
        """
        Plan the whole experiment (currently only one stage is supported)
        (uses :func:`self.save_plans()` to save the results)

        Args:
            model_name: name of model to plan for
            model_cfg: config to initialize model for VRAM estimation

        Returns:
            List: identifiers of created plans
        """
        identifiers = []

        # create full resolution 3d plan
        mode = "3d"
        plan_3d = self.plan_base(mode=mode)
        plan_3d["network_dim"] = 3
        plan_3d["dataloader_kwargs"] = {}
        plan_3d["data_identifier"] = self.get_data_identifier(mode=mode)
        plan_3d["postprocessing"] = self.determine_postprocessing(mode=mode)

        plan_3d = self.plan_base_stage(
            plan_3d,
            model_name=model_name,
            model_cfg=model_cfg,
        )

        # determine if additional low res model needs to be trained
        plan_3d["trigger_lr1"] = self.trigger_low_res_model(
            prev_res_patch_size=plan_3d["patch_size"],
            transpose_forward=plan_3d["transpose_forward"],
        )
        identifiers.append(self.save_plan(plan=plan_3d, mode=plan_3d["mode"]))

        if plan_3d["trigger_lr1"]:
            logger.info("Triggered Low Resolution Model")
            mode = "3dlr1"
            plan_3dlr1 = self.plan_base(mode=mode)
            plan_3dlr1["network_dim"] = 3
            plan_3dlr1["dataloader_kwargs"] = {}
            plan_3dlr1["data_identifier"] = self.get_data_identifier(mode=mode)
            plan_3dlr1["postprocessing"] = self.determine_postprocessing(mode=mode)

            plan_3dlr1 = self.plan_base_stage(
                plan_3dlr1,
                model_name=model_name,
                model_cfg=model_cfg,
            )
            identifiers.append(self.save_plan(plan=plan_3dlr1, mode=plan_3dlr1["mode"]))
        return identifiers

    def create_architecture_planner(
        self,
        model_name: str,
        model_cfg: dict,
        mode: str,
    ) -> BoxC002:
        """
        Create Architecture planner
        """
        estimator = MemoryEstimatorDetection()
        architecture_planner = BoxC002(
            preprocessed_output_dir=self.preprocessed_output_dir,
            save_dir=self.preprocessed_output_dir / "analysis" / f"{self.__class__.__name__}_{mode}",
            estimator=estimator,
            network_cls=MODULE_REGISTRY.get(model_name),
            model_cfg=model_cfg,
        )
        return architecture_planner

    @staticmethod
    def create_preprocessor(plan: Dict) -> GenericPreprocessor:
        """
        Create Preprocessor
        """
        preprocessor = GenericPreprocessor(
            norm_scheme_per_modality=plan["normalization_schemes"],
            use_mask_for_norm=plan["use_mask_for_norm"],
            transpose_forward=plan["transpose_forward"],
            intensity_properties=plan["dataset_properties"]["intensity_properties"],
            resample_anisotropy_threshold=plan["resample_anisotropy_threshold"],
            patch_size=plan["patch_size"],
            preprocessed_data_format=plan["preprocessed_data_format"],
        )
        return preprocessor

    def determine_forward_backward_permutation(self, mode: str):
        """
        Determine position of z direction (absolute position is defined by z_first)
        Result is
        saved into :param:`transpose_forward` and :param:`transpose_backward`
        """
        # spacings = self.data_properties['all_spacings']
        # sizes = self.data_properties['all_sizes']

        target_spacing = self.determine_target_spacing(mode=mode)
        # new_sizes = [np.array(i) / target_spacing * np.array(j) for i, j in zip(spacings, sizes)]

        dims = len(target_spacing)
        max_spacing_axis = np.argmax(target_spacing)
        remaining_axes = [i for i in list(range(dims)) if i != max_spacing_axis]
        # self.transpose_forward = remaining_axes + [max_spacing_axis] # y, x, z
        self.transpose_forward = [max_spacing_axis] + remaining_axes  # z, y, x
        self.transpose_backward = [np.argwhere(np.array(self.transpose_forward) == i)[0][0] for i in range(dims)]

    def determine_target_spacing(self, mode: str) -> np.ndarray:
        """
        Determine target spacing

        Args:
            mode: Current planning mode. Typically one of '2d' | '3d' | '3dlr1'

        Raises:
            RuntimeError: not supported mode (supported are 2d, 3d, 3dlrX)

        Returns:
            np.ndarray: target spacing
        """
        base_target_spacing = self._target_spacing_base()
        if mode == "3d" or mode == "2d":
            target_spacing = base_target_spacing
        else:
            if "lr" not in mode:
                raise RuntimeError(f"Mode {mode} is not supported for target spacing.")
            downscale = int(mode.split("lr")[-1])
            target_spacing = base_target_spacing * (2**downscale)
        return target_spacing

    def _target_spacing_base(self) -> np.ndarray:
        """
        Determine target spacing.

        Same as nnUNet v21
        https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/experiment_planning/experiment_planner_baseline_3DUNet_v21.py
        """
        spacings = self.data_properties["all_spacings"]
        sizes = self.data_properties["all_sizes"]

        target = np.percentile(np.vstack(spacings), self.target_spacing_percentile, 0)

        target_size = np.percentile(np.vstack(sizes), self.target_spacing_percentile, 0)
        # target_size_mm = np.array(target) * np.array(target_size)
        # we need to identify datasets for which a different target spacing could be beneficial. These datasets have
        # the following properties:
        # - one axis which much lower resolution than the others
        # - the lowres axis has much less voxels than the others
        # - (the size in mm of the lowres axis is also reduced)
        worst_spacing_axis = np.argmax(target)
        other_axes = [i for i in range(len(target)) if i != worst_spacing_axis]
        other_spacings = [target[i] for i in other_axes]
        other_sizes = [target_size[i] for i in other_axes]

        has_aniso_spacing = target[worst_spacing_axis] > (self.anisotropy_threshold * min(other_spacings))
        has_aniso_voxels = target_size[worst_spacing_axis] * self.anisotropy_threshold < min(other_sizes)
        # we don't use the last one for now
        # median_size_in_mm = target[target_size_mm] * RESAMPLING_SEPARATE_Z_ANISOTROPY_THRESHOLD < max(target_size_mm)

        if has_aniso_spacing and has_aniso_voxels:
            spacings_of_that_axis = np.vstack(spacings)[:, worst_spacing_axis]
            target_spacing_of_that_axis = np.percentile(spacings_of_that_axis, 10)
            # don't let the spacing of that axis get higher than the other axes
            if target_spacing_of_that_axis < min(other_spacings):
                target_spacing_of_that_axis = max(min(other_spacings), target_spacing_of_that_axis) + 1e-5
            target[worst_spacing_axis] = target_spacing_of_that_axis
        return target

    def trigger_low_res_model(
        self,
        prev_res_patch_size: Sequence[int],
        transpose_forward: Sequence[int],
    ) -> bool:
        """
        Trigger additional low resolution model

        Args:
            prev_res_patch_size: patch size of previous stage

        Returns:
            bool: If True, trigger a low resolution model. If False, current
                resolution is ok.
        """
        all_boxes = [case["boxes"] for case_id, case in self.data_properties["instance_props_per_patient"].items()]
        all_boxes = concatenate_property_boxes(all_boxes)
        object_size = np.percentile(ops_np.box_size_np(all_boxes), 99.5, axis=0)
        object_size = object_size[list(transpose_forward)]

        if (np.asarray(prev_res_patch_size) < object_size).any():
            return True
        else:
            return False

    def get_plan_identifiers(self) -> List[str]:
        """
        Retrieve all plan identifier starting from highest res (fullres) to
        lowest res (highest target spacing)

        Returns:
            List[str]: ordered list of plan identifier
        """
        fullres_identifier = self._get_identifier("3d")
        fullres_plan = load_plan_from_dir(self.preprocessed_output_dir, fullres_identifier)
        ids = [fullres_identifier]
        if fullres_plan["trigger_lr1"]:
            ids.append(self._get_identifier("3dlr1"))
        return ids


@PLANNER_REGISTRY.register
class D3V001DynDtype(D3V001):
    @staticmethod
    def create_preprocessor(plan: Dict) -> DynDTypePreprocessor:
        """
        Create Preprocessor
        """
        preprocessor = DynDTypePreprocessor(
            norm_scheme_per_modality=plan["normalization_schemes"],
            use_mask_for_norm=plan["use_mask_for_norm"],
            transpose_forward=plan["transpose_forward"],
            intensity_properties=plan["dataset_properties"]["intensity_properties"],
            resample_anisotropy_threshold=plan["resample_anisotropy_threshold"],
        )
        return preprocessor


@PLANNER_REGISTRY.register
class D3V001Blosc(D3V001):
    def __init__(self, preprocessed_output_dir: PathLike):
        """
        Class for experiment planning using Blosc2 for preprocessed data

        Args:
            preprocessed_output_dir: path to directory where preprocessed
                data will be saved
        """
        super().__init__(preprocessed_output_dir)
        self.preprocessed_data_format = "b2nd"
