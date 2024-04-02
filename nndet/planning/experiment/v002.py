# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Sequence

import numpy as np
from loguru import logger

import nndet.core.ops_np as ops_np
from nndet.planning.architecture.boxes import BoxC002
from nndet.planning.architecture.boxes.utils import concatenate_property_boxes
from nndet.planning.estimator import NoGPUMemoryEstimator
from nndet.planning.experiment import PLANNER_REGISTRY
from nndet.planning.experiment.v001 import D3V001
from nndet.preprocessing.preprocessor import GenericPreprocessor
from nndet.ptmodule import MODULE_REGISTRY

# TODO: introduce use_box_io as plan parameter + add different plan identifiers

# TODO: think about this ... -> dynamic dtype for data and seg


@PLANNER_REGISTRY.register
class D3V002(D3V001):
    pass


@PLANNER_REGISTRY.register
class D3V002T(D3V001):
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

        # determine if additional low res models need to be trained
        for lowres_idx in range(1, 6):  # trigger max 6 low resolution stages
            trigger_lr = self.trigger_low_res_model(
                prev_res_patch_size=plan_3d["patch_size"] * (2 ** (lowres_idx - 1)),
                transpose_forward=plan_3d["transpose_forward"],
            )
            if lowres_idx == 1:
                plan_3d["trigger_lr"] = trigger_lr
            if not trigger_lr:
                break

            logger.info(f"Triggered Low Resolution Model {lowres_idx}")
            mode = f"3dlr{lowres_idx}"
            plan_3dlr = self.plan_base(mode=mode)
            plan_3dlr["network_dim"] = 3
            plan_3dlr["dataloader_kwargs"] = {}
            plan_3dlr["data_identifier"] = self.get_data_identifier(mode=mode)
            plan_3dlr["postprocessing"] = self.determine_postprocessing(mode=mode)

            plan_3dlr = self.plan_base_stage(
                plan_3dlr,
                model_name=model_name,
                model_cfg=model_cfg,
            )
            identifiers.append(self.save_plan(plan=plan_3dlr, mode=plan_3dlr["mode"]))  # save lowres
        identifiers.append(self.save_plan(plan=plan_3d, mode=plan_3d["mode"]))  # save fullres
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
        estimator = NoGPUMemoryEstimator(
            target_mem_mb=11247,
            batch_size=4,
            buffer_mb=910,
        )
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
        )
        return preprocessor

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

    @classmethod
    def get_plan_identifiers(cls):
        ids = []
        for mode in ["3d", "3dlr1"]:
            ids.append(f"{cls.__name__}_{mode}")
        return ids
