# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Sequence

from loguru import logger

from nndet.planning.architecture.boxes import BaseBoxesPlanner
from nndet.planning.architecture.boxes.v002 import BoxV002
from nndet.planning.estimator import MemoryEstimatorDetection, NoGPUMemoryEstimator
from nndet.planning.experiment import PLANNER_REGISTRY
from nndet.planning.experiment.v001 import D3V001
from nndet.preprocessing.preprocessor.generic import DynDTypePreprocessor
from nndet.ptmodule import MODULE_REGISTRY
from nndet.utils.config import load_plan_from_dir


@PLANNER_REGISTRY.register
class D3V002(D3V001):
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
        plan_3d["lowres_identifiers"] = identifiers
        identifiers.append(self.save_plan(plan=plan_3d, mode=plan_3d["mode"]))  # save fullres
        return identifiers

    def get_data_identifier(self, mode: str) -> str:
        """
        D3V001 and D3V002 share the same data preprocessing paramters
        and preprocessor -> thus we use D3V001 data for this plan as well

        Args:
            mode: current operation mode

        Returns:
            str: data identifier
        """
        class_identifier = list(self.__class__.__name__)
        class_identifier[5] = "1"
        class_identifier = "".join(class_identifier)
        return f"{class_identifier}_{mode}"

    def create_architecture_planner(
        self,
        model_name: str,
        model_cfg: dict,
        mode: str,
    ) -> BaseBoxesPlanner:
        """
        Create Architecture planner
        """
        estimator = NoGPUMemoryEstimator(
            target_mem_mb=11247,
            batch_size=4,
            buffer_mb=0,
        )
        architecture_planner = BoxV002(
            preprocessed_output_dir=self.preprocessed_output_dir,
            save_dir=self.preprocessed_output_dir / "analysis" / f"{self.__class__.__name__}_{mode}",
            estimator=estimator,
            network_cls=MODULE_REGISTRY.get(model_name),
            model_cfg=model_cfg,
        )
        return architecture_planner

    def get_plan_identifiers(self) -> List[str]:
        """
        Retrieve all plan identifier starting from highest res (fullres) to
        lowest res (highest target spacing)

        Returns:
            List[str]: ordered list of plan identifier
        """
        fullres_identifier = self._get_identifier("3d")
        fullres_plan = load_plan_from_dir(self.preprocessed_output_dir, fullres_identifier)
        return [fullres_identifier] + fullres_plan["lowres_identifiers"]

    def determine_dummy_2d_data_augmentation(
        self,
        target_spacing_transposed: Sequence[float],
        median_shape_transposed: Sequence[int],
        patch_size: Sequence[int],
    ):
        """
        Determine if dummy 2d data augmentation should be used

        Args:
            target_spacing_transposed: target spacing after applying forward
                transposing
            median_shape_transposed: median shape after applying forward
                transposing
            patch_size: patch size for training

        Returns:
            bool: if dummy 2d data augmentation should be used
        """
        do_dummy_2d_data_aug = (max(patch_size) / min(patch_size)) >= self.anisotropy_threshold
        return do_dummy_2d_data_aug


@PLANNER_REGISTRY.register
class D3V002EstV1(D3V002):
    def get_data_identifier(self, mode: str) -> str:
        """
        D3V001 and D3V002 share the same data preprocessing paramters
        and preprocessor -> thus we use D3V001 data for this plan as well

        Args:
            mode: current operation mode

        Returns:
            str: data identifier
        """
        return f"D3V001_{mode}"

    def create_architecture_planner(
        self,
        model_name: str,
        model_cfg: dict,
        mode: str,
    ) -> BaseBoxesPlanner:
        """
        Create Architecture planner
        """
        estimator = MemoryEstimatorDetection()
        architecture_planner = BoxV002(
            preprocessed_output_dir=self.preprocessed_output_dir,
            save_dir=self.preprocessed_output_dir / "analysis" / f"{self.__class__.__name__}_{mode}",
            estimator=estimator,
            network_cls=MODULE_REGISTRY.get(model_name),
            model_cfg=model_cfg,
        )
        return architecture_planner


@PLANNER_REGISTRY.register
class D3V002DynDtype(D3V002):
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
