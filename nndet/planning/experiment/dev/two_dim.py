from typing import Dict, List

from loguru import logger

from nndet.planning.experiment import PLANNER_REGISTRY
from nndet.planning.experiment.v001 import D3V001
from nndet.preprocessing.preprocessor import (
    GenericPreprocessor,
    Preprocessor2D,
    Preprocessor2DRGB01,
)


@PLANNER_REGISTRY.register
class D2C004(D3V001):
    @staticmethod
    def create_preprocessor(plan: Dict) -> GenericPreprocessor:
        """
        Create Preprocessor
        """
        if "2d" in plan["mode"]:
            preprocessor = Preprocessor2D(
                norm_scheme_per_modality=plan["normalization_schemes"],
                use_mask_for_norm=plan["use_mask_for_norm"],
                transpose_forward=plan["transpose_forward"],
                intensity_properties=plan["dataset_properties"]["intensity_properties"],
                resample_anisotropy_threshold=plan["resample_anisotropy_threshold"],
            )
        else:
            preprocessor = super().create_preprocessor(plan=plan)
        return preprocessor

    def determine_forward_backward_permutation(self, mode: str):
        """
        Do not transpose data in 2D case
        """
        if mode == "2d":
            self.transpose_forward = [0, 1, 2]
            self.transpose_backward = [0, 1, 2]
        else:
            super().determine_forward_backward_permutation(mode=mode)

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
        if self.data_properties["dim"] == 2:
            logger.info("Using 2D planning...")
            identifiers = []
            # create full resolution 3d plan
            mode = "2d"
            plan_2d = self.plan_base(mode=mode)
            plan_2d["network_dim"] = 2
            plan_2d["dataloader_kwargs"] = {}
            plan_2d["data_identifier"] = self.get_data_identifier(mode=mode)
            plan_2d["postprocessing"] = self.determine_postprocessing(mode=mode)

            plan_2d = self.plan_base_stage(
                plan_2d,
                model_name=model_name,
                model_cfg=model_cfg,
            )
            plan_2d["do_dummy_2D_data_aug"] = False
            # determine if additional low res model needs to be trained
            # plan_2d["trigger_lr1"] = self.trigger_low_res_model(
            #     prev_res_patch_size=plan_2d["patch_size"],
            #     transpose_forward=plan_2d["transpose_forward"],
            # )
            identifiers.append(self.save_plan(plan=plan_2d, mode=mode))
            return identifiers
        else:
            raise RuntimeError("Don't use 2d preprocessor for 3d data.")
            # return super().plan_experiment(
            #     model_name=model_name,
            #     model_cfg=model_cfg,
            # )


@PLANNER_REGISTRY.register
class RGB01C001(D2C004):
    @staticmethod
    def create_preprocessor(plan: Dict) -> GenericPreprocessor:
        """
        Create Preprocessor
        """
        if "2d" in plan["mode"]:
            preprocessor = Preprocessor2DRGB01(
                norm_scheme_per_modality=plan["normalization_schemes"],
                use_mask_for_norm=plan["use_mask_for_norm"],
                transpose_forward=plan["transpose_forward"],
                intensity_properties=plan["dataset_properties"]["intensity_properties"],
                resample_anisotropy_threshold=plan["resample_anisotropy_threshold"],
            )
        else:
            preprocessor = super().create_preprocessor(plan=plan)
        return preprocessor
