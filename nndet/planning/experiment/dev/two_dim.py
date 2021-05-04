from typing import Dict, List

from nndet.planning.experiment import PLANNER_REGISTRY, D3V001
from nndet.preprocessing.preprocessor import GenericPreprocessor, Preprocessor2D


@PLANNER_REGISTRY.register
class D2C002(D3V001):
    @staticmethod
    def create_preprocessor(plan: Dict) -> GenericPreprocessor:
        """
        Create Preprocessor
        """
        preprocessor = Preprocessor2D(
            norm_scheme_per_modality=plan['normalization_schemes'],
            use_mask_for_norm=plan['use_mask_for_norm'],
            transpose_forward=plan['transpose_forward'],
            intensity_properties=plan['dataset_properties']['intensity_properties'],
            resample_anisotropy_threshold=plan['resample_anisotropy_threshold'],
        )
        return preprocessor

    def determine_forward_backward_permutation(self):
        """
        Do not transpose data in 2D case
        """
        self.transpose_forward = [0, 1, 2]
        self.transpose_backward = [0, 1, 2]

    def plan_experiment(self,
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
        base_plan = self.plan_base()
        base_plan["postprocessing"] = self.determine_postprocessing()

        base_plan["mode"] = "2d"
        base_plan["data_identifier"] = self.get_data_identifier(mode=base_plan["mode"])
        base_plan["network_dim"] = 2
        base_plan["dataloader_kwargs"] = {}

        self.plan = self.plan_base_stage(base_plan,
                                         model_name=model_name,
                                         model_cfg=model_cfg,
                                         )
        self.plan["do_dummy_2D_data_aug"] = False
        identifiers.append(self.save_plan(mode=base_plan["mode"]))
        return identifiers
