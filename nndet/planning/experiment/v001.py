from typing import Dict, Optional, List

import numpy as np

from nndet.ptmodule import MODULE_REGISTRY
from nndet.planning.experiment import PLANNER_REGISTRY, AbstractPlanner
from nndet.planning.estimator import MemoryEstimatorDetection
from nndet.planning.architecture.boxes import BoxC002
from nndet.preprocessing.preprocessor import GenericPreprocessor


@PLANNER_REGISTRY.register
class D3V001(AbstractPlanner):
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

        base_plan["mode"] = "3d"
        base_plan["data_identifier"] = self.get_data_identifier(mode=base_plan["mode"])
        base_plan["network_dim"] = 3
        base_plan["dataloader_kwargs"] = {}

        self.plan = self.plan_base_stage(base_plan,
                                         model_name=model_name,
                                         model_cfg=model_cfg,
                                         )
        identifiers.append(self.save_plan(mode=base_plan["mode"]))
        return identifiers

    def create_architecture_planner(self,
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
            norm_scheme_per_modality=plan['normalization_schemes'],
            use_mask_for_norm=plan['use_mask_for_norm'],
            transpose_forward=plan['transpose_forward'],
            intensity_properties=plan['dataset_properties']['intensity_properties'],
            resample_anisotropy_threshold=plan['resample_anisotropy_threshold'],
        )
        return preprocessor

    def determine_forward_backward_permutation(self):
        """
        Determine position of z direction (absolute position is defined by z_first)
        Result is
        saved into :param:`transpose_forward` and :param:`transpose_backward`
        """
        spacings = self.data_properties['all_spacings']
        sizes = self.data_properties['all_sizes']
        
        target_spacing = self.determine_target_spacing()
        new_sizes = [np.array(i) / target_spacing * np.array(j) for i, j in zip(spacings, sizes)]

        dims = len(target_spacing)
        max_spacing_axis = np.argmax(target_spacing)
        remaining_axes = [i for i in list(range(dims)) if i != max_spacing_axis]
        # self.transpose_forward = remaining_axes + [max_spacing_axis] # y, x, z
        self.transpose_forward = [max_spacing_axis] + remaining_axes  # z, y, x
        self.transpose_backward = [np.argwhere(np.array(
            self.transpose_forward) == i)[0][0] for i in range(dims)]
