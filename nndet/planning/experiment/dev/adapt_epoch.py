import math
from typing import Dict, List

from loguru import logger

from nndet.planning.experiment.v001 import D3V001
from nndet.planning.experiment import PLANNER_REGISTRY


@PLANNER_REGISTRY.register
class D3V001AEP(D3V001):
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
        base_plan.update(self.determine_num_epochs())

        self.plan = self.plan_base_stage(base_plan,
                                         model_name=model_name,
                                         model_cfg=model_cfg,
                                         )
        identifiers.append(self.save_plan(mode=base_plan["mode"]))
        return identifiers

    def determine_num_epochs(self) -> Dict[str, int]:
        num_instances = sum(self.data_properties["num_instances"].values())

        epochs  = math.floor(max(30., 40. + 5. * math.log(num_instances / 500., 2)))
        
        logger.info(f"Found {num_instances} instances in dataset, using {epochs} epochs for training. "
                    "Assuming 2500iter/epoch.")
        return {
            "max_num_epochs": epochs,
        }

    def get_data_identifier(self, mode: str):
        """
        Use D3V001 preprocessed data
        """
        return f"D3V001_{mode}"
