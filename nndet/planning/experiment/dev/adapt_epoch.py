# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Dict, List

from loguru import logger

from nndet.planning.experiment import PLANNER_REGISTRY
from nndet.planning.experiment.v001 import D3V001


@PLANNER_REGISTRY.register
class D3V001AEP(D3V001):
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
        plan_3d.update(self.determine_num_epochs())
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
            plan_3dlr1.update(self.determine_num_epochs())
            identifiers.append(self.save_plan(plan=plan_3dlr1, mode=plan_3dlr1["mode"]))
        return identifiers

    def determine_num_epochs(self) -> Dict[str, int]:
        num_instances = sum(self.data_properties["num_instances"].values())

        epochs = math.floor(max(30.0, 40.0 + 5.0 * math.log(num_instances / 500.0, 2)))

        logger.info(
            f"Found {num_instances} instances in dataset, using {epochs} epochs for training. "
            "Assuming 2500iter/epoch."
        )
        return {
            "max_num_epochs": epochs,
        }

    def get_data_identifier(self, mode: str):
        """
        Use D3V001 preprocessed data
        """
        return f"D3V001_{mode}"
