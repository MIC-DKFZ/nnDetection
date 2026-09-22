# SPDX-FileCopyrightText: 2020-2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from itertools import repeat
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List

from loguru import logger

from nndet.io.paths import get_case_ids_from_dir
from nndet.planning.experiment.utils import run_create_label_preprocessed
from nndet.planning.experiment.v002 import D3V002Blosc


class StaticPlanner(D3V002Blosc):
    def plan_experiment(
        self,
        model_name: str,
        model_cfg: Dict,
    ) -> List[str]:
        """
        Plan the whole experiment

        Args:
            model_name: name of model to plan for
            model_cfg: config to initialize model for VRAM estimation

        Returns:
            List: identifiers of created plans
        """
        identifiers = []
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

        # unlike the dynamic planners, a fixed-config plan does not search
        # for additional low-resolution stages -- the architecture is
        # dictated by the pretrained checkpoint, not by a VRAM-driven search.
        plan_3d["preprocessed_data_format"] = "b2nd"
        plan_3d["lowres_identifiers"] = []
        identifiers.append(self.save_plan(plan=plan_3d, mode=plan_3d["mode"]))  # save fullres
        return identifiers

    def determine_forward_backward_permutation(self, mode: str):
        """
        No transpose!
        """
        self.transpose_forward = [0, 1, 2]
        self.transpose_backward = [0, 1, 2]

    def create_labels_tr_preprocessed(
        self,
        preprocessed_plan_dir: Path,
        dim: int,
        num_processes: int = 6,
    ):
        """
        Creates labels for visualization and analysis purposes from
        preprocessed data

        Args:
            preprocessed_plan_dir: path to preprocessed plan dir
            dim: number of spatial dimensions
            num_processes: number of processed to use
        """
        source_dir = preprocessed_plan_dir / "imagesTr"
        target_dir = preprocessed_plan_dir / "labelsTr"
        target_dir.mkdir(parents=True, exist_ok=True)

        case_ids = get_case_ids_from_dir(
            source_dir,
            remove_modality=False,
            pattern=f"*.{self.preprocessed_data_format}",
        )
        logger.info("Preparing preprocessed evaluation labels")
        if num_processes > 0:
            with Pool(processes=num_processes) as p:
                p.starmap(
                    run_create_label_preprocessed,
                    zip(
                        repeat(source_dir),
                        case_ids,
                        repeat(dim),
                        repeat(target_dir),
                        repeat(self.preprocessed_data_format),
                    ),
                )
        else:
            for cid in case_ids:
                run_create_label_preprocessed(source_dir, cid, dim, target_dir, self.preprocessed_data_format)
