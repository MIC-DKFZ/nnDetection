from __future__ import annotations

import os
from pathlib import Path
from itertools import repeat
from multiprocessing import Pool
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Optional, List, TypeVar

import numpy as np
from loguru import logger

from nndet.io.load import load_pickle, save_pickle
from nndet.io.paths import (
    get_case_ids_from_dir,
    get_paths_from_splitted_dir,
    )
from nndet.planning.architecture.abstract import ArchitecturePlannerType
from nndet.preprocessing.preprocessor import PreprocessorType
from nndet.planning.experiment.utils import run_create_label_preprocessed


class AbstractPlanner(ABC):
    def __init__(self,
                 preprocessed_output_dir: os.PathLike,
                 ):
        """
        Base class for experiment planning

        Args:
            preprocessed_output_dir: path to directory where preprocessed
                data will be saved
        """
        super().__init__()
        self.preprocessed_output_dir = Path(preprocessed_output_dir)

        self.transpose_forward = None
        self.transpose_backward = None

        self.anisotropy_threshold = 3
        self.resample_anisotropy_threshold = 3
        self.target_spacing_percentile = 50

        self.data_properties = self.load_data_properties()

    @abstractmethod
    def plan_experiment(self,
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
        raise NotImplementedError

    @abstractmethod
    def create_architecture_planner(self,
                                    model_name: str,
                                    model_cfg: dict,
                                    mode: str,
                                    ) -> ArchitecturePlannerType:
        """
        Create Architecture planner

        Args:
            model_name: name of model to plan for
            model_cfg: config to initialize model for VRAM estimation
            mode: current mode of experiment planner
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def create_preprocessor(plan: Dict) -> PreprocessorType:
        """
        Create Preprocessor
        """
        raise NotImplementedError

    @abstractmethod
    def determine_forward_backward_permutation(self, mode: str):
        """
        Permute dimensions of input. Results should be saved into
        :param:`transpose_forward` and :param:`transpose_backward`

        Args:
            mode: define current operation mode. Typically one of
                '2d' | '3d' | '3dlr1'

        Raises:
            NotImplementedError: Should be overwritten in subcalsses
        """
        raise NotImplementedError

    @abstractmethod
    def determine_target_spacing(self, mode: str) -> np.ndarray:
        """
        Determine target spacing.

        Args:
            mode: define current operation mode. Typically one of
                '2d' | '3d' | '3dlr1'

        Same as nnUNet v21
        https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/experiment_planning/experiment_planner_baseline_3DUNet_v21.py
        """
        raise NotImplementedError

    def load_data_properties(self):
        """
        Load properties from analysis of dataset

        Returns:
            dict: loaded properties
        """
        data_properties_path = self.preprocessed_output_dir / "properties" / "dataset_properties.pkl"
        assert data_properties_path.is_file(), "data properties need to exist. Run data analysis first"
        data_properties = load_pickle(data_properties_path)
        return data_properties

    def get_data_identifier(self, mode: str):
        """
        By default each plan is associated with its own folder

        If only the architecture changed, this can be overwritten
        to use the data from a different plan (useful for dev)
        """
        return f"{self.__class__.__name__}_{mode}"

    def plan_base(self, mode: str) -> Dict:
        """
        Create the base plan

        Args:
            mode: define current operation mode. Typically one of
                '2d' | '3d' | '3dlr1'

        Returns:
            Dict: plan with base attributes
                'mode': selected mode for plan
                `target_spacing`: target to resample data
                `normalization_schemes` normalization type for each modality
                `use_mask_for_norm`: use mask for norm
                `anisotropy_threshold`: threshold used to trigger anisotropy
                    settings
                `resample_anisotropy_threshold`: threshold to trigger different
                    resampling schemes
                `target_spacing_percentile`: target spacing percentile
                    used to create target spacing
                `dim`: dimensionality of data (2 or 3)
                `transpose_forward`: transpose forward order
                `transpose_backward`: transpose back order
                `list_of_npz_files`: files used to preprocessing
        """
        use_nonzero_mask_for_normalization = self.determine_whether_to_use_mask_for_norm()
        logger.info(f"Are we using the nonzero maks for normalization? {use_nonzero_mask_for_normalization}")
        target_spacing = self.determine_target_spacing(mode=mode)
        logger.info(f"Base target spacing is {target_spacing}")
        self.determine_forward_backward_permutation(mode=mode)
        normalization_schemes = self.determine_normalization()
        logger.info(f"Normalization schemes {normalization_schemes}")

        plan = {
            'mode': mode,
            'target_spacing': target_spacing,
            'normalization_schemes': normalization_schemes,
            'use_mask_for_norm': use_nonzero_mask_for_normalization,
            'anisotropy_threshold': self.anisotropy_threshold,
            'resample_anisotropy_threshold': self.resample_anisotropy_threshold,
            'target_spacing_percentile': self.target_spacing_percentile,
            'dim': self.data_properties['dim'],
            "num_modalities": len(list(self.data_properties['modalities'].keys())),
            "all_classes": self.data_properties['all_classes'],
            "num_classes": len(self.data_properties['all_classes']),
            'transpose_forward': self.transpose_forward,
            'transpose_backward': self.transpose_backward,
            'dataset_properties': self.data_properties,
            "planner_id": self.__class__.__name__,
        }
        return plan

    def plan_base_stage(self,
                        base_plan: Dict,
                        model_name: str,
                        model_cfg: dict,
                        ):
        """
        Plan the first stage of training

        Args:
            base_plan: basic plan
            model_name: name of model to plan for
            model_cfg: config to initialize model for VRAM estimation

        Returns:
            dict: properties of stage
                `patch_size`
                `batch_size`
                `architecture` (dict): kwargs for architecture
                `current_spacing`
                `original_spacing`
                `median_shape_transposed`
                `do_dummy_2D_data_aug`
        """
        target_spacing = base_plan['target_spacing']
        spacings = self.data_properties['all_spacings']
        sizes = self.data_properties['all_sizes']

        new_shapes = [np.array(i) / target_spacing * np.array(j) for i, j in zip(spacings, sizes)]
        median_shape = np.median(np.vstack(new_shapes), 0)
        logger.info(f"The median shape of the dataset is {median_shape}")
        max_shape = np.max(np.vstack(new_shapes), 0)
        logger.info(f"The max shape in the dataset is {max_shape}")
        min_shape = np.min(np.vstack(new_shapes), 0)
        logger.info(f"The min shape in the dataset is {min_shape}")

        target_spacing_transposed = np.array(target_spacing)[self.transpose_forward]
        median_shape_transposed = np.array(median_shape)[self.transpose_forward]
        logger.info(f"The transposed median shape of the dataset is {median_shape_transposed}")

        architecture_planner = self.create_architecture_planner(
            model_name=model_name,
            model_cfg=model_cfg,
            mode=base_plan["mode"],
        )
        architecture_plan = architecture_planner.plan(
            target_spacing_transposed=target_spacing_transposed,
            median_shape_transposed=median_shape_transposed,
            transpose_forward=self.transpose_forward,
            mode=base_plan["mode"],
        )

        patch_size = architecture_plan["patch_size"]
        do_dummy_2d_data_aug = (max(patch_size) / min(patch_size)) > self.anisotropy_threshold

        base_plan.update(architecture_plan)
        base_plan["target_spacing_transposed"] = target_spacing_transposed
        base_plan["median_shape_transposed"] = median_shape_transposed
        base_plan["do_dummy_2D_data_aug"] = do_dummy_2d_data_aug
        return base_plan

    def determine_postprocessing(self, mode: str) -> dict:
        """
        Placeholder for the future

        Args:
            mode: define current operation mode. Typically one of
                '2d' | '3d' | '3dlr1'

        Deprecated version returned:
            'keep_only_largest_region'
            'min_region_size_per_class'
            'min_size_per_class'
        """
        logger.warning("No planning for post-processing implemented.")
        return {}

    def determine_normalization(self) -> Dict[int, str]:
        """
        Determine normalization scheme for data

        Returns:
            Dict[int, str]: integer index represents modality and string is
                either `CT` or `nonCT`
        """
        schemes = OrderedDict()
        modalities = self.data_properties['modalities']
        num_modalities = len(list(modalities.keys()))

        for i in range(num_modalities):
            if modalities[i] == "CT":
                schemes[i] = "CT"
            elif modalities[i] == "CT2":
                schemes[i] = "CT2"
            else:
                schemes[i] = "nonCT"
        return schemes

    def determine_whether_to_use_mask_for_norm(self) -> Dict[int, bool]:
        """
        Determine if only foreground values should be used for normalization for all modalities

        Returns:
            Dict[int, bool]: result for each modality
        """
        # only use the nonzero mask for normalization of the cropping based on it resulted in a decrease in
        # image size (this is an indication that the data is something like brats/isles and then we want to
        # normalize in the brain region only)
        modalities = self.data_properties['modalities']
        num_modalities = len(list(modalities.keys()))
        use_mask_for_norm = OrderedDict()

        for i in range(num_modalities):
            if "CT" in modalities[i]:
                use_mask_for_norm[i] = False
            else:
                all_size_reductions = list(self.data_properties["size_reductions"].values())

                if np.median(all_size_reductions) < 3 / 4.:
                    logger.info("using nonzero mask for normalization")
                    use_mask_for_norm[i] = True
                else:
                    logger.info("not using nonzero mask for normalization")
                    use_mask_for_norm[i] = False
        return use_mask_for_norm

    def save_plan(self, plan: dict, mode: str) -> str:
        """
        Save plan

        Args:
            mode: plan mode

        Return:
            str: plan identifier
        """
        self.preprocessed_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        identifier = f"{self.__class__.__name__}_{mode}"
        save_pickle(plan, self.preprocessed_output_dir / f"{identifier}.pkl")
        return identifier

    def run_preprocessing(
            self,
            cropped_data_dir: os.PathLike,
            plan: dict,
            num_processes: int,
    ):
        """
        Runs data preprocessing

        Args:
            cropped_data_dir: base cropped dir
            plan: plan to use for preprocessing
            num_processes: number of processes to use for preprocessing
        """
        preprocessor = self.create_preprocessor(plan=plan)
        preprocessor.run(
            target_spacings=[plan["target_spacing"]],
            identifiers=[plan["data_identifier"]],
            cropped_data_dir=Path(cropped_data_dir),
            preprocessed_output_dir=self.preprocessed_output_dir,
            num_processes=num_processes,
            )
        self.create_labels_tr_preprocessed(
            preprocessed_plan_dir=self.preprocessed_output_dir / plan["data_identifier"],
            dim=3,
            num_processes=num_processes,
        )

    @staticmethod
    def create_labels_tr_preprocessed(
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

        case_ids = get_case_ids_from_dir(source_dir,
                                         remove_modality=False,
                                         pattern="*.npz",
                                         )
        logger.info('Preparing preprocessed evaluation labels')
        if num_processes > 0:
            with Pool(processes=num_processes) as p:
                p.starmap(run_create_label_preprocessed,
                        zip(repeat(source_dir),
                            case_ids,
                            repeat(dim),
                            repeat(target_dir),
                            )
                        )
        else:
            for cid in case_ids:
                run_create_label_preprocessed(source_dir, cid, dim, target_dir)

    @classmethod
    def run_preprocessing_test(cls,
                               preprocessed_output_dir: os.PathLike,
                               splitted_4d_output_dir: os.PathLike,
                               plan: dict,
                               num_processes: int = 0,
                               ):
        """
        Run preprocessing of test data

        Args:
            splitted_4d_output_dir: base dir of splitted data
            plan: plan to use for preprocessing
            num_processes: number of processes to use for preprocessing
        """
        logger.info("Running preprocessing of test cases")
        splitted_4d_output_dir = Path(splitted_4d_output_dir)

        target_dir = Path(preprocessed_output_dir) / plan["data_identifier"] / "imagesTs"
        target_dir.mkdir(parents=True, exist_ok=True)

        cases_processed = get_case_ids_from_dir(
            target_dir,
            remove_modality=False,
            pattern="*.npz",
        )
        cases = get_paths_from_splitted_dir(
            num_modalities=plan["num_modalities"],
            splitted_4d_output_dir=splitted_4d_output_dir,
            test=True,
            labels=False,
            remove_ids=cases_processed,
            )

        logger.info(f"Found {len(cases)} cases for preprocssing in {splitted_4d_output_dir} "
                    f"and {len(cases_processed)} alrady processed cases.")
        preprocessor = cls.create_preprocessor(plan=plan)

        if num_processes > 0:
            with Pool(processes=num_processes) as p:
                p.starmap(preprocessor.run_test,
                        zip(cases,
                            repeat(plan["target_spacing"]),
                            repeat(target_dir),
                            )
                        )
        else:
            for c in cases:
                preprocessor.run_test(c, plan["target_spacing"], target_dir)


PlannerType = TypeVar('PlannerType', bound=AbstractPlanner)
