"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np

from os import PathLike
from loguru import logger
from abc import ABC, abstractmethod
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Sequence, List, Tuple, TypeVar, Union
from itertools import repeat

from nndet.io.transforms.instances import instances_to_boxes_np
from nndet.io.paths import get_case_ids_from_dir, get_case_id_from_path
from nndet.io.load import load_case_cropped, save_pickle
from nndet.preprocessing.resampling import resample_patient
from nndet.io.crop import ImageCropper


class AbstractPreprocessor(ABC):
    DATA_ID = "abstractdata"

    def __init__(self, **kwargs):
        """
        Interface for preprocessor
        """
        for key, item in kwargs.items():
            setattr(self, key, item)

    @abstractmethod
    def run(self,
            target_spacings: Sequence[Sequence[float]],
            identifiers: Sequence[str],
            cropped_data_dir: Path,
            preprocessed_output_dir: Path,
            num_processes: int,
            force_separate_z=None,
            ):
        """
        Run preprocessing

        Args:
            target_spacings: target spacing for each case
            identifiers: identifier strings used to name the directory
            cropped_data_dir: source directory
            preprocessed_output_dir: target directory
            num_processes: number of processes used for preprocessing
            force_separate_z: force independent resampling of z direction
        """
        raise NotImplementedError

    @abstractmethod
    def run_test(self,
                 data_files,
                 target_spacing,
                 target_dir: PathLike,
                 ) -> None:
        """
        Preprocess and save test data

        Args:
            data_files: path to data files
            target_spacing: spacing to resample
            target_dir: directory to save data to
        """
        raise NotImplementedError

    @abstractmethod
    def preprocess_test_case(self,
                             data_files,
                             target_spacing,
                             seg_file=None,
                             force_separate_z=None,
                             ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Preprocess a test file

        Args:
            data_files: path to data files
            target_spacing: spacing to resample
            seg_file: optional segmentation file
            force_separate_z: separate resampling in z direction

        Returns:
            np.ndarray: preprocessed data [C, dims]
            np.ndarray: preprocessed segmentation [1, dims]
            dict: updated properties
        """
        raise NotImplementedError


class GenericPreprocessor:
    DATA_ID = "Generic"

    def __init__(self,
                 norm_scheme_per_modality: Dict[int, str],
                 use_mask_for_norm: Dict[int, bool],
                 transpose_forward: Sequence[int],
                 intensity_properties: Dict[int, Dict] = None,
                 resample_anisotropy_threshold: float = 3.,
                 ):
        """
        Preprocess data

        Args:
            norm_scheme_per_modality: integer index represents modality and string is
                either `CT`, `CT2`, 'BValRaw'. Other modalities are treated the with zeo mean and unit std.
            use_mask_for_norm: only foreground values should be used for normalization
                (defined for each modality)
            transpose_forward: transpose input data
            intensity_properties: Intensity properties of foreground over the dataset.
                Evaluated statistics: `median`; `mean`; `std`; `min`; `max`;
                `percentile_99_5`; `percentile_00_5`
                `local_props`: contains a dict (with case ids) where statistics
                where computed per case

        Overwrites:
            :self:`data_id`: unique identifier of GenericPreprocessor
        """
        self.resample_anisotropy_threshold = resample_anisotropy_threshold
        self.intensity_properties = intensity_properties
        self.transpose_forward = list(transpose_forward)
        self.use_mask_for_norm = use_mask_for_norm
        self.norm_scheme_per_modality = norm_scheme_per_modality

        self.norm_schemes = {
            "CT": self.normalize_ct,
            "CT2": self.normalize_ct2,
            "CT3": self.normalize_ct,
            "raw": self.no_norm,
        }

    def run(self,
            target_spacings: Sequence[Sequence[float]],
            identifiers: Sequence[str],
            cropped_data_dir: Path,
            preprocessed_output_dir: Path,
            num_processes: Union[int, Sequence[int]],
            overwrite: bool = False,
            ):
        """
        Run preprocessing

        Args:
            target_spacings: target spacing for each case
            identifiers: identifier strings used to name the directory
            cropped_data_dir: source directory
            preprocessed_output_dir: target directory
            num_processes: number of processes used for preprocessing
            overwrite: overwrite existing data
        """
        case_ids, num_processes = self.initialize_run(
            target_spacings=target_spacings,
            cropped_data_dir=cropped_data_dir,
            preprocessed_output_dir=preprocessed_output_dir,
            num_processes=num_processes,
        )

        for identifier, spacing, nump in zip(identifiers, target_spacings, num_processes):
            logger.info(f"+++ Preprocessing {identifier} +++")
            output_dir_stage = preprocessed_output_dir / identifier / "imagesTr"
            output_dir_stage.mkdir(parents=True, exist_ok=True)

            if not overwrite:
                case_ids_npz_present = get_case_ids_from_dir(
                    output_dir_stage, remove_modality=False, pattern="*.npz")
                case_ids_pkl_present = get_case_ids_from_dir(
                    output_dir_stage, remove_modality=False, pattern="*.pkl")
                case_ids_present = list(set.intersection(set(case_ids_npz_present), set(case_ids_pkl_present)))
                logger.info(f"Skipping case ids which are already present {case_ids_present}")
                _case_ids = list(filter(lambda x: x not in case_ids_present, case_ids))
            else:
                _case_ids = case_ids

            logger.info(f"Running preprocessing on {_case_ids}")
            if nump == 0:
                for _cid in _case_ids:
                    self.run_process(spacing, _cid, output_dir_stage, cropped_data_dir)
            else:
                with Pool(processes=nump) as p:
                    p.starmap(self.run_process,
                              zip(repeat(spacing),
                                  _case_ids,
                                  repeat(output_dir_stage),
                                  repeat(cropped_data_dir),
                                  ))

    def initialize_run(self,
                       target_spacings: Sequence[Sequence[float]],
                       cropped_data_dir: Path,
                       preprocessed_output_dir: Path,
                       num_processes: int,
                       ) -> Tuple[List[str], List[int]]:
        """
        Prepare preprocessing run

        Args:
            target_spacings: target spacings
            cropped_data_dir: source dir
            preprocessed_output_dir: target dir
            num_processes: number of processes

        Returns:
            List[str]: case ids from source dir
            List[int]: number of processes for each stage
        """
        logger.info("Initializing preprocessing")
        logger.info(f"Folder with cropped data: {cropped_data_dir}")
        logger.info(f"Folder for preprocessed data:{preprocessed_output_dir}")
        for key, modality in self.norm_scheme_per_modality.items():
            if modality in self.norm_schemes:
                logger.info(f"Found normalization scheme for {modality}")
            else:
                logger.info(f"No normalization scheme for {modality} using zero mean unit std.")
        preprocessed_output_dir.mkdir(parents=True, exist_ok=True)

        num_stages = len(target_spacings)
        if not isinstance(num_processes, Sequence):
            num_processes = [num_processes] * num_stages
        assert len(num_processes) == num_stages

        case_ids = get_case_ids_from_dir(
            cropped_data_dir, pattern="*.npz", remove_modality=False)
        return case_ids, num_processes

    def run_process(self,
                    target_spacing: Sequence[float],
                    case_id: str,
                    output_dir_stage: Path,
                    cropped_data_dir: Path,
                    ) -> None:
        """
        Process a single case
        Result is saved into :param:`output_dir_stage`

        Args:
            target_spacing: target spacing for processed case
            case_id: case identifier
            output_dir_stage: path to output directory
            cropped_data_dir: path to source directory
        """
        data, seg, properties = load_case_cropped(cropped_data_dir, case_id)
        seg = seg[None]

        data, seg, properties = self.apply_process(
            data, target_spacing, properties, seg)
        properties["use_nonzero_mask_for_norm"] = self.use_mask_for_norm

        data = data.astype(np.float32)
        seg = seg.astype(np.int32)

        candidates = self.compute_candidates(
            data=data,
            seg=seg,
            properties=properties,
            )

        logger.info(f"Saving: {case_id} into {output_dir_stage}.")
        np.savez_compressed(str(output_dir_stage / f"{case_id}.npz"),
                            data=data,
                            seg=seg,
                            )

        save_pickle(candidates, output_dir_stage / f"{case_id}_boxes.pkl")
        save_pickle(properties, output_dir_stage / f"{case_id}.pkl")

    def apply_process(self,
                      data: np.ndarray,
                      target_spacing: Sequence[float],
                      properties: dict,
                      seg: np.ndarray = None,
                      ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Applies all preprocessing steps to data and segmentation

        Args:
            data: input data
            target_spacing: target spacing (not! transposed)
            properties: dict with properties for preprocessing
            seg: input segmentation

        Returns:
            np.ndarray: preprocessed data
            np.ndarray: preprocessed segmentation
            dict: updated properties
        """
        data, seg, original_spacing, target_spacing, before = self.transpose(
            data, seg, properties["original_spacing"], target_spacing)

        data, seg, after = self.resample(
            data, seg, original_spacing, target_spacing)

        # logger.info(f"\nBefore: {before} \nAfter: {after}\n")

        if seg is not None:
            seg[seg < -1] = 0

        properties["size_after_resampling"] = data[0].shape
        properties["spacing_after_resampling"] = after["spacing"]

        data = self.normalize(data, seg)
        return data, seg, properties

    def transpose(self,
                  data: np.ndarray,
                  seg: np.ndarray,
                  original_spacing: Sequence[float],
                  target_spacing: Sequence[float]) -> Tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Transpose data, segmentation and spacings

        Args:
            data: input data
            seg: input segmentation
            original_spacing: original spacing
            target_spacing: target spacing

        Returns:
            np.ndarray: transposed data
            np.ndarray: transposed segmentation
            np.ndarray: transposed original spacing
            np.ndarray: transposed target spacing
            dict: values for debugging
        """
        data = data.transpose((0, *[i + 1 for i in self.transpose_forward]))
        seg = seg.transpose((0, *[i + 1 for i in self.transpose_forward]))
        _original_spacing = np.array(original_spacing)[self.transpose_forward]
        _target_spacing = np.array(target_spacing)[self.transpose_forward]

        before = {
            "spacing": original_spacing,
            "transpose": self.transpose_forward,
            "spacing_transposed": _original_spacing,
            "shape (transposed": data.shape,
            }

        return data, seg, _original_spacing, _target_spacing, before

    def resample(self,
                 data: np.ndarray,
                 seg: np.ndarray,
                 original_spacing: Sequence[float],
                 target_spacing: Sequence[float],
                 ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Resample data and segmentation to new spacing

        Args:
            data: input data
            seg: input segmentation
            original_spacing: original spacing
            target_spacing: target spacing

        Returns:
            np.ndarray: resampled data
            np.ndarray: resampled segmentation
            dict: properties after resampling
                `spacing`: spacing after resampling
                `shape (resampled)`: shape after resampling
        """
        original_spacing = np.array(original_spacing)
        target_spacing = np.array(target_spacing)
        data[np.isnan(data)] = 0
        data, seg = resample_patient(data,
                                     seg,
                                     original_spacing,
                                     target_spacing,
                                     order_data=3,
                                     order_seg=0,
                                     force_separate_z=False,
                                     order_z_data=9999,
                                     order_z_seg=9999,
                                     separate_z_anisotropy_threshold=
                                     self.resample_anisotropy_threshold,
                                     )

        after = {
            "spacing": target_spacing,
            "shape (resampled)": data.shape,
            }
        return data, seg, after

    def normalize(self, data: np.ndarray, seg: np.ndarray) -> np.ndarray:
        """
        Normalize data with correct scheme

        Args:
            data: input data
            seg: input data

        Returns:
            np.ndarray: normalized data
        """
        assert len(self.norm_scheme_per_modality) == len(data), \
            f"norm_scheme_per_modality must have as many entries as data has modalities"
        assert len(self.use_mask_for_norm) == len(data), \
            f"use_mask_for_norm must have as many entries as data has modalities"

        for c in range(len(data)):
            scheme = self.norm_scheme_per_modality[c]
            scheme_fn = self.norm_schemes.get(scheme, self.normalize_other)
            data = scheme_fn(data, seg, c, self.use_mask_for_norm)
        return data

    def normalize_ct(self,
                     data: np.ndarray,
                     seg: np.ndarray,
                     modality: int,
                     use_nonzero_mask: Dict[int, bool],
                     ) -> np.ndarray:
        """
        clip to lb and ub from train data foreground and use foreground mn and sd from training data
        (This uses the foreground mean and std!)
        Args:
            data: data to normalize [C, dims]
            seg: segmentation [C, dims]
            modality: current modality
            use_nonzero_mask: use non zero region for normalization and set all values
                outside to zero [C]

        Returns:
            np.ndarray: normalized data (only modality channel was changes)
        """
        assert self.intensity_properties is not None, \
            "ERROR: if there is a CT then we need intensity properties"
        mean_intensity = self.intensity_properties[modality]['mean']
        std_intensity = self.intensity_properties[modality]['std']
        lower_bound = self.intensity_properties[modality]['percentile_00_5']
        upper_bound = self.intensity_properties[modality]['percentile_99_5']
        data[modality] = np.clip(data[modality], lower_bound, upper_bound)
        data[modality] = (data[modality] - mean_intensity) / std_intensity
        if use_nonzero_mask[modality]:
            data[modality][seg[-1] < 0] = 0
        return data

    def normalize_ct2(self, data: np.ndarray, seg: np.ndarray, modality: int,
                      use_nonzero_mask: Dict[int, bool]) -> np.ndarray:
        """
        clip to lb and ub from train data foreground, use mn and sd from each case for normalization
        (This uses mean and std from whole case!)

        Args:
            data: data to normalize [C, dims]
            seg: segmentation [C, dims]
            modality: current modality
            use_nonzero_mask: use non zero region for normalization [C]

        Returns:
            np.ndarray: normalized data (only modality channel was changes)
        """
        assert self.intensity_properties is not None, \
            "ERROR: if there is a CT then we need intensity properties"
        lower_bound = self.intensity_properties[modality]['percentile_00_5']
        upper_bound = self.intensity_properties[modality]['percentile_99_5']
        mask = (data[modality] > lower_bound) & (data[modality] < upper_bound)
        data[modality] = np.clip(data[modality], lower_bound, upper_bound)
        mn = data[modality][mask].mean()
        sd = data[modality][mask].std()
        data[modality] = (data[modality] - mn) / sd
        if use_nonzero_mask[modality]:
            data[modality][seg[-1] < 0] = 0
        return data

    def normalize_ct3(self,
                     data: np.ndarray,
                     seg: np.ndarray,
                     modality: int,
                     use_nonzero_mask: Dict[int, bool],
                     ) -> np.ndarray:
        """
        clip to lb and ub from train data foreground and use foreground mn
        and sd from training data (This uses the foreground mean and std!)
        Use this if channels are overloaded with spatial information (
        in case of CT)

        Args:
            data: data to normalize [C, dims]
            seg: segmentation [C, dims]
            modality: current modality
            use_nonzero_mask: use non zero region for normalization and set all values
                outside to zero [C]

        Returns:
            np.ndarray: normalized data (only modality channel was changes)
        """
        assert self.intensity_properties is not None, \
            "ERROR: if there is a CT then we need intensity properties"
        mean_intensity = np.mean([k["mean"] for k in self.intensity_properties.values()])
        # the intensity values are not independent but we do not have enough information here
        std_intensity = np.sqrt(np.sum([k["std"] ** 2 for k in self.intensity_properties.values()]))

        lower_bound = np.mean([k["percentile_00_5"] for k in self.intensity_properties.values()])
        upper_bound = np.mean([k["percentile_99_5"] for k in self.intensity_properties.values()])

        data[modality] = np.clip(data[modality], lower_bound, upper_bound)
        data[modality] = (data[modality] - mean_intensity) / std_intensity
        if use_nonzero_mask[modality]:
            data[modality][seg[-1] < 0] = 0
        return data

    def normalize_other(self, data: np.ndarray, seg: np.ndarray, modality: int,
                        use_nonzero_mask: Dict[int, bool]) -> np.ndarray:
        """
        Zero mean and unit std

        Args:
            data: data to normalize [C, dims]
            seg: segmentation [C, dims]
            modality: current modality
            use_nonzero_mask: use non zero region for normalization [C]

        Returns:
            np.ndarray: normalized data (only modality channel was changes)
        """
        if use_nonzero_mask[modality]:
            mask = seg[-1] >= 0
        else:
            mask = np.ones(seg.shape[1:], dtype=bool)
        data[modality][mask] = (data[modality][mask] - data[modality][mask].mean()) / \
                               (data[modality][mask].std() + 1e-8)
        data[modality][mask == 0] = 0
        return data

    def no_norm(self, data: np.ndarray, seg: np.ndarray, modality: int,
                use_nonzero_mask: Dict[int, bool]) -> np.ndarray:
        """
        No normalization only masking

        Args:
            data: data to normalize [C, dims]
            seg: segmentation [C, dims]
            modality: current modality
            use_nonzero_mask: use non zero region for normalization [C]

        Returns:
            np.ndarray: masked data (only modality channel was changed)
        """
        if use_nonzero_mask[modality]:
            mask = seg[-1] >= 0
        else:
            mask = np.ones(seg.shape[1:], dtype=bool)
        data[modality][mask == 0] = 0
        return data

    @staticmethod
    def compute_candidates(
                           data: np.ndarray,
                           seg: np.ndarray,
                           properties: dict,
                           ) -> dict:
        """
        Precompute candidate sampling positions for training
        This method computes the bounding boxes of each present
        instance which can be used to oversample foreground effectively.
        
        Args:
            data: data after resampling
            seg: instance segmentation after resampling
        """        
        dim = data.ndim - 1
        boxes = instances_to_boxes_np(seg[0], dim=dim)[0]

        instances = np.unique(seg)
        instances = instances[instances > 0].astype(np.int32)  # [N]
        instances = instances.tolist()

        instances_props = properties["instances"]
        labels = [int(instances_props[str(i)]) for i in instances]
        
        assert (len(boxes) == len(instances)) or ((boxes.size == 0) and (len(instances) == 0))
        assert len(labels) == len(instances)
        return {
            "boxes": boxes,
            "instances": instances,
            "labels": labels,
        }

    def run_test(self,
                 data_files,
                 target_spacing,
                 target_dir: PathLike,
                 ) -> None:
        """
        Preprocess and save test data

        Args:
            data_files: path to data files
            target_spacing: spacing to resample
            target_dir: directory to save data to
        """
        target_dir = Path(target_dir)
        data, seg, properties = self.preprocess_test_case(
            data_files=data_files,
            target_spacing=target_spacing,
        )
        case_id = get_case_id_from_path(str(data_files[0]), remove_modality=True)
        np.savez_compressed(str(target_dir / f"{case_id}.npz"), data=data)
        save_pickle(properties, target_dir / f"{case_id}")

    def preprocess_test_case(self,
                             data_files,
                             target_spacing,
                             seg_file=None,
                             ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Preprocess a test file

        Args:
            data_files: path to data files
            target_spacing: spacing to resample
            seg_file: optional segmentation file

        Returns:
            np.ndarray: preprocessed data
            np.ndarray: preprocessed segmentation
            dict: updated properties
        """
        data, seg, properties = ImageCropper.load_crop_from_list_of_files(
            data_files, seg_file)
        data, seg, properties = self.apply_process(
            data=data,
            target_spacing=target_spacing,
            properties=properties,
            seg=seg,
        )
        return data.astype(np.float32), seg.astype(np.int32), properties


PreprocessorType = TypeVar('PreprocessorType', bound=AbstractPreprocessor)
