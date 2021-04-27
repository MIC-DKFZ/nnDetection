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

from __future__ import annotations
from os import PathLike

import pickle

from pathlib import Path
from typing import Dict, Sequence, Callable

from nndet.io.paths import get_case_ids_from_dir


class DatasetAnalyzer:
    def __init__(self,
                 cropped_output_dir: PathLike,
                 preprocessed_output_dir: PathLike,
                 data_info: dict,
                 num_processes: int,
                 overwrite: bool = True,
                 ):
        """
        Class to analyse a dataset
        :func:`analyze_dataset` saves result into `dataset_properties.pkl`

        Args:
            cropped_output_dir: path to directory where prepared/cropped data is saved
            data_info: additional information about the data
                `modalities`: numeric dict which maps modalities to strings (e.g. `CT`)
                `labels`: numeric dict which maps segmentation to classes
            num_processes: number of processes to use for analysis
            overwrite: overwrite existing properties
        """
        self.cropped_output_dir = Path(cropped_output_dir)
        self.cropped_data_dir = self.cropped_output_dir / "imagesTr"
        self.preprocessed_output_dir = Path(preprocessed_output_dir)
        self.save_dir = self.preprocessed_output_dir / "properties"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.num_processes = num_processes
        self.overwrite = overwrite

        self.sizes = self.spacings = None
        self.data_info = data_info

        self.case_ids = sorted(get_case_ids_from_dir(
            self.cropped_output_dir / "imagesTr", pattern="*.npz", remove_modality=False))
        self.props_per_case_file = self.save_dir / "props_per_case.pkl"
        self.intensity_properties_file = self.save_dir / "intensity_properties.pkl"

    def analyze_dataset(self,
                        properties: Sequence[Callable[[DatasetAnalyzer], Dict]],
                        ) -> Dict:
        """
        Analyze dataset
        Result is also saved in cropped_output_dir as `dataset_properties.pkl`

        Args:
            properties: properties to analyze over dataset

        Returns:
            Dict: filled with computed results
        """
        props = {"dim": self.data_info["dim"]}
        for property_fn in properties:
            props.update(property_fn(self))

        with open(self.save_dir / "dataset_properties.pkl", "wb") as f:
            pickle.dump(props, f)
        return props
