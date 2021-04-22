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

from os import PathLike
from pathlib import Path
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union, TypeVar

import torch

from nndet.io.load import save_pickle
from nndet.utils.tensor import to_numpy
from nndet.utils.info import maybe_verbose_iterable


class BaseEnsembler(ABC):
    ID = "abstract"

    def __init__(self,
                 properties: Dict[str, Any],
                 parameters: Dict[str, Any],
                 device: Optional[Union[torch.device, str]] = None,
                 **kwargs):
        """
        Base class to containerize and ensemble the predictions of a single case.
        Call :method:`process_batch` to add batched predictions of a case
        to the ensembler and :method:`add_model` to signal the next model
        if multiple models are used.

        Args:
            properties: properties of the patient/case (e.g. tranpose axes)
            parameters: parameters for ensembling
            device: device to use for internal computations
            **kwargs: parameters for ensembling

        Notes:
            Call :method:`add_model` before adding predictions.
        """
        self.model_current = None
        self.model_results = {}
        self.model_weights = {}

        self.properties = properties
        self.case_result: Optional[Dict] = None

        self.parameters = parameters
        self.parameters.update(kwargs)
        
        if device is None:
            self.device = torch.device("cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            raise ValueError(f"Wrong type {type(device)} for device argument.")

    @classmethod
    def from_case(cls,
                  case: Dict,
                  properties: Optional[Dict] = None,
                  parameters: Optional[Dict] = None,
                  **kwargs,
                  ):
        """
        Primary way to instantiate this class. Automatically extracts all
        properties and uses a default set of parameters for ensembling.

        Args:
            case: case which is predicted
            properties: Additional properties. Defaults to None.
            parameters: Additional parameters. Defaults to None.
        """
        return cls(properties=properties, parameters=parameters, **kwargs)

    def add_model(self,
                  name: Optional[str] = None,
                  model_weight: Optional[float] = None,
                  ) -> str:
        """
        This functions signales the ensembler to add a new model for internal
        processing

        Args:
            name: Name of the model. If None, uses counts the models.
            model_weight: Optional weight for this model. Defaults to None.
        """
        if name is None:
            name = len(self.model_weights) + 1
        if name in self.model_results:
            raise ValueError(f"Invalid model name, model {name} is already present")

        if model_weight is None:
            model_weight = 1.0

        self.model_weights[name] = model_weight
        self.model_results[name] = defaultdict(list)
        self.model_current = name
        return name

    @abstractmethod
    @torch.no_grad()
    def process_batch(self, result: Dict, batch: Dict):
        """
        Process a single batch

        Args:
            result: predictions to save and ensemble
            batch: input batch used for predictions (for additional meta data)

        Raises:
            NotImplementedError: Overwrite this function in subclasses for the
            specific use case.

        Warnings:
            Make sure to move cached values to the CPU after they have been
            processed.
        """
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def get_case_result(self, restore: bool = False) -> Dict[str, torch.Tensor]:
        """
        Retrieve the results of a single case

        Args:
            restore: restores predictions in original image space

        Raises:
            NotImplementedError: Overwrite this function in subclasses for the
            specific use case.

        Returns:
            Dict[str, torch.Tensor]: the result of a single case
        """
        raise NotImplementedError

    def update_parameters(self, **parameters: Dict):
        """
        Update internal parameters used for ensembling the results

        Args:
            parameters: parameters to update
        """
        self.parameters.update(parameters)

    @classmethod
    @abstractmethod
    def sweep_parameters(cls) -> Tuple[Dict[str, Any], Dict[str, Sequence[Any]]]:
        """
        Return a set of parameters which can be used to sweep ensembling
        parameters in a postprocessing step

        Returns:
            Dict[str, Any]: default state to start with
            Dict[str, Sequence[Any]]]: Defines the values to search for each
                parameter
        """
        raise NotImplementedError

    def save_state(self,
                   target_dir: Path,
                   name: str,
                   **kwargs,
                   ):
        """
        Save case result as pickle file. Identifier of ensembler will
        be added to the name

        Args:
            target_dir: folder to save result to
            name: name of case
            **kwargs: data to save
        """
        kwargs["properties"] = self.properties
        kwargs["parameters"] = self.parameters

        kwargs["model_current"] = self.model_current
        kwargs["model_results"] = self.model_results
        kwargs["model_weights"] = self.model_weights

        kwargs["case_result"] = self.case_result

        with open(Path(target_dir) / f"{name}_{self.ID}.pt", "wb") as f:
            torch.save(kwargs, f)

    def load_state(self, base_dir: PathLike, case_id: str) -> Dict:
        """
        Path to result file
        """
        ckp = torch.load(str(Path(base_dir) / f"{case_id}_{self.ID}.pt"))
        self._load(ckp)
        return ckp

    def _load(self, state: Dict):
        for key, item in state.items():
            setattr(self, key, item)

    @classmethod
    def from_checkpoint(cls, base_dir: PathLike, case_id: str):
        ckp = torch.load(str(Path(base_dir) / f"{case_id}_{cls.ID}.pt"))
        t = cls(
            properties=ckp["properties"],
            parameters=ckp["parameters"],
        )
        t._load(ckp)
        return t

    @classmethod
    def get_case_ids(cls, base_dir: PathLike):
        return [c.stem.rsplit(f"_{cls.ID}", 1)[0] 
                for c in Path(base_dir).glob(f"*_{cls.ID}.pt")]


class OverlapMap:
    def __init__(self, data_shape: Sequence[int]):
        """
        Handler for overlap map

        Args:
            data_shape: spatial dimensions of data (
                no batch dim and no channel dim!)
        """
        self.overlap_map: torch.Tensor = \
            torch.zeros(*data_shape, requires_grad=False, dtype=torch.float)

    def add_overlap(self, crop: Sequence[slice]):
        """
        Increase values of :param:`self.overlap_map` inside of crop

        Args:
            crop: defines crop. Negative values are assumed to be outside
                of the data and thus discarded
        """
        # discard leading indexes which could be due to batches and channels
        if len(crop) > self.overlap_map.ndim:
            crop = crop[-self.overlap_map.ndim:]

        # clip crop to data shape
        slicer = []
        for data_shape, crop_dim in zip(tuple(self.overlap_map.shape), crop):
            start = max(0, crop_dim.start)
            stop = min(data_shape, crop_dim.stop)
            slicer.append(slice(start, stop, crop_dim.step))
        self.overlap_map[slicer] += 1

    def mean_num_overlap_of_box(self, box: Sequence[int]) -> float:
        """
        Extract mean number of overlaps from a bounding box area

        Args:
            box: defines bounding box (x1, y1, x2, y2, (z1, z2))

        Returns:
            int: mean number of overlaps
        """
        slicer = [slice(int(box[0]), int(box[2])), slice(int(box[1]), int(box[3]))]
        if len(box) == 6:
            slicer.append(slice(int(box[4]), int(box[5])))
        return torch.mean(self.overlap_map[slicer].float()).item()

    def mean_num_overlap_of_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Extract mean number of overlaps from a bounding box area

        Args:
            boxes: defines multiple bounding boxes (x1, y1, x2, y2, (z1, z2))
                [N, dim * 2]

        Returns:
            Tensor: mean number of overlaps per box [N]
        """
        return torch.tensor(
            [self.mean_num_overlap_of_box(box) for box in boxes]).to(
            dtype=torch.float, device=boxes.device)

    def avg(self) -> torch.Tensor:
        """
        Compute mean over all overlaps
        """
        return self.overlap_map.float().median()

    def restore_mean(self, val):
        """
        Generate a new overlap map filled with the specified value
        """
        self.overlap_map = torch.zeros_like(self.overlap_map)
        self.overlap_map = float(val)


def extract_results(source_dir: PathLike,
                    target_dir: PathLike,
                    ensembler_cls: Callable,
                    restore: bool,
                    **params,
                    ) -> None:
    """
    Compute case result from ensembler and save it

    Args:
        source_dir: directory which contains the saved predictions/state from
            the ensembler class
        target_dir: directory to save results
        ensembler_cls: ensembler class for prediction
        restore: if true, the results are converted into the opriginal image
            space
    """
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    for case_id in maybe_verbose_iterable(ensembler_cls.get_case_ids(source_dir)):
        ensembler = ensembler_cls.from_checkpoint(base_dir=source_dir, case_id=case_id)
        ensembler.update_parameters(**params)

        pred = to_numpy(ensembler.get_case_result(restore=restore))

        save_pickle(pred, Path(target_dir) / f"{case_id}_{ensembler_cls.ID}.pkl")


BaseEnsemblerType = TypeVar('BaseEnsemblerType', bound=BaseEnsembler)
