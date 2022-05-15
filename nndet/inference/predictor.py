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

import collections
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar, Union

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

from nndet.core.abstract import AbstractDetector
from nndet.io.load import save_pickle
from nndet.io.patching import create_grid, save_get_crop
from nndet.io.transforms import NoOp
from nndet.io.transforms.base import AbstractTransform
from nndet.utils import maybe_verbose_iterable, to_device

torch_device = Union[torch.device, str]


class Predictor:
    def __init__(
        self,
        ensembler: Dict[str, Callable],
        models: Sequence[AbstractDetector],
        crop_size: Sequence[int],
        overlap: float = 0.5,
        tile_keys: Sequence[str] = ("data",),
        model_keys: Sequence[str] = ("data",),
        tta_transforms: Sequence[AbstractTransform] = (NoOp(),),
        tta_inverse_transforms: Sequence[AbstractTransform] = (NoOp(),),
        pre_transform: AbstractTransform = None,
        post_transform: AbstractTransform = None,
        batch_size: int = 4,
        model_weights: Sequence[float] = None,
        device: torch_device = "cuda:0",
        ensemble_on_device: bool = True,
    ):
        """
        Predict entire cases with TTA and Model-Ensembling

        Workflow
        - Load whole patient
        -> create predictor from patient
        - tile patient
        * for each model:
            * for each batch (batches of tiles):
                * for each tta transform:
                    - pre transform
                    - tta transform
                    - post transform
                    - predict batch
                    - inverse tta transform
                    - forward predictions and batch to ensembler classes
        <- return patient result

        Args:
            ensembler: Callable to instantiate ensembler from case and
                properties, The keys need to be obtained from the constructor
                function of the respective ensembler class!
            models: models to ensemble
            crop_size: size of each crop (for most cases this should be
                the same as in training)
            overlap: overlap of crops
            tile_keys: keys which are tiles
            model_keys: this kyes are passed as positional arugments to the
                model
            tta_transforms: tta transformations
            tta_inverse_transforms: inverse tta transformation
            pre_transform: transform which is performed before every tta
                transform
            post_transform: transform which is performed after every tta
                transform
            batch_size: batch size to use for prediction
            model_weights: additional weighting of individual models
            device: device used for prediction
            ensemble_on_device: The results will be passed to the ensembler
                class with the current device. The ensembler needs to make
                sure to avoid memory leaks!
        """
        self.ensemble_on_device = ensemble_on_device
        self.device = device

        self.ensembler_fns = ensembler
        self.ensembler = {}

        self.models = models
        self.model_weights = (
            [1.0] * len(models) if model_weights is None else model_weights
        )

        self.crop_size = crop_size
        self.overlap = overlap
        self.tile_keys = tile_keys
        self.model_keys = model_keys

        self.batch_size = batch_size

        if len(tta_transforms) != len(tta_inverse_transforms):
            raise ValueError("Every tta transform needs a reverse transform")
        self.tta_transforms = tta_transforms
        self.tta_inverse_transforms = tta_inverse_transforms
        self.post_transform = post_transform
        self.pre_transform = pre_transform

        self.grid_mode = "symmetric"  # FIXME
        self.save_get_mode = "shift"  # FIXME
        logger.info(
            f"Initialized predictor with patch size {self.crop_size} "
            f"batch size {self.batch_size} overlap {self.overlap}"
        )

    def save_case(
        self,
        result: Dict,
        target_dir: Path,
        case_id: str,
    ) -> None:
        """
        Save prediction from a single case. In contrast to the `save_dir`
        option from the `predict_case` function this will not save any state
        information of the ensembler and is thus inly intended for efficient
        saving of the final predictions of a case. (Usually final results
        will be saved in the form of numpy arrays)

        Args:
            result: results of case to save. The keys of this dict
                need to correspond to the keys of the provided ensemblers.
            target_dir: directory to save data
            case_id: name of case to save result
        """
        for key, item in result.items():
            self.ensembler[key].save_result(
                data=item,
                target_dir=target_dir,
                case_name=case_id,
            )

    def predict_case(
        self,
        case: Dict,
        properties: Optional[Dict] = None,
        save_dir: Optional[Union[Path, str]] = None,
        case_id: Optional[str] = None,
        restore: bool = False,
    ) -> dict:
        """
        Load and predict a single case.

        Args:
            case: data of a single case
            properties: additional properties of the case. E.g. to
                restore prediction in original image space
            save_dir: directory to save predictions
            case_id: used for saving
            restore: restore prediction in original image space
                ("revert" preprocessing)

        Returns:
            dict: result of each ensembler (converted to numpy)
        """
        tic = time.perf_counter()
        self.ensembler = {}  # clear any previous ensembler classes
        for name, fn in self.ensembler_fns.items():
            if name in self.ensembler:
                raise ValueError(
                    f"{name} is already in ensemblers of predictor, "
                    "can not use multiple ensembler of same type."
                )
            _ensembler = fn(case, properties=properties)
            if name != _ensembler.ID:
                raise RuntimeError(
                    f"Provided predictor key {name} does not match ensembler ID {_ensembler.ID}!"
                )
            self.ensembler[name] = _ensembler

        tiles = self.tile_case(case)
        self.predict_tiles(tiles)

        result = {
            key: value.get_case_result(restore=restore)
            for key, value in self.ensembler.items()
        }
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            for ensembler in self.ensembler.values():
                ensembler.save_state(save_dir, name=case_id)
            save_pickle(properties, save_dir / f"{case_id}_properties.pkl")
        toc = time.perf_counter()
        logger.info(f"Prediction took {toc - tic} s")
        return result

    def tile_case(
        self, case: dict, update_remaining: bool = True
    ) -> Sequence[Dict[str, np.ndarray]]:
        """
        Create patches from whole patient for prediction

        Args:
            case: data of a single case
            update_remaining: properties from case which are not tiles
                are saved into all patches

        Returns:
            Sequence[Dict[str, np.ndarray]]: extracted crops from case
                and added new key:
                    `tile_origin`: Sequence[int] offset of tile relative
                        to case origin
        """
        dshape = case[self.tile_keys[0]].shape
        overlap = [int(c * self.overlap) for c in self.crop_size]
        crops = create_grid(
            cshape=self.crop_size,
            dshape=dshape[1:],
            overlap=overlap,
            mode=self.grid_mode,
        )

        tiles = []
        for crop in crops:
            try:
                # try selected extraction mode
                tile = {
                    key: save_get_crop(case[key], crop, mode=self.save_get_mode)[0]
                    for key in self.tile_keys
                }
                _, tile["tile_origin"], tile["crop"] = save_get_crop(
                    case[self.tile_keys[0]], crop, mode=self.save_get_mode
                )
            except RuntimeError:
                # fallback to padding
                logger.warning(
                    "Path size is bigger than whole case, padding case to match patch size"
                )
                tile = {
                    key: save_get_crop(
                        case[key],
                        crop,
                        mode="constant",
                        constant_values=0,
                    )[0]
                    for key in self.tile_keys
                }
                _, tile["tile_origin"], tile["crop"] = save_get_crop(
                    case[self.tile_keys[0]],
                    crop,
                    mode="constant",
                    constant_values=0,
                )

            if update_remaining:
                tile.update(
                    {
                        key: item
                        for key, item in case.items()
                        if key not in self.tile_keys
                    }
                )
            tiles.append(tile)
        return tiles

    @torch.no_grad()
    def predict_tiles(self, tiles: Sequence[Dict]) -> None:
        """
        Predict tiles of a single case with ensembling and tta. Results
        are saved inside ensemblers

        Args:
            tiles: tiles from single case
        """
        dataloader = DataLoader(
            tiles,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=slice_collate,
        )
        for model_idx, (model, model_weight) in enumerate(
            zip(self.models, self.model_weights)
        ):
            logger.info(
                f"Predicting model {model_idx + 1} of "
                f"{len(self.models)} with weight {model_weight}."
            )

            model.to(device=self.device)
            model.eval()

            for t, (transform, inverse_transform) in enumerate(
                maybe_verbose_iterable(
                    list(zip(self.tta_transforms, self.tta_inverse_transforms)),
                    desc="Transform",
                    position=0,
                )
            ):
                for ensembler in self.ensembler.values():
                    ensembler.add_model(
                        name=f"model{model_idx}_t{t}", model_weight=model_weight
                    )

                for batch_num, batch in enumerate(
                    maybe_verbose_iterable(dataloader, desc="Crop", position=1)
                ):
                    self.predict_with_transformation(
                        model=model,
                        batch=batch,
                        batch_num=batch_num,
                        transform=transform,
                        inverse_transform=inverse_transform,
                    )

            model.cpu()
            torch.cuda.empty_cache()

    def predict_with_transformation(
        self,
        model: AbstractDetector,
        batch: Dict,
        batch_num: int,
        transform: Callable,
        inverse_transform: Callable,
    ):
        """
        Run prediction with the specified transformations

        Args:
            model: model to predict
            batch: input batch to model
            batch_num: batch index
            transform: transform to apply to batch.
            inverse_transform: inverse transform to apply to batch and resuls
        """
        batch = to_device(batch, device=self.device)
        if self.pre_transform is not None:
            batch = self.pre_transform(**batch)

        transformed = transform(**batch)

        if self.post_transform is not None:
            transformed = self.post_transform(**transformed)

        inp = [transformed[key] for key in self.model_keys]
        with torch.cuda.amp.autocast():
            result = model.inference_step(*inp, batch_num=batch_num)

        # inverse transformation (_data_shapes need for mirror)
        if "_data_shapes" in transformed:
            result["_data_shapes"] = transformed["_data_shapes"]
        result = inverse_transform(**result)

        if not self.ensemble_on_device:
            result = to_device(result, device="cpu")

        for ensembler in self.ensembler.values():
            ensembler.process_batch(result=result, batch=batch)


def slice_collate(batch: List[Any]):
    """
    Add support for slices to collate function

    Args:
        batch: batch to collate

    Returns:
        Any: collated items
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(batch[0], slice):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: slice_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(slice_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        transposed = zip(*batch)
        return [slice_collate(samples) for samples in transposed]
    else:
        return torch.utils.data._utils.collate.default_collate(batch)


PredictorType = TypeVar("PredictorType", bound=Predictor)
