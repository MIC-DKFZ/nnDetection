import os
from abc import ABC
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, Sequence, Type

from loguru import logger

from nndet.inference.ensembler.detection import (
    BoxEnsemblerSelective,
    BoxEnsemblerSelective2D,
)
from nndet.inference.ensembler.segmentation import SegmentationEnsembler
from nndet.inference.helper import predict_dir
from nndet.inference.loading import get_loader_fn
from nndet.inference.predictor import Predictor
from nndet.inference.sweeper import BoxSweeper
from nndet.inference.transforms import Inference2D, get_tta_transforms
from nndet.ptmodule.module import LightningBaseModule


class PredictionMixin(ABC):
    @staticmethod
    def get_ensembler_cls(key: Hashable, dim: int) -> Callable:
        """
        Get ensembler classes to combine multiple predictions
        Needs to be overwritten in subclasses!
        """
        ...

    @classmethod
    def get_predictor(
        cls,
        plan: Dict,
        models: Sequence[LightningBaseModule],
        num_tta_transforms: int = None,
        **kwargs,
    ) -> Type[Predictor]:
        """
        Get predictor
        Needs to be overwritten in subclasses!
        """
        ...

    def sweep(
        self,
        cfg: dict,
        save_dir: os.PathLike,
        train_data_dir: os.PathLike,
        case_ids: Sequence[str],
        run_prediction: bool = True,
    ) -> Dict[str, Any]:
        """
        Sweep parameters to find the best predictions
        Needs to be overwritten in subclasses!

        Args:
            cfg: config used for training
            save_dir: save dir used for training
            train_data_dir: directory where preprocessed training/validation
                data is located
            case_ids: case identifies to prepare and predict
            run_prediction: predict cases
            **kwargs: keyword arguments passed to predict function
        """
        ...


class BoxPredictionMixin(PredictionMixin):
    @staticmethod
    def get_ensembler_cls(key: Hashable, dim: int) -> Callable:
        """
        Get ensembler classes to combine multiple predictions
        Needs to be overwritten in subclasses!
        """
        _lookup = {
            2: {
                "boxes": BoxEnsemblerSelective2D,
                "seg": SegmentationEnsembler,
            },
            3: {
                "boxes": BoxEnsemblerSelective,
                "seg": SegmentationEnsembler,
            },
        }
        return _lookup[dim][key]

    @classmethod
    def get_predictor(
        cls,
        plan: Dict,
        models: Sequence[LightningBaseModule],
        num_tta_transforms: int = None,
        do_seg: bool = False,
        **kwargs,
    ) -> Predictor:
        # process plan
        crop_size = plan["patch_size"]
        batch_size = plan["batch_size"]
        inferene_plan = plan.get("inference_plan", {})
        logger.info(f"Found inference plan: {inferene_plan} for prediction")
        if num_tta_transforms is None:
            num_tta_transforms = 8 if plan["network_dim"] == 3 else 4

        # setup
        tta_transforms, tta_inverse_transforms = get_tta_transforms(
            num_tta_transforms,
            seg=do_seg,
        )
        logger.info(
            f"Using {len(tta_transforms)} tta transformations for prediction (one dummy trafo)."
        )

        ensembler = {
            "boxes": partial(
                cls.get_ensembler_cls(key="boxes", dim=plan["network_dim"]).from_case,
                parameters=inferene_plan,
            )
        }
        if do_seg:
            ensembler["seg"] = partial(
                cls.get_ensembler_cls(key="seg", dim=plan["network_dim"]).from_case,
            )

        predictor = Predictor(
            ensembler=ensembler,
            models=models,
            crop_size=crop_size,
            tta_transforms=tta_transforms,
            tta_inverse_transforms=tta_inverse_transforms,
            batch_size=batch_size,
            **kwargs,
        )
        if plan["network_dim"] == 2:
            predictor.pre_transform = Inference2D(["data"])
        return predictor

    def sweep(
        self,
        cfg: dict,
        save_dir: os.PathLike,
        train_data_dir: os.PathLike,
        case_ids: Sequence[str],
        run_prediction: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Sweep detection parameters to find the best predictions

        Args:
            cfg: config used for training
            save_dir: save dir used for training
            train_data_dir: directory where preprocessed training/validation
                data is located
            case_ids: case identifies to prepare and predict
            run_prediction: predict cases
            **kwargs: keyword arguments passed to predict function

        Returns:
            Dict: inference plan
                e.g. (exact params depend on ensembler class usef for prediction)
                `iou_thresh` (float): best IoU threshold
                `score_thresh (float)`: best score threshold
                `no_overlap` (bool): enable/disable class independent NMS (ciNMS)
        """
        logger.info(f"Running parameter sweep on {case_ids}")

        train_data_dir = Path(train_data_dir)
        preprocessed_dir = train_data_dir.parent
        processed_eval_labels = preprocessed_dir / "labelsTr"

        _save_dir = save_dir / "sweep"
        _save_dir.mkdir(parents=True, exist_ok=True)

        prediction_dir = save_dir / "sweep_predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)

        if run_prediction:
            logger.info("Predict cases with default settings...")
            predict_dir(
                source_dir=train_data_dir,
                target_dir=prediction_dir,
                cfg=cfg,
                plan=self.plan,
                source_models=save_dir,
                num_models=1,
                num_tta_transforms=None,
                case_ids=case_ids,
                save_state=True,
                model_fn=get_loader_fn(mode=self.trainer_cfg.get("sweep_ckpt", "last")),
                **kwargs,
            )

        logger.info("Start parameter sweep...")
        ensembler_cls = self.get_ensembler_cls(
            key="boxes", dim=self.plan["network_dim"]
        )
        sweeper = BoxSweeper(
            classes=[item for _, item in cfg["data"]["labels"].items()],
            pred_dir=prediction_dir,
            gt_dir=processed_eval_labels,
            target_metric=self.eval_score_key,
            ensembler_cls=ensembler_cls,
            save_dir=_save_dir,
        )
        inference_plan = sweeper.run_postprocessing_sweep()
        return inference_plan
