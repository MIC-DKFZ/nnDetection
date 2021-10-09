from abc import ABC
from typing import Dict, List

from loguru import logger

from nndet.evaluator import AbstractEvaluator
from nndet.evaluator.det import BoxEvaluator
from nndet.evaluator.seg import SegmentationEvaluator
from nndet.io.transforms import (
    AbstractTransform,
    FindInstances,
    Instances2Boxes,
    Instances2Fg,
    Instances2Segmentation,
)
from nndet.utils.tensor import to_numpy


class ModeMixin(ABC):
    """
    This mixin module defines the operation modes of the network.
    It provides the transformation to prepare the ground truth and input
    for the networks and defines the evaluations to perforn.
    """

    evaluators: Dict = {}  # needs to be overwritten in subclass

    def get_pre_transforms(self, plan: dict) -> List[AbstractTransform]:
        """
        Perform a sequence of transformations to the intput before passing it
        to the network. These transforamtions need to support pytorch tensors
        and are executed on the GPU.

        Raises:
            NotImplementedError: needs to be overwritten in the subclasses

        Returns:
            List[AbstractTransform]: return a list of transformations

        Notes:
            make sure to call the super classes here!
        """
        return []

    def evaluation_init(self, plan: dict) -> Dict[str, AbstractEvaluator]:
        """
        Initialize evaluation. Needs to be called before
        :method:`evaluation_step` and :method:`evaluation_end`.

        Notes:
            make sure to call the super classes here!
        """
        return {}

    def evaluation_step(
        self,
        predictions: dict,
        targets: dict,
    ) -> None:
        """
        Evaluate a validation batch

        Args:
            predictions: dict with predictions.
                Exact keys depend on the module class
            targets: dict with ground truth.
                Exact keys depend on the module class.

        Notes:
            make sure to call the super classes here!
        """
        pass  # end parent calls

    def evaluation_end(self) -> Dict[str, float]:
        """
        Compute validation metrics of epoch

        ```
        General pipeline should look something like this:
        # collect other scores
        scores = super().evaluation_end()

        # compute own scores
        own_scores = ...

        # add own scores
        metric_scores.update(own_scores)
        ```

        Returns:
            Dict[str, float]: computed metrics

        Notes:
            make sure to call the super classes here!
        """
        return {}


class BoxMixin(ModeMixin):
    def get_pre_transforms(self, plan: dict) -> List[AbstractTransform]:
        """
        Search for unqiue instances -> Instances to Boxes

        Returns:
            List[AbstractTransform]: return a list of transformations

        Notes:
            make sure to call the super classes here!
        """
        trafos = super().get_pre_transforms(plan=plan)
        trafos.append(
            FindInstances(
                instance_key="target",
                save_key="present_instances",
            )
        )
        trafos.append(
            Instances2Boxes(
                instance_key="target",
                map_key="instance_mapping",
                box_key="boxes",
                class_key="classes",
                present_instances="present_instances",
            )
        )
        return trafos

    def evaluation_init(self, plan: dict) -> Dict[str, AbstractEvaluator]:
        """
        Initialize BoxEvaluator

        Notes:
            make sure to call the super classes here!
        """
        evaluators = super().evaluation_init(plan=plan)
        if "boxes" in evaluators:
            raise RuntimeError(
                "Found BoxEvaluator in evaluators, can not register a second one!"
            )

        _classes = [
            f"class{c}" for c in range(plan["architecture"]["classifier_classes"])
        ]
        evaluators["boxes"] = BoxEvaluator.create(
            classes=_classes,
            fast=True,
            save_dir=None,
        )
        return evaluators

    def evaluation_step(
        self,
        predictions: dict,
        targets: dict,
    ) -> None:
        """
        Evaluate a validation batch with metrics

        Args:
            predictions: dict with predictions.
                Exact keys depend on the module class
            targets: dict with ground truth.
                Exact keys depend on the module class.

        Notes:
            make sure to call the super classes here!
        """
        super().evaluation_step(predictions=predictions, targets=targets)

        pred_boxes = to_numpy(predictions["pred_boxes"])
        pred_classes = to_numpy(predictions["pred_labels"])
        pred_scores = to_numpy(predictions["pred_scores"])

        gt_boxes = to_numpy(targets["target_boxes"])
        gt_classes = to_numpy(targets["target_classes"])
        gt_ignore = None

        self.evaluators["boxes"].run_online_evaluation(
            pred_boxes=pred_boxes,
            pred_classes=pred_classes,
            pred_scores=pred_scores,
            gt_boxes=gt_boxes,
            gt_classes=gt_classes,
            gt_ignore=gt_ignore,
        )

    def evaluation_end(self) -> Dict[str, float]:
        """
        Compute validation metrics of epoch

        Notes:
            make sure to call the super classes here!
        """
        # collect other scores
        metric_scores = super().evaluation_end()

        # compute own scores
        box_scores, _ = self.evaluators["boxes"].finish_online_evaluation()
        self.evaluators["boxes"].reset()

        # add own scores
        metric_scores.update(box_scores)

        # [optional] log own scores
        logger.info(
            f"mAP@0.1:0.5:0.05: {metric_scores['mAP_IoU_0.10_0.50_0.05_MaxDet_100']:0.3f}  "
            f"AP@0.1: {metric_scores['AP_IoU_0.10_MaxDet_100']:0.3f}  "
            f"AP@0.5: {metric_scores['AP_IoU_0.50_MaxDet_100']:0.3f} "
            f"AR@0.1: {metric_scores['AR_IoU_0.10_MaxDet_100']:0.3f} "
            f"AR@0.5: {metric_scores['AR_IoU_0.50_MaxDet_100']:0.3f} "
        )
        return metric_scores


class SemanticMixin(ModeMixin):
    """
    This Mixin only works with BoxMixin!
    BoxMixin needs to be subclassed last e.g.
    `Module(.. SemanticMixin, BoxMixin, ..)`

    Args:
        OperationModeMixin ([type]): [description]
    """

    def get_pre_transforms(self, plan: dict) -> List[AbstractTransform]:
        """
        Search for unqiue instances -> Instances to Boxes

        Returns:
            List[AbstractTransform]: return a list of transformations

        Notes:
            make sure to call the super classes here!
        """
        trafos = super().get_pre_transforms(plan=plan)
        trafos.append(
            Instances2Segmentation(
                instance_key="target",
                map_key="instance_mapping",
                present_instances="present_instances",
                seg_key="target_seg",
            )
        )
        return trafos

    def evaluation_init(self, plan: dict) -> Dict[str, AbstractEvaluator]:
        """
        Initialize SegmentationEvaluator

        Notes:
            make sure to call the super classes here!
        """
        evaluators = super().evaluation_init(plan=plan)
        if "semantic" in evaluators:
            raise RuntimeError(
                "Found SegmentationEvaluator in evaluators, can not register a second one!"
            )

        evaluators["semantic"] = SegmentationEvaluator.create()
        return evaluators

    def evaluation_step(
        self,
        predictions: dict,
        targets: dict,
    ) -> None:
        """
        Evaluate a validation batch with metrics

        Args:
            predictions: dict with predictions.
                Exact keys depend on the module class
            targets: dict with ground truth.
                Exact keys depend on the module class.

        Notes:
            make sure to call the super classes here!
        """
        super().evaluation_step(predictions=predictions, targets=targets)

        pred_seg = to_numpy(predictions["pred_seg"])
        gt_seg = to_numpy(targets["target_seg"])

        self.evaluators["semantic"].run_online_evaluation(
            seg_probs=pred_seg,
            target=gt_seg,
        )

    def evaluation_end(self) -> Dict[str, float]:
        """
        Compute validation metrics of epoch

        Notes:
            make sure to call the super classes here!
        """
        # collect other scores
        metric_scores = super().evaluation_end()

        # compute own scores
        seg_scores, _ = self.evaluators["semantic"].finish_online_evaluation()
        self.evaluators["semantic"].reset()

        # add own scores
        metric_scores.update(seg_scores)

        # [optional] log own scores
        logger.info(f"Proxy FG Dice: {seg_scores['seg_dice']:0.3f}")

        return metric_scores


class SemanticFgMixin(ModeMixin):
    """
    Run segmentation evaluation in FG mode

    This Mixin only works with BoxMixin!
    BoxMixin needs to be subclassed last e.g.
    `Module(.. SemanticMixin, BoxMixin, ..)`
    """

    def get_pre_transforms(self, plan: dict) -> List[AbstractTransform]:
        """
        Search for unqiue instances -> Instances to Boxes

        Returns:
            List[AbstractTransform]: return a list of transformations

        Notes:
            make sure to call the super classes here!
        """
        trafos = super().get_pre_transforms(plan=plan)
        trafos.append(
            Instances2Fg(
                instance_key="target",
                seg_key="target_seg",
            )
        )
        return trafos

    def evaluation_init(self, plan: dict) -> Dict[str, AbstractEvaluator]:
        """
        Initialize SegmentationEvaluator

        Notes:
            make sure to call the super classes here!
        """
        evaluators = super().evaluation_init(plan=plan)
        if "semantic_fg" in evaluators:
            raise RuntimeError(
                "Found SegmentationEvaluator in evaluators, can not register a second one!"
            )

        evaluators["semantic_fg"] = SegmentationEvaluator.create(fg_mode=True)
        return evaluators

    def evaluation_step(
        self,
        predictions: dict,
        targets: dict,
    ) -> None:
        """
        Evaluate a validation batch with metrics

        Args:
            predictions: dict with predictions.
                Exact keys depend on the module class
            targets: dict with ground truth.
                Exact keys depend on the module class.

        Notes:
            make sure to call the super classes here!
        """
        super().evaluation_step(predictions=predictions, targets=targets)

        pred_seg = to_numpy(predictions["pred_seg"])
        gt_seg = to_numpy(targets["target_seg"])

        self.evaluators["semantic_fg"].run_online_evaluation(
            seg_probs=pred_seg,
            target=gt_seg,
        )

    def evaluation_end(self) -> Dict[str, float]:
        """
        Compute validation metrics of epoch

        Notes:
            make sure to call the super classes here!
        """
        # collect other scores
        metric_scores = super().evaluation_end()

        # compute own scores
        seg_scores, _ = self.evaluators["semantic_fg"].finish_online_evaluation()
        self.evaluators["semantic_fg"].reset()

        # add own scores
        metric_scores.update(seg_scores)

        # [optional] log own scores
        logger.info(f"Proxy FG Dice (fg_mode=True): {seg_scores['seg_dice']:0.3f}")

        return metric_scores


class InstanceMixin(ModeMixin):
    pass
