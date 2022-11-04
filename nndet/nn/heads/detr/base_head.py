from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nndet.core.boxes import (
    box_cxcywhczd_to_xyxyzz,
    box_edges_pixel_to_center_normalized,
)
from nndet.losses import GIoULoss
from nndet.nn.heads.detr.matcher import SimpleHungarianMatcher
from nndet.utils.detr_misc import accuracy


class BaseDETRHead(nn.Module):
    """
    The Basic DETR Head
    All heads for DETR need to be derived from this, implements the basic loss computation and target preprocessing.
    """

    classifier_cls = ...
    regressor_cls = ...
    matcher_cls = ...
    matcher_config_keys = []
    config_keys = []

    def __init__(
        self,
        classifier: Union[nn.Module, nn.ModuleList],
        regressor: Union[nn.Module, nn.ModuleList],
        matcher: SimpleHungarianMatcher,
        weight_dict: Dict,
        losses: List[str],
        num_classes: int,
        aux_loss: bool = True,
    ):
        """
        Create the base head.

        Args:
            classifier: Classifier Module or Module List
            regressor: Regressor Module or Module List
            matcher: Matcher Module
            weight_dict: Weight Dict for loss computation
            losses: List of losses to compute from [boxes, labels, cardinality]
            num_classes: number of classes in the dataset
            aux_loss: Toggle whether output of each decoder layer should be used as auxiliary loss
        """
        super().__init__()
        # Modules
        self.classifier = classifier
        self.regressor = regressor
        self.matcher = matcher
        self.weight_dict = weight_dict
        # Parameters
        self.num_classes = num_classes
        # Losses
        self.losses = losses
        self.giou = GIoULoss(reduction="sum", eps=1e-8)
        self.l1 = torch.nn.L1Loss(reduction="none")
        # Toggle aux loss
        self.aux_loss = aux_loss

    def loss_labels(
        self,
        outputs: Dict[str, Tensor],
        targets: List[Dict[str, Tensor]],
        indices: List[Tuple[Tensor, Tensor]],
        num_boxes: int,
        log: bool = True,
    ) -> Dict[str, Tensor]:
        """
        Calculate the classification loss using self.class_loss which has to be set in a child class of this head
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        # indices of length R for the total number of boxes in the batch
        idx = _get_src_permutation_idx(indices)

        # Target Classes matched by the model
        target_classes_o = torch.cat(
            [
                t["labels"][J]
                if t["labels"].shape[0] > 0
                else torch.as_tensor(
                    [self.num_classes], dtype=torch.int64, device=t["labels"].device
                )
                for t, (_, J) in zip(targets, indices)
            ]
        )

        # Fill non-matched slots with background class as ground truth
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        # Calculate the loss and write to dict
        loss_ce = self.class_loss(src_logits.permute(0, 2, 1), target_classes)
        losses = {"loss_ce": loss_ce}
        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(
        self,
        outputs: Dict[str, Tensor],
        targets: List[Dict[str, Tensor]],
        indices: List[Tuple[Tensor, Tensor]],
        num_boxes: int,
    ) -> Dict[str, Tensor]:
        """
        Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device
        )
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(
        self,
        outputs: Dict[str, Tensor],
        targets: List[Dict[str, Tensor]],
        indices: List[Tuple[Tensor, Tensor]],
        num_boxes: int,
    ) -> Dict[str, Tensor]:
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 6]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = _get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]

        # idx = (batch_idx, source_idx)
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = self.l1(src_boxes, target_boxes)
        losses = {
            "loss_bbox": loss_bbox.sum() / num_boxes,
            "loss_giou": (
                self.giou(
                    box_cxcywhczd_to_xyxyzz(src_boxes),
                    box_cxcywhczd_to_xyxyzz(target_boxes),
                )
                / num_boxes
            ),
        }
        return losses

    def get_loss(
        self,
        loss: str,
        outputs: Dict[str, Tensor],
        targets: List[Dict[str, Tensor]],
        indices: List[Tuple[Tensor, Tensor]],
        num_boxes: int,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Given the string loss, return the corresponding loss
        """
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            # 'masks': self.loss_masks
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    @staticmethod
    def prepare_targets(
        targets: Dict[str, List[Tensor]], im_shape: Tuple
    ) -> List[Dict[str, Tensor]]:
        """
        Turn target Dict from nndetection into a list needed for DETR loss computation functions and converts boxes from
        edge format in pixels to center and width format normalized
        Args
            targets: Dict containing targets:
                ``'target_boxes'`` List[Tensor]
                    list contain batch size tensors with target boxes

                ``'target_classes'`` List[Tensor]
                    list containing batch size tensors with target classes

            im_shape: Tuple(3) giving the image shapes to compute the normalized coordinates
        Returns

        """
        target_boxes: List[Tensor] = targets["target_boxes"]
        target_classes: List[Tensor] = targets["target_classes"]
        target_list = []
        for i, box in enumerate(target_boxes):
            temp_dict = {"labels": target_classes[i].long()}
            if box.shape[1] > 0:
                temp_dict["boxes"] = box_edges_pixel_to_center_normalized(box, im_shape)
            else:
                temp_dict["boxes"] = []
            target_list.append(temp_dict)
        return target_list

    def compute_loss(
        self,
        outputs: Dict[str, Tensor],
        target_dict: Dict[str, List[Tensor]],
        im_shape: Tuple,
    ) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """
        Main function that computes all losses and preprocesses the ground truths

        Args:
            outputs: Dict containing "pred_boxes" and "pred_logits"
            target_dict: Dict containing "target_boxes" and "target_classes" (see prepare_targets)
            im_shape: Tuple of the original patch shape for box conversions
        Returns
            losses: Dict containing basic losses and "aux_outputs" if aux_loss
            targets: List of targets
        """
        targets = self.prepare_targets(target_dict, im_shape)
        losses = self.match_and_get_all_losses(outputs, targets)
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        # Count the number of layers to normalize the loss
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                l_dict = self.match_and_get_all_losses(aux_outputs, targets)
                l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                losses.update(l_dict)
        return losses, targets

    def match_and_get_all_losses(
        self,
        outputs: Dict,
        targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """
        Match the outputs with the targets and then compute all losses
        """
        (
            num_boxes,
            mask,
            masked_indices,
            full_indices,
            masked_outputs,
            masked_targets,
        ) = self.matcher(outputs, targets)

        # Box Loss computation on the masked (=non-empty) patches
        # If there is a box in any patch -> match it and calculate box loss on masked outputs
        losses = self.get_all_losses(
            num_boxes,
            masked_indices,
            full_indices,
            masked_outputs,
            outputs,
            masked_targets,
            targets,
        )
        return losses

    def get_all_losses(
        self,
        num_boxes: int,
        masked_indices: List[Tuple[Tensor, Tensor]],
        indices: List[Tuple[Tensor, Tensor]],
        masked_outputs: Dict[str, Tensor],
        outputs: Dict[str, Union[Tensor, List, Dict]],
        masked_targets: List[Dict[str, Tensor]],
        targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """
        Calculates Class and box losses
        Args:
            num_boxes: number of ground truth boxes in the batch
            masked_indices: matched indices with only the non-empty images
            indices: matched indices including empty images
            masked_outputs: outputs masked
            outputs: all outputs
            masked_targets: targets masked
            targets: all targets
        Returns:
            Losses
        """
        losses = {}
        if num_boxes > 0 and "boxes" in self.losses:
            # Compute the box loss using the masked outputs, targets and indices
            box_loss = self.get_loss(
                "boxes",
                masked_outputs,
                masked_targets,
                masked_indices,
                num_boxes,
            )
            losses.update(box_loss)

        # Calculate the non-box losses
        for loss in self.losses:
            kwargs = {}
            if loss != "boxes":
                if "labels" in loss:
                    kwargs = {"log": False}
                losses.update(
                    self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs)
                )
        return losses

    @torch.jit.unused
    def _set_aux_loss(
        self, outputs_class: Tensor, outputs_coord: Tensor
    ) -> List[Dict[str, Tensor]]:
        """
        this is a workaround to make torchscript happy, as torchscript
        doesn't support dictionary with non-homogeneous values, such
        as a dict having both a Tensor and a list.
        """
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def postprocess_for_inference(
        self,
        images: Tensor,
        pred_detection: Dict[str, Tensor],
    ):
        """
        Function to post-process predictions (calculate logits and convert to pixel format). Needs to be implemented in
        specific heads.
        """
        raise NotImplementedError


def _get_src_permutation_idx(
    indices: List[Tuple[Tensor, Tensor]]
) -> Tuple[Tensor, Tensor]:
    """
    Permute predictions following indices
    """
    batch_idx = torch.cat(
        [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
    )
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx
