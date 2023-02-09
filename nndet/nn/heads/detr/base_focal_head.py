import math
from typing import Dict, List, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nndet.core.boxes.ops import box_center_normalized_to_edges_original
from nndet.losses.classification.focal import BFocalLoss
from nndet.nn.heads.detr import BaseDETRHead, _get_src_permutation_idx
from nndet.nn.heads.detr.matcher import FocalHungarianMatcher, SimpleHungarianMatcher
from nndet.nn.heads.detr.util import LinearClassifierFocalLoss
from nndet.utils.detr_misc import accuracy
from nndet.utils.mlp import MLP

# Don't use this


class BaseFocalDETRHead(BaseDETRHead):
    """
    Basic Head with Focal Loss
    """

    classifier_cls = LinearClassifierFocalLoss
    regressor_cls = MLP
    matcher_cls = FocalHungarianMatcher
    matcher_config_keys = ["alpha", "gamma"]
    config_keys = ["topk", "threshold", "alpha", "gamma"]

    def __init__(
        self,
        classifier: Union[nn.Module, nn.ModuleList],
        regressor: Union[nn.Module, nn.ModuleList],
        matcher: SimpleHungarianMatcher,
        weight_dict: Dict,
        losses: List[str],
        num_classes: int,
        alpha: float = 0.75,
        gamma: float = 1,
        topk: int = 5,
        threshold: float = 0.1,
        aux_loss: bool = True,
        **kwargs,
    ):
        super().__init__(
            classifier,
            regressor,
            matcher,
            weight_dict=weight_dict,
            losses=losses,
            num_classes=num_classes,
            aux_loss=aux_loss,
        )

        self.class_loss = BFocalLoss(gamma=gamma, alpha=alpha, loss_fp32=True, reduction="sum", loss_weight=1)
        self.topk = topk
        self.threshold = threshold
        # init prior_prob setting for focal loss
        prior_prob = 0.05
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        if self.classifier.__class__ == nn.ModuleList:
            for class_embed in self.classifier:
                class_embed.bias.data = torch.ones_like(class_embed.bias.data) * bias_value
            for bbox_embed_layer in self.regressor:
                nn.init.constant_(bbox_embed_layer.layers[-1].bias.data[3:], 0.0)
        else:
            self.classifier.bias.data = torch.ones_like(self.classifier.bias.data) * bias_value
            # Changed to Debug
            # nn.init.constant_(self.regressor.layers[-1].weight.data, 0)
            nn.init.constant_(self.regressor.layers[-1].bias.data, 0)

    def forward(self, out_sequence: Tensor, reference: Tensor):
        """
        Predict bounding boxes and classes using two MLPs
        Args:
            out_sequence: output sequence of the transformer
            reference: reference output of the transformer, not used in this head
        Returns:
            Dict containing "pred_logits" and "pred_boxes"
        """
        boxes = self.regressor(out_sequence).sigmoid()
        classes = self.classifier(out_sequence)
        out = {"pred_logits": classes[-1], "pred_boxes": boxes[-1]}
        # Predict for all decoder levels but only propagate last decoder output
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(classes, boxes)
        return out

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
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
                t["labels"][J] + 1
                if t["labels"].shape[0] > 0
                else torch.as_tensor([0], dtype=torch.int64, device=t["labels"].device)
                for t, (_, J) in zip(targets, indices)
            ]
        )

        # Fill non-matched slots with background class as ground truth
        target_classes = torch.zeros(
            src_logits.shape[:2],
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        # Calculate the loss and write to dict
        # focal_loss_unreduced = self.class_loss(src_logits.permute(0, 2, 1), 1 - target_classes)
        focal_loss = self.class_loss(src_logits.permute(0, 2, 1), target_classes)
        if num_boxes > 0:
            focal_loss /= num_boxes
        losses = {"loss_ce": focal_loss}
        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object", adapted for focal loss
        card_pred = (pred_logits.max(1).values >= self.threshold).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def postprocess_for_inference(
        self,
        images: torch.Tensor,
        pred_detection: Dict[str, torch.Tensor],
    ):
        """
        Post Processing of the predictions, uses topk and threshold to determine what should count as a non background
        prediction

        Returns:
                Dict: post processed predictions
                    'pred_boxes': List[Tensor]: predicted bounding boxes for each
                        image List[[R, dim * 2]]
                    'pred_scores': List[Tensor]: predicted probability for
                        the class List[[R]]
                    'pred_labels': List[Tensor]: predicted class List[[R]]
                    'pred_seg': Tensor: predicted segmentation [N, C, dims]
        """
        # There are no background predictions in here so different from other head
        # Get the highest scoring class for each prediction
        batch_pred_scores = pred_detection["pred_logits"].sigmoid()
        batch_pred_boxes = box_center_normalized_to_edges_original(pred_detection["pred_boxes"], images.shape[-3:])
        batch_size = images.shape[0]
        assert batch_size == batch_pred_scores.shape[0]
        assert batch_size == batch_pred_boxes.shape[0]
        assert batch_pred_scores.shape[1] == batch_pred_boxes.shape[1]

        # flatten query and class dimension
        batch_pred_scores_topk, topk_indices = torch.topk(
            batch_pred_scores.view(batch_pred_scores.shape[0], -1), self.topk, dim=1
        )
        # index div classes gives the to the index corresponding query
        topk_boxes = torch.div(topk_indices, batch_pred_scores.shape[2], rounding_mode="floor")
        # modulo gives the to the index corresponding class
        batch_pred_labels_topk = topk_indices % batch_pred_scores.shape[2]
        batch_pred_boxes_topk = torch.gather(batch_pred_boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 6))

        prediction = {"pred_boxes": [], "pred_scores": [], "pred_labels": []}
        for batch_idx in range(batch_size):
            prediction["pred_boxes"].append(batch_pred_boxes_topk[batch_idx])
            prediction["pred_scores"].append(batch_pred_scores_topk[batch_idx])
            prediction["pred_labels"].append(batch_pred_labels_topk[batch_idx])
        return prediction
