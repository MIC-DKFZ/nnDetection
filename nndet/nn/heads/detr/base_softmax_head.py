from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nndet.core.boxes import box_center_normalized_to_edges_original
from nndet.losses.classification import CrossEntropyLoss
from nndet.nn.heads.detr import BaseDETRHead
from nndet.nn.heads.detr.matcher import SimpleHungarianMatcher
from nndet.nn.heads.detr.util import LinearClassifierCE
from nndet.utils.mlp import MLP


class BaseSoftmaxDETRHead(BaseDETRHead):
    classifier_cls = LinearClassifierCE
    regressor_cls = MLP
    matcher_cls = SimpleHungarianMatcher
    matcher_config_keys = []
    config_keys = []

    def __init__(
        self,
        classifier: nn.Module,
        regressor: nn.Module,
        matcher: nn.Module,
        weight_dict: Dict,
        losses: List[str],
        num_classes: int,
        eos_coef: float = 0.1,
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
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.class_loss = CrossEntropyLoss(weight=empty_weight, loss_weight=1, reduction="mean")

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

    def postprocess_for_inference(
        self,
        images: torch.Tensor,
        pred_detection: Dict[str, torch.Tensor],
    ):
        """
        Postprocessing of the Predictions
        Returns:
                Dict: post processed predictions
                    'pred_boxes': List[Tensor]: predicted bounding boxes for each
                        image List[[R, dim * 2]]
                    'pred_scores': List[Tensor]: predicted probability for
                        the class List[[R]]
                    'pred_labels': List[Tensor]: predicted class List[[R]]
                    'pred_seg': Tensor: predicted segmentation [N, C, dims]
        """ ""

        # Get the highest scoring class for each prediction
        pred_labels = pred_detection["pred_logits"].argmax(dim=2)
        # Get mask of non-background predictions and mask the predictions
        no_bg_mask = pred_labels < self.num_classes
        pred_labels_masked = [pred_labels[i][mask] for i, mask in enumerate(no_bg_mask)]

        # Get all corresponding boxes
        prediction = {
            "pred_boxes": [
                box_center_normalized_to_edges_original(pred_detection["pred_boxes"][i][mask], images.shape[-3:])
                for i, mask in enumerate(no_bg_mask)
            ]
        }

        # Mask the logits to the same size
        logits_mask = no_bg_mask.unsqueeze(-1).expand(pred_detection["pred_logits"].size())
        logits_masked = [pred_detection["pred_logits"][i][mask] for i, mask in enumerate(logits_mask)]

        # Obtain the logit score for the highest scoring class from the masked logits
        pred_scores = [F.softmax(logits_masked[i], dim=0)[pred_labels_masked[i]] for i in range(len(logits_masked))]
        prediction["pred_labels"] = pred_labels_masked
        prediction["pred_scores"] = pred_scores
        return prediction
