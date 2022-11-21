import torch
from torch import Tensor

from nndet.nn.heads.detr import BaseFocalDETRHead, BaseSoftmaxDETRHead
from nndet.utils.detr_misc import inverse_sigmoid


class ConditionalDETRForward:
    def forward(self, out_sequence: Tensor, reference: Tensor):
        """
        Predict bounding boxes and classes using two MLPs
        Args:
            out_sequence: output sequence of the transformer
            reference: reference output of the transformer
        Returns:
            Dict containing "pred_logits" and "pred_boxes"
        """
        reference_before_sigmoid = inverse_sigmoid(reference)
        outputs_coords = []
        inds = torch.tensor([0, 1, 4], device=out_sequence.device)
        # Also let intermediate level predict, but don't use it for the output
        for lvl in range(out_sequence.shape[0]):
            tmp = self.regressor(out_sequence[lvl])
            tmp[..., inds] += reference_before_sigmoid
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
        boxes = torch.stack(outputs_coords)
        classes = self.classifier(out_sequence)
        out = {"pred_logits": classes[-1], "pred_boxes": boxes[-1]}
        # Predict for all decoder levels but only propagate last decoder output
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(classes, boxes)
        return out


class ConditionalDETRHead(
    ConditionalDETRForward,
    BaseFocalDETRHead,
):
    """
    Conditional DETR Head with Focal Loss
    """


class ConditionalDETRCEHead(
    ConditionalDETRForward,
    BaseSoftmaxDETRHead,
):
    """
    Conditional DETR Cross Entropy Head, different forward pass than the BaseSoftmax head because of the reference
    addition
    """
