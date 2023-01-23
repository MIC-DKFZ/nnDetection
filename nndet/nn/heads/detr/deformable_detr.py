from typing import Dict, List, Optional, Tuple

import torch

from nndet.nn.heads.detr.base import DETRHead


class DeformableDETRHead(DETRHead):
    def forward(
        self,
        out_sequence: torch.Tensor,
        reference: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[List[Dict[str, torch.Tensor]]]]:
        """
        Predict bounding boxes and classes using two MLPs
        Args:
            out_sequence:
            init_reference:
            reference:
        Returns:
            Dict containing "pred_logits" and "pred_boxes"
        """
        # Calculate output coordinates and classes.
        class_logit_list = []
        box_logits_list = []
        for lvl in range(out_sequence.shape[0]):
            # TODO references passed must be all be stored in reference
            reference = reference[lvl]
            reference = self.regressor.apply_inverse_non_lin(reference)
            outputs_class = self.classifier[lvl](out_sequence[lvl])
            tmp = self.regressor[lvl](out_sequence[lvl])
            if reference.shape[-1] == 6:
                tmp += reference
            else:
                assert reference.shape[-1] == 3
                tmp[..., :3] += reference
            outputs_coord = tmp.sigmoid()
            class_logit_list.append(outputs_class)
            box_logits_list.append(outputs_coord)
        class_logits = torch.stack(class_logit_list)
        # tensor shape: [num_decoder_layers, bs, num_query, num_classes]
        box_logits = torch.stack(box_logits_list)
        # tensor shape: [num_decoder_layers, bs, num_query, 6]
        box_logits = box_logits[..., [0, 1, 3, 4, 2, 5]]
        preds = {"pred_cls_logits": class_logits[-1], "pred_box_coords": box_logits[-1]}
        if self.aux_loss:
            aux = [{"pred_cls_logits": a, "pred_box_coords": b} for a, b in zip(class_logits[:-1], box_logits[:-1])]
            preds["aux_outputs"] = aux

        return preds
