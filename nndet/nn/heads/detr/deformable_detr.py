# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, Tuple

import torch

from nndet.nn.heads.detr.base import DETRHead


class DeformableDETRHead(DETRHead):
    def forward(
        self,
        out_sequence: torch.Tensor,
        refs_ccddcd_norm: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[List[Dict[str, torch.Tensor]]]]:
        """
        Predict bounding boxes and classes using two MLPs
        Args:
            out_sequence:
            references:
        Returns:
            Dict containing "pred_logits" and "pred_boxes"
        """
        # Calculate output coordinates and classes.
        class_logit_list = []
        box_logits_list = []

        # select center point indices
        if refs_ccddcd_norm.shape[-1] in [2, 4]:
            inds = torch.tensor([0, 1], device=out_sequence.device)  # [cx, cy]
        else:
            inds = torch.tensor([0, 1, 4], device=out_sequence.device)  # [cx, cy, cz]

        for lvl in range(out_sequence.shape[0]):
            outputs_class = self.classifier(out_sequence[lvl], lvl)
            ref_ccddcd_norm = refs_ccddcd_norm[lvl]
            ref_ccddcd_raw = self.regressor.apply_inverse_non_lin(ref_ccddcd_norm)
            box_coords_ccddcd_raw = self.regressor(out_sequence[lvl], lvl)

            if ref_ccddcd_norm.shape[-1] in [4, 6]:  # entire box
                box_coords_ccddcd_raw += ref_ccddcd_raw
            else:  # only center point ccc
                assert ref_ccddcd_norm.shape[-1] in [2, 3]
                box_coords_ccddcd_raw[..., inds] += ref_ccddcd_raw

            box_coords_ccddcd_norm = self.regressor.apply_non_lin(box_coords_ccddcd_raw)
            class_logit_list.append(outputs_class)
            box_logits_list.append(box_coords_ccddcd_norm)

        # [num_decoder_layers, bs, num_query, num_classes]
        class_logits = torch.stack(class_logit_list)
        # [num_decoder_layers, bs, num_query, 6]
        box_logits = torch.stack(box_logits_list)

        preds = {"pred_cls_logits": class_logits[-1], "pred_box_coords": box_logits[-1]}
        if self.aux_loss:
            aux = [{"pred_cls_logits": a, "pred_box_coords": b} for a, b in zip(class_logits[:-1], box_logits[:-1])]
            preds["aux_outputs"] = aux
        return preds
