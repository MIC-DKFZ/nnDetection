# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
#
# Parts of this code are from DETR https://github.com/facebookresearch/detr
# SPDX-FileCopyrightText: 2020 Facebook, Inc
# SPDX-License-Identifier: Apache-2.0
#
# Parts of this code are from Conditional DETR https://github.com/Atten4Vis/ConditionalDETR
# SPDX-FileCopyrightText: 2020 SenseTime
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, Tuple

import torch

from nndet.nn.heads.detr.base import DETRHead


class ConditionalDETRHead(DETRHead):
    def forward(
        self,
        out_sequence: torch.Tensor,
        reference: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[List[Dict[str, torch.Tensor]]]]:
        """
        Predict bounding boxes and classes using the ClassifierFFN and
        RegressionFFN

        Args:
            out_sequence: output sequence of the transformer [D, B, R, C]
                where D=number of decoder layers, B=batch size,
                R=number of predictions, C=number of channels
            reference: reference output of the transformer
                Used as (unnormalized) reference point in conditional
                detr [B, R, dims], where B is the batch size, R=number
                of predictions, dims=number of spatial dimensions

        Returns:
            Dict[str, torch.Tensor]: predictions and auxiliary information

                ``"pred_cls_logits"`` torch.Tensor
                    predicted logits from ClassifierFFN [B, R, num_classes]
                    where B=batch size, R=number of predictions,
                    num_classes=number of classes

                ``"pred_box_coords"`` torch.Tensor
                    predicted normalized coords from RegressorFFN
                    [B, R, dims * 2] where B=batch size, R=number of
                    predictions, dims=number of spatial dimensions

                ``"aux_outputs"`` List[Dict[str, torch.Tensor]]
                    list with predictions from previous decoder layers
                    following the same format as `pred_cls_logits` and
                    `pred_box_coords`
        """

        # regressor
        reference_before_sigmoid = self.regressor.apply_inverse_non_lin(reference)

        outputs_coords = []
        # Also let intermediate level predict, but don't use it for the output
        for lvl in range(out_sequence.shape[0]):
            tmp = self.regressor(out_sequence[lvl])

            # select center point indices
            if tmp.shape[-1] == 4:
                inds = torch.tensor([0, 1], device=out_sequence.device)  # [cx, cy]
            else:
                inds = torch.tensor([0, 1, 4], device=out_sequence.device)  # [cx, cy, cz]

            tmp[..., inds] += reference_before_sigmoid
            outputs_coord = self.regressor.apply_non_lin(tmp)
            outputs_coords.append(outputs_coord)
        box_logits = torch.stack(outputs_coords)

        # classifier
        class_logits = self.classifier(out_sequence)

        preds = {"pred_cls_logits": class_logits[-1], "pred_box_coords": box_logits[-1]}
        if self.aux_loss:
            # put remaining outputs into aux info
            aux = [{"pred_cls_logits": a, "pred_box_coords": b} for a, b in zip(class_logits[:-1], box_logits[:-1])]
            preds["aux_outputs"] = aux
        return preds
