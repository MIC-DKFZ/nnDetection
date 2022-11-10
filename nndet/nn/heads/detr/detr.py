from typing import Dict

import torch

from nndet.nn.heads.detr.base import BaseDETRHead


class DETRHead(BaseDETRHead):
    def forward(
        self,
        out_sequence: torch.Tensor,
        reference: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict bounding boxes and classes using heads

        Args:
            out_sequence: output sequence of the transformer
            reference: reference output of the transformer, not used in this head

        Returns:
            Dict: predictions from regressor and classifier

                ``'pred_logits'``: torch.Tensor
                    predicted logits by classifier #TODO

                ``'pred_boxes'``: torch.Tensor
                    predicted normalized boxes by regressor #TODO

                ``'aux_outputs'``: List[torch.Tensor]
                    # TODO
        """
        boxes = self.regressor(out_sequence).sigmoid()
        classes = self.classifier(out_sequence)

        out = {"pred_logits": classes[-1], "pred_boxes": boxes[-1]}
        # Predict for all decoder levels but only propagate last decoder output
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(classes, boxes)
        return out
