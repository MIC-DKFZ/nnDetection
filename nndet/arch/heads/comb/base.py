from abc import abstractmethod
from typing import TypeVar, Dict, List, Tuple

import torch

from nndet.arch.heads.abstract import BaseHead


class AnchorHead(BaseHead):
    def forward(self,
                fmaps: List[torch.Tensor],
                ) -> Dict[str, torch.Tensor]:
        """
        Forward feature maps through head modules

        Args:
            fmaps: list of feature maps for head module
                [N, C, spatial_dims]

        Returns:
            Dict[str, torch.Tensor]: predictions
                `box_deltas`(Tensor): bounding box offsets
                    [Num_Anchors_Batch, (num_classes), dim * 2];
                    num classes is only present if anchors were regressed
                    for each class individually
                `box_logits`(Tensor): classification logits
                    [Num_Anchors_Batch, num_classes]
        """
        logits, offsets = [], []
        for level, p in enumerate(fmaps):
            if self.shared is not None:
                intermediate = self.shared(p, level=level)
            else:
                intermediate = p
            
            offsets.append(self.regressor(intermediate, level=level))
            logits.append(self.classifier(intermediate, level=level))

        sdim = fmaps[0].ndim - 2
        if self.regress_multi_class:
            # TODO multi class regression
            raise NotImplementedError
        else:
            box_deltas = torch.cat(offsets, dim=1).reshape(-1, sdim * 2)
        box_logits = torch.cat(logits, dim=1).flatten(0, -2)
        return {"box_deltas": box_deltas, "box_logits": box_logits}

    def postprocess_for_inference(self,
                                prediction: Dict[str, torch.Tensor],
                                anchors: List[torch.Tensor],
                                ) -> Dict[str, torch.Tensor]:
        """
        Postprocess predictions for inference e.g. ocnvert logits to probs

        Args:
            Dict[str, torch.Tensor]: predictions from this head
                `box_logits`: classification logits for each anchor [N]
                `box_deltas`: offsets for each anchor
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
            List[torch.Tensor]: anchors per image
        """
        postprocess_predictions = {
            "pred_boxes": self.coder.decode(prediction["box_deltas"], anchors),
            "pred_probs": self.classifier.box_logits_to_probs(prediction["box_logits"]),
        }
        return postprocess_predictions

    @abstractmethod
    def compute_loss(self,
                     prediction: Dict[str, torch.Tensor],
                     target_labels: List[torch.Tensor],
                     matched_gt_boxes: List[torch.Tensor],
                     anchors: List[torch.Tensor],
                     ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Compute regression and classification loss
        N anchors over all images; M anchors per image => sum(M) = N

        Args:
            prediction: detection predictions for loss computation
                `box_logits`: classification logits for each anchor
                    [N, num_classes]
                `box_deltas`: offsets for each anchor
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
            target_labels: target labels for each anchor (per image) [M]
            matched_gt_boxes: matched gt box for each anchor
                List[[M, dim *  2]]
            anchors: anchors per image List[[M, dim *  2]]

        Returns:
            Tensor: dict with losses (reg for regression loss, cls for
                classification loss)
            Tensor: sampled positive indices of anchors
                (after concatenation if sampled otherwise None)
            Tensor: sampled negative indices of anchors
            (after concatenation, if sampled otherwise None)
        """
        raise NotImplementedError


class RoIHead(BaseHead):
    def forward(self,
                fmaps: torch.Tensor,
                ) -> Dict[str, torch.Tensor]:
        """
        Forward feature maps through RoI head modules

        Args:
            fmaps: feature maps exracted from pooling operation for each
                proposal [num_proposals, C, spatial_dims]

        Returns:
            Dict[str, torch.Tensor]: predictions
                `box_deltas`(Tensor): bounding box offsets
                    [num_proposals, (num_classes), dim * 2];
                    num classes is only present if anchors were regressed
                    for each class individually
                `box_logits`(Tensor): classification logits
                    [num_proposals, num_classes]
        """
        if self.shared is not None:
            intermediate = self.shared(fmaps)
        else:
            intermediate = fmaps
        
        box_deltas = self.regressor(intermediate)
        box_logits = self.classifier(intermediate)

        # TODO: check if reshape is needed

        return {
            "box_deltas": box_deltas,
            "box_logits": box_logits,
            }

    def postprocess_for_inference(self,
                        prediction: Dict[str, torch.Tensor],
                        anchors: List[torch.Tensor],
                        ) -> Dict[str, torch.Tensor]:
        """
        Postprocess predictions for inference e.g. ocnvert logits to probs

        Args:
            Dict[str, torch.Tensor]: predictions from this head
                `box_logits`: classification logits for each anchor [N]
                `box_deltas`: offsets for each anchor
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
            List[torch.Tensor]: anchors per image
        """
        postprocess_predictions = {
            "pred_boxes": self.coder.decode(prediction["box_deltas"], anchors),
            "pred_probs": self.classifier.box_logits_to_probs(prediction["box_logits"]),
        }
        return postprocess_predictions

    @abstractmethod
    def compute_loss(self,
                     prediction: Dict[str, torch.Tensor],
                     target_labels: torch.Tensor,
                     matched_gt_boxes: torch.Tensor,
                     proposal_boxes: torch.Tensor,
                     ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Compute regression and classification loss

        Args:
            prediction: detection predictions for loss computation
                `box_logits`: classification logits for each proposal
                    [N, num_classes]
                `box_deltas`: offsets for each anchor
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
            target_labels: target labels for each proposal [N]
            matched_gt_boxes: matched gt box for each proposal
                [N, dim *  2]
            proposal_boxes: concatenated and extended proposals with batch index
                (batch_idx, x1, y1, x2, y2, (z1, z2))[N, 1 + dim * 2]

        Returns:
            Tensor: dict with losses (reg for regression loss, cls for
                classification loss)
        """
        raise NotImplementedError

AnchorHeadType = TypeVar('AnchorHeadType', bound=AnchorHead)
RoIHeadType = TypeVar('RoIHeadType', bound=RoIHead)
