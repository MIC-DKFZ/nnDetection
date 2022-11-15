from typing import Dict, List, Optional, Tuple, Union

import torch

from nndet.core.boxes import box_center_normalized_to_edges_original
from nndet.nn.heads.classifier.ffn import FFNClassifier
from nndet.nn.heads.regressor.ffn import FFNRegressor


class DETRHead(torch.nn.Module):
    def __init__(
        self,
        classifier: FFNClassifier,
        regressor: FFNRegressor,
        matcher: None,  # TODO
        aux_loss: bool = True,
    ) -> None:
        super().__init__()
        self.classifier = classifier
        self.reggressor = regressor
        self.matcher = matcher

        self.aux_loss = aux_loss

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
                (not used in original DETR head)
                #TODO

        Returns:
            Dict[str, torch.Tensor]: predictions and auxiliary information

                ``"pred_cls_logits"`` torch.Tensor
                    predicted logits from ClassifierFFN [B, R, num_classes]
                    where B=batch size, R=number of predictions,
                    num_classes=number of classes

                ``"pred_box_logits"`` torch.Tensor
                    predicted normalized coords from RegressorFFN
                    [B, R, dims * 2] where B=batch size, R=number of
                    predictions, dims=number of spatial dimensions

                ``"aux_outputs"`` List[Dict[str, torch.Tensor]]
                    list with predictions from previous decoder layers
                    following the same format as `pred_cls_logits` and
                    `pred_box_logits`
        """
        box_logits = self.regressor(out_sequence)
        class_logits = self.classifier(out_sequence)

        preds = {"pred_cls_logits": class_logits[-1], "pred_box_logits": box_logits[-1]}
        # Predict for all decoder levels but only propagate last decoder output

        if self.aux_loss:
            # put remaining oututs into aux info
            aux = [{"pred_cls_logits": a, "pred_box_logits": b} for a, b in zip(class_logits[:-1], box_logits[:-1])]
        else:
            aux = None
        return preds, aux

    def compute_loss(
        self,
        prediction_logits: Dict[str, torch.Tensor],
        target_boxes_norm: List[torch.Tensor],
        target_labels: List[torch.Tensor],
        img_shape: Union[Tuple[int, int], Tuple[int, int, int]],
    ) -> Dict[str, torch.Tensor]:
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

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        target_dict: Dict[str, List[torch.Tensor]],
        img_shape: Union[Tuple[int, int], Tuple[int, int, int]],
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
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

    def postprocess_for_inference(
        self,
        images: torch.Tensor,
        pred_detection: Dict[str, torch.Tensor],
    ):
        """
        Postprocessing of the Predictions

        Args:
            images: input batch [N, sdims], N=batch size, sdims=spatial
                dimensions
            pred_detection: prediction from network

                ``'pred_boxes'``: torch.
                    predicted boxes by regressor in #TODO format []

                ``'pred_logits'``: torch.Tensor
                    predicted logits by classifier [] #TODO

        Returns:
                Dict: post processed predictions

                    ``'pred_boxes'``: List[torch.ensor]
                        predicted bounding boxes for each image List[[R, dim * 2]]

                    ``'pred_scores'``: List[torch.Tensor]
                        predicted probability for the class List[[R]]

                    ``'pred_labels'``: List[torch.Tensor]
                        predicted class List[[R]]
        """
        batch_pred_scores_fg = self.classifier.postprocess_logits(pred_detection["pred_logits"])
        batch_pred_scores, batch_pred_labels = batch_pred_scores_fg.max(-1)
        batch_pred_boxes = box_center_normalized_to_edges_original(pred_detection["pred_boxes"], images.shape[-3:])

        batch_size = batch_pred_scores.shape[0]
        assert batch_size == batch_pred_labels.shape[0]
        assert batch_size == batch_pred_boxes.shape[0]
        assert batch_pred_labels.shape[1] == batch_pred_boxes.shape[1]

        prediction = {"pred_boxes": [], "pred_scores": [], "pred_labels": []}
        for batch_idx in range(batch_size):
            prediction["pred_boxes"].append(batch_pred_boxes[batch_idx])
            prediction["pred_scores"].append(batch_pred_scores[batch_idx])
            prediction["pred_labels"].append(batch_pred_labels[batch_idx])
        return prediction
