import torch

from nndet.core.boxes.criterions.base import ClassCriterion


class SimpleClassCriterionSoftmax(ClassCriterion):
    def __init__(self, loss_weight: float) -> None:
        """
        Comute simple class criterion with softmax logits

        Args:
            loss_weight: weighting for computed loss
        """
        super().__init__(loss_weight=loss_weight)
        self.logits_convert_fn = torch.nn.Softmax(dim=-1)

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute softmax based class criterion

        Args:
            pred_logits: predicted logits [B * R, C] where B=batch size,
            R=number of predictions, C=number of
                classes
            target_labels: target label for each bounding box [L] where
                L is the number of ground truth objects

        Returns:
            torch.Tensor: cost matrix [B * R, L], where B=batch size,
                R=number of predictions, L is the number of ground truth
                objects
        """
        pred_probs = self.logits_convert_fn(pred_logits)
        return self.loss_weight * -1 * pred_probs[:, target_labels]


class SimpleClassCriterionSigmoid(ClassCriterion):
    def __init__(self, loss_weight: float) -> None:
        """
        Comute simple class criterion with softmax logits

        Args:
            loss_weight: weighting for computed loss
        """
        super().__init__(loss_weight=loss_weight)
        self.logits_convert_fn = torch.nn.Sigmoid()

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute softmax based class criterion

        Args:
            pred_logits: predicted logits [B * R, C] where B=batch size,
            R=number of predictions, C=number of
                classes
            target_labels: target label for each bounding box [L] where
                L is the number of ground truth objects

        Returns:
            torch.Tensor: cost matrix [B * R, L], where B=batch size,
                R=number of predictions, L is the number of ground truth
                objects
        """
        pred_probs = self.logits_convert_fn(pred_logits)
        return self.loss_weight * -1 * pred_probs[:, (target_labels - 1)]
