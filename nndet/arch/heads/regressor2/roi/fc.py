import torch


class TwoLayerMLP:
    def __init__(self) -> None:
        pass
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        

        Args:
            features: input features of a batch of RoIs
                [N, C, dims]

        Returns:
            torch.Tensor: [description]
        """
        pass

    @abstractmethod
    def compute_loss(self, pred_logits: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """
        Compute classification loss (cross entropy loss)

        Args:
            pred_logits (Tensor): predicted logits
            targets (Tensor): classification targets

        Returns:
            Tensor: classification loss
        """
        raise NotImplementedError

    @abstractmethod
    def box_logits_to_probs(self, box_logits: Tensor) -> Tensor:
        """
        Convert bounding box logits to probabilities

        Args:
            box_logits (Tensor): bounding box logits [N, C], C=number of classes

        Returns:
            Tensor: probabilities
        """
        raise NotImplementedError