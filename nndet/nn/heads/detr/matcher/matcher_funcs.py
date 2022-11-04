from torch import nn


class SimpleClassLossforMatcher(nn.Module):
    """
    Simple Class Loss used in original DETR Paper for the Matcher Class loss
    """

    def __init(self):
        super().__init__()

    def forward(self, out_prob, tgt_ids):
        return -out_prob[:, tgt_ids]


class FocalLossforMatcher(nn.Module):
    """
    Focal Class loss used by Conditional DETR for the matcher
    """

    def __init__(self, alpha=0.75, gamma=1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, out_prob, tgt_ids):
        neg_cost_class = (
            (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
        )
        pos_cost_class = (
            self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
        )
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        return cost_class
