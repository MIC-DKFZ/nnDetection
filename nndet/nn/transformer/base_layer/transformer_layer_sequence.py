import copy

import torch.nn as nn


class TransformerLayerSequence(nn.Module):
    """Base class for TransformerEncoder and TransformerDecoder, which will copy
    the passed `transformer_layers` module `num_layers` time or save the passed
    list of `transformer_layers` as parameters named ``self.layers``
    which is the type of ``nn.ModuleList``.
    The users should inherit `TransformerLayerSequence` and implemente their
    own forward function.
    Args:
        transformer_layers (list[BaseTransformerLayer] | BaseTransformerLayer): A list
            of BaseTransformerLayer. If it is obj:`BaseTransformerLayer`, it
            would be repeated `num_layers` times to a list[BaseTransformerLayer]
        num_layers (int): The number of `TransformerLayer`. Default: None.
    """

    def __init__(
        self,
        transformer_layers=None,
        num_layers=None,
    ):
        super(TransformerLayerSequence, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        if isinstance(transformer_layers, nn.Module):
            for _ in range(num_layers):
                self.layers.append(copy.deepcopy(transformer_layers))
        else:
            assert isinstance(transformer_layers, list) and len(transformer_layers) == num_layers

    def forward(self):
        """Forward function of `TransformerLayerSequence`. The users should inherit
        `TransformerLayerSequence` and implemente their own forward function.
        """
        raise NotImplementedError()
