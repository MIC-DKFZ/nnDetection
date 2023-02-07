# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
#
# Parts of this code are from detrex licensed under
# SPDX-FileCopyrightText: 2022, The IDEA Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn


class BaseTransformerLayer(nn.Module):
    def __init__(
        self,
        attn: Union[nn.Module, List[nn.Module]],
        ffn: nn.Module,
        norm: nn.Module,
        operation_order: Tuple[str, ...],
    ):
        """
        The implementation of Base `TransformerLayer` used in Transformer.
        It can be built by directly passing the `Attentions`, `FFNs`, `Norms`
        module, which support more flexible cusomization combined with
        `LazyConfig` system. The `BaseTransformerLayer` also supports `prenorm`
        when specifying the `norm` as the first element of `operation_order`
        Args:
            attn: nn.Module or a list
                contains the attention module used in TransformerLayer.
            ffn: FFN module used in TransformerLayer.
            norm: Normalization layer used in TransformerLayer.
            operation_order: The execution order of operation in
                transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
                Support `prenorm` when specifying the first element as `norm`
                Default = None.
        """
        super(BaseTransformerLayer, self).__init__()
        assert set(operation_order).issubset({"self_attn", "norm", "cross_attn", "ffn"})

        # count attention nums
        num_attn = operation_order.count("self_attn") + operation_order.count("cross_attn")

        if isinstance(attn, nn.Module):
            attn = [copy.deepcopy(attn) for _ in range(num_attn)]
        else:
            assert len(attn) == num_attn, (
                f"The length of attn (nn.Module or List[nn.Module]) {num_attn}"
                f"is not consistent with the number of attention in "
                f"operation_order {operation_order}"
            )

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.pre_norm = operation_order[0] == "norm"
        self.attentions = nn.ModuleList()
        index = 0
        for operation_name in operation_order:
            if operation_name in ["self_attn", "cross_attn"]:
                self.attentions.append(attn[index])
                index += 1

        self.embed_dim = self.attentions[0].embed_dim

        # count ffn nums
        self.ffns = nn.ModuleList()
        num_ffns = operation_order.count("ffn")
        for _ in range(num_ffns):
            self.ffns.append(copy.deepcopy(ffn))

        # count norm nums
        self.norms = nn.ModuleList()
        num_norms = operation_order.count("norm")
        for _ in range(num_norms):
            self.norms.append(copy.deepcopy(norm))

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attn_masks: Optional[List[torch.Tensor]] = None,
        query_key_padding_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward function for `BaseTransformerLayer`.
        **kwargs contains the specific arguments of attentions.
        Args:
            query: Query embeddings with shape `(num_query, bs, embed_dim)` or
                `(bs, num_query, embed_dim)` which should be specified follows
                the attention module used in `BaseTransformerLayer`.
            key: Key embeddings used in `Attention`.
            value: Value embeddings with the same shape as `key`.
            query_pos: The position embedding for `query`.
            key_pos: The position embedding for `key`.
            attn_masks: A list of 2D ByteTensor used in calculation the
                corresponding attention. The length of `attn_masks` should be
                equal to the number of `attention` in `operation_order`.
            query_key_padding_mask: ByteTensor for `query`, with
                shape `(bs, num_query)`. Only used in `self_attn` layer.
            key_padding_mask: ByteTensor for `key`, with shape `(bs, num_key)`.
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(f"Use same attn_mask in all attentions in " f"{self.__class__.__name__} ")
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )

        for layer in self.operation_order:
            if layer == "self_attn":
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


class TransformerLayerSequence(nn.Module):
    def __init__(
        self,
        transformer_layers: Union[List[BaseTransformerLayer], BaseTransformerLayer],
        num_layers: int,
    ):
        """
        Base class for TransformerEncoder and TransformerDecoder, which will
        copy the passed `transformer_layers` module `num_layers` time or save
        the passed list of `transformer_layers` as parameters named
        ``self.layers`` which is the type of ``nn.ModuleList``.
        The users should inherit `TransformerLayerSequence` and implemente their
        own forward function.
        Args:
            transformer_layers: A list of BaseTransformerLayer. If it is
            obj:`BaseTransformerLayer`, it would be repeated `num_layers` times
            to a list[BaseTransformerLayer]
            num_layers: The number of `TransformerLayer`.
        """
        super(TransformerLayerSequence, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        if isinstance(transformer_layers, nn.Module):
            for _ in range(num_layers):
                self.layers.append(copy.deepcopy(transformer_layers))
        else:
            assert isinstance(transformer_layers, list) and len(transformer_layers) == num_layers

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_pos: torch.Tensor = None,
        key_pos: torch.Tensor = None,
        attn_masks: torch.Tensor = None,
        query_key_padding_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        """
        Forward function of `TransformerLayerSequence`. The users should inherit
        `TransformerLayerSequence` and implement their own forward function.
        """
        raise NotImplementedError()
