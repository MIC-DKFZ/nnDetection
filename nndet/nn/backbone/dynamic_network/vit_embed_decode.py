from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
from torch import nn

from nndet.nn.backbone.dynamic_network.helper import convert_dim_to_conv_op


class LayerNormNd(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        index_tuple = (None, slice(None)) + (None,) * (x.ndim - 2)
        x = self.weight[index_tuple] * x + self.bias[index_tuple]
        return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    Loosely inspired by https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L364

    """

    def __init__(
            self,
            patch_size: Tuple[int, ...] = (16, 16, 16),
            input_channels: int = 3,
            embed_dim: int = 768,
    ) -> None:
        """
        Args:
            patch_size (Tuple): patch size.
            padding (Tuple): padding size of the projection layer.
            input_channels (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = convert_dim_to_conv_op(len(patch_size))(
            input_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        returns shape (B, embed_dim, px, py, pz) where (px, py, pz) is patch_size.
        This output will need to be rearranged to whatever your transformer expects!
        """
        x = self.proj(x)
        return x


class PatchDecode(nn.Module):
    """
    Loosely inspired by SAM decoder
    https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py#L53
    """

    def __init__(
            self,
            patch_size,
            embed_dim: int,
            out_channels: int,
            norm=LayerNormNd,
            activation=nn.GELU,
    ):
        """
        patch size must be 2^x, so 2, 4, 8, 16, 32, etc. Otherwise we die
        """
        super().__init__()

        def _round_to_8(inp):
            return int(max(8, np.round((inp + 1e-6) / 8) * 8))

        num_stages = int(np.log(max(patch_size)) / np.log(2))
        strides = [[2 if (p / 2**n) % 2 == 0 else 1 for p in patch_size] for n in range(num_stages)][::-1]


        # decode channels independent of out_channels, so pretrained weights can
        # be loaded regardless of the downstream task's number of output channels
        dim_red = (embed_dim / (2 * 1)) ** (1/num_stages)

        channels = [embed_dim] + [_round_to_8(embed_dim / dim_red ** (x + 1)) for x in range(num_stages)]
        self.out_channels = deepcopy(channels)
        channels[-1] = out_channels

        stages = []
        for s in range(num_stages - 2): # num_stages - 2 because we do not want the final feature maps
            stages.append(nn.Sequential(
                nn.ConvTranspose3d(channels[s], channels[s+1], kernel_size=strides[s], stride=strides[s]),
                norm(channels[s+1]),
                activation()
            ))

        # we do not want the final 1x1 conv
        self.decode = nn.Sequential(*stages)

    def forward(self, x):
        """
        Expects input of shape (B, embed_dim, px, py, pz)! This will require you to reshape the output of your transformer!
        """
        out = [x]
        

        #only lowest two transformer resolution
        out.append(self.decode[0](out[-1]))
        return out