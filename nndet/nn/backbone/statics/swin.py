from typing import List, Optional, Sequence

import torch.nn as nn

from nndet.nn.backbone.abstract import AbstractBackbone
from nndet.utils.format import to_nd_tuple
from nndet.utils.typing import CONVGEN, ND_INT, ND_TUPLE_INT

try:
    from monai.networks.nets.swin_unetr import SwinTransformer as BaseSwinTransformer
except ImportError:
    BaseSwinTransformer = None


if BaseSwinTransformer is not None:

    class SwinTransformer(BaseSwinTransformer, AbstractBackbone):
        def __init__(
            self,
            dim: int,
            in_channels: int,
            layers: Sequence[int],
            num_heads: Sequence[int],
            embed_dim: int,
            patch_size: ND_INT,
            window_size: ND_INT,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            drop_path_rate: float = 0.1,
            patch_norm: bool = False,
            use_checkpoint: bool = False,
            create_stride_one: bool = True,
            **kwargs,
        ) -> None:
            """
            Swin Transformer

            Args:
                dim: number of spatial dimensions
                in_channels: number of input channels
                layers: depth of each level
                num_heads: number of attention heads per level
                embed_dim: patch embedding
                patch_size: patch size of embedding
                window_size: window size
                mlp_ratio: Ratio between mlp and embed dim. Defaults to 4.0.
                qkv_bias: Add bias to query, key, value. Defaults to True.
                drop_rate: Dropout rate. Defaults to 0.0.
                attn_drop_rate: Attention dropout rate. Defaults to 0.0.
                drop_path_rate: Stachastic Depth. Defaults to 0.0.
                patch_norm: Normalisation layer after patch embedding.
                use_checkpoint: Use gradient checkpointing for mem reduction.
                    Defaults to False.
                create_stride_one: Upsample first feature map to create stride
                    1 output. Defaults to True. Can be used with `UFPN` in this
                    case. Otherwise only `FPN` and `UpFPN` support can be used.
            """
            self.dim = dim
            super().__init__(
                in_chans=in_channels,
                depths=layers,
                spatial_dims=dim,
                patch_size=to_nd_tuple(patch_size, dim),
                embed_dim=embed_dim,
                window_size=to_nd_tuple(window_size, dim),
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                patch_norm=patch_norm,
                use_checkpoint=use_checkpoint,
                norm_layer=nn.LayerNorm,
                **kwargs,
            )
            if create_stride_one:
                self.stride_one = nn.Sequential(
                    nn.ConvTranspose3d(
                        in_channels=self.embed_dim,
                        out_channels=self.embed_dim // 2,
                        kernel_size=2,
                        stride=2,
                    ),
                )
            else:
                self.stride_one = None

        def forward(self, x, normalize=False):
            """
            Forward input through network

            Args
                x: input tensor

            Returns
                list: list with feature maps from multiple resolutions
                    Sorted from P0 (highest res) to PX (lowest res)
            """
            outs = super().forward(x, normalize)
            if self.stride_one is not None:
                outs = [self.stride_one(outs[0])] + outs
            else:
                outs = [None] + outs
            return outs

        def get_channels(self) -> List[Optional[int]]:
            """
            Compute number of channels for each returned feature map
            inside the forward pass

            Returns
                List[int]: list with number of channels corresponding to
                    returned feature maps. Undefined levels will be `None`.
            """
            channels = [self.embed_dim * (2**i) for i in range(5)]
            if self.stride_one is not None:
                channels = [self.embed_dim // 2] + channels
            else:
                channels = [None] + channels
            return channels

        def get_relative_strides(self) -> List[Optional[ND_TUPLE_INT]]:
            """
            Retrieve relative strides of the backbone feature maps.
            Starting with the highest resolution feature map to the lowest
            resolution feature map. Usually the first feature map will have stride
            1.

            Returns
                List[Tuple[int]]: defines the absolute stride for each output
                    feature map with respect to input size. Undefined levels
                    will be `None`.
            """
            strides = [to_nd_tuple(2, self.dim) for _ in range(5)]
            if self.stride_one is not None:
                strides = [to_nd_tuple(1, self.dim)] + strides
            else:
                strides = [None] + strides
            return strides

    class USwinTM(SwinTransformer):
        """
        Model inspired by SwinT network with some changed params
        "Swin Transformer: Hierarchical Vision Transformer
        using Shifted Windows"
        "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain
        Tumors in MRI Images"

        Changes:
            - drop_path_rate = 0.0 (default) from 0.2
            - patch_size = 2 (default) from 4
            - embed_dim = 48 (default) from 96
            - layers = [2, 2, 2, 2] (default) from [2, 2, 6, 2]
            - patch_norm = False (default) from True
            - create_stride_one = True (default)
        """

        @classmethod
        def from_config_plan(
            cls,
            conv: CONVGEN,
            backbone_cfg: dict,
            plan_arch: dict,
        ):
            """
            Instantiate Backbone from given configs

            Args
                conv: ignored
                backbone_cfg: provide backbone config

                    ``'create_stride_one'`` bool
                        create stride one output. Needed for `UFPN`.
                        Default `True`.

                    ``'patch_size'`` int
                        patch size of embedding. Default: `2`.

                    ``'layers'`` List[int]
                        depth of each level. Default `[2, 2, 2, 2]`

                    ``'num_heads'`` List[int]
                        number of attention heads per level.
                        Default `[3, 6, 12, 24]`

                    ``'embed_dim'`` int
                        dimensionality of embedding. Default `48`.

                    ``'window_size'`` int
                        window size. Default 7

                    ``'drop_path_rate'`` float
                        stochastic depth. Default `0.0`

                    ``'patch_norm'`` bool
                        Norm layer of patch embedding. Default `False`

                    ``'use_checkpoint'`` bool
                        Gradient checkpointing

                    ``'create_stride_one'`` bool
                        Add stride 1 output. Default `True`

                plan_arch:

                    ``"in_channels"`` List[ND_INT]
                        Number of input channels, usually equal to number of
                        modalities.

            """
            return cls(
                dim=conv.dim,
                in_channels=plan_arch["in_channels"],
                patch_size=backbone_cfg.get("patch_size", 2),
                layers=tuple(backbone_cfg.get("layers", (2, 2, 2, 2))),
                num_heads=tuple(backbone_cfg.get("num_heads", (3, 6, 12, 24))),
                embed_dim=backbone_cfg.get("embed_dim", 48),
                window_size=backbone_cfg.get("window_size", 7),
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=backbone_cfg.get("drop_path_rate", 0.0),
                patch_norm=backbone_cfg.get("patch_norm", False),
                use_checkpoint=backbone_cfg.get("use_checkpoint", True),
                create_stride_one=backbone_cfg.get("create_stride_one", True),
            )

else:
    SwinTransformer = None
    USwinTM = None
