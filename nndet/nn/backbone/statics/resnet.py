from typing import List, Optional, Type, Union

import torch

try:
    from monai.networks.nets.resnet import ResNet as BaseResNet
    from monai.networks.nets.resnet import ResNetBlock, ResNetBottleneck
except ImportError:
    BaseResNet = None
    ResNetBlock = None
    ResNetBottleneck = None

from nndet.nn.backbone.abstract import AbstractBackbone
from nndet.utils.format import to_nd_tuple
from nndet.utils.typing import CONVGEN, ND_INT, ND_TUPLE_INT

if BaseResNet is not None:

    class ResNet(BaseResNet, AbstractBackbone):
        """
        Please refer to the MONAI documentation for an overview of all args.

        Overview::

            Strides
            P0    - None
            P1-5  - stride 2

        Notes:
            - 1x1 downsample in residual
            - max stride: 32
            - always uses isotropic kernels
            - feed_forward should always be set to `False`
            - can only be used with normal FPN and UpFPN (no UFPN!)
        """

        def __init__(
            self,
            block,
            block_inplanes: List[int],
            widen_factor: float = 1.0,
            conv1_t_stride: ND_INT = 1,
            dim: int = 3,
            **kwargs,
        ):
            self.planes = [int(x * widen_factor) for x in block_inplanes]
            self.stride1 = conv1_t_stride
            self.dim = dim
            self.expansion = block.expansion
            super().__init__(
                block=block,
                block_inplanes=block_inplanes,
                widen_factor=widen_factor,
                conv1_t_stride=conv1_t_stride,
                spatial_dims=dim,
                **kwargs,
            )

        @classmethod
        def _from_config_plan(
            cls,
            backbone_cfg: dict,
            plan_arch: dict,
            dim: int,
            block: Type[Union[ResNetBlock, ResNetBottleneck]],
            layers: List[int],
        ):
            """
            Instantiate Backbone from given configs

            Args
                backbone_cfg: backbone configuration
                plan_arch: arguments provided plan
            """
            planes = [64, 128, 256, 512]
            return cls(
                block=block,
                layers=layers,
                block_inplanes=planes,
                dim=dim,
                n_input_channels=plan_arch["in_channels"],
                conv1_t_size=3,
                conv1_t_stride=2,
                no_max_pool=False,
                shortcut_type="B",
                feed_forward=False,
            )

        def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
            """
            Foward batch through all levels

            Args:
                batch: input batch [N, C, dims]

            Returns:
                List[torch.Tensor]: output features from each level
                    (ordered from highest to lowest resolution). P0 is None!
            """
            x = self.conv1(x)
            x = self.bn1(x)
            p1 = self.relu(x)

            if not self.no_max_pool:
                x = self.maxpool(p1)
                p2 = self.layer1(x)
            else:
                p2 = self.layer1(p1)

            p3 = self.layer2(p2)
            p4 = self.layer3(p3)
            p5 = self.layer4(p4)
            return [None, p1, p2, p3, p4, p5]

        def get_channels(self) -> List[Optional[int]]:
            """
            Compute number of channels for each returned feature map
            inside the forward pass

            Returns
                List[int]: list with number of channels corresponding to
                    returned feature maps
            """
            out_planes = [p * self.expansion for p in self.planes]
            channels = [None, self.planes[0], *out_planes]
            assert len(channels) == 6
            return channels

        def get_relative_strides(self) -> List[Optional[ND_TUPLE_INT]]:
            """
            Retrieve relative strides of the backbone feature maps.
            Starting with the highest resolution feature map to the lowest
            resolution feature map. Usually the first feature map will have stride
            1.

            Returns
                List[Tuple[int]]: defines the absolute stride for each output
                    feature map with respect to input size
            """
            rel_strides = [None]
            rel_strides.append(to_nd_tuple(self.stride1, self.dim))

            if self.no_max_pool:
                rel_strides.append(to_nd_tuple(1, self.dim))
            else:
                rel_strides.append(to_nd_tuple(2, self.dim))

            for _ in range(3):
                rel_strides.append(to_nd_tuple(2, self.dim))
            assert len(rel_strides) == 6
            return rel_strides

    class ResNet10(ResNet):
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
                backbone_cfg: ignored
                plan_arch:

                    ``"in_channels"`` List[ND_INT]
                        Number of input channels, usually equal to number of
                        modalities.

            """
            return cls._from_config_plan(
                backbone_cfg=backbone_cfg,
                plan_arch=plan_arch,
                dim=conv.dim,
                block=ResNetBlock,
                layers=[1, 1, 1, 1],
            )

    class ResNet18(ResNet):
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
                backbone_cfg: ignored
                plan_arch:

                    ``"in_channels"`` List[ND_INT]
                        Number of input channels, usually equal to number of
                        modalities.

            """
            return cls._from_config_plan(
                backbone_cfg=backbone_cfg,
                plan_arch=plan_arch,
                dim=conv.dim,
                block=ResNetBlock,
                layers=[2, 2, 2, 2],
            )

    class ResNet34(ResNet):
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
                backbone_cfg: ignored
                plan_arch:

                    ``"in_channels"`` List[ND_INT]
                        Number of input channels, usually equal to number of
                        modalities.

            """
            return cls._from_config_plan(
                backbone_cfg=backbone_cfg,
                plan_arch=plan_arch,
                dim=conv.dim,
                block=ResNetBottleneck,
                layers=[2, 2, 2, 2],
            )

    class ResNet50(ResNet):
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
                backbone_cfg: ignored
                plan_arch:

                    ``"in_channels"`` List[ND_INT]
                        Number of input channels, usually equal to number of
                        modalities.

            """
            return cls._from_config_plan(
                backbone_cfg=backbone_cfg,
                plan_arch=plan_arch,
                dim=conv.dim,
                block=ResNetBottleneck,
                layers=[3, 4, 6, 3],
            )

else:
    ResNet = None
    ResNet10 = None
    ResNet18 = None
    ResNet34 = None
    ResNet50 = None
