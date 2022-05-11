from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from nndet.nn.backbone.abstract import AbstractBackbone
from nndet.utils.typing import CONV_GENERATOR, ND_INT, ND_TUPLE_INT


def to_nd_tuple(x: Any, dim: int) -> Tuple:
    if not isinstance(x, Sequence):
        return tuple([x] * dim)
    else:
        assert len(x) == dim
        return tuple(x)


class BackboneLevel(torch.nn.Module):
    """
    Define a single level in the backbone.
    Compared to a `torch.nn.Module` this will return
    two tensor: the first one will be forwards to the deeper levels
    while the second tensor will be returned as an intermediate feature
    map from the backbone module.
    """

    @abstractmethod
    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Foward batch through level

        Args:
            batch: input batch [N, C, dims]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: First tensor will be forwarded
                to the next level. Second tensor will be returned from the
                backbone as an intermediate feature map
        """
        raise NotImplementedError


class WrapperBackboneLevel(BackboneLevel):
    def __init__(self, component: torch.nn.Module) -> None:
        """
        Provides a simple wrapper for any torch modules which
        return their output for further processing and as an
        intermediate feature.

        Args:
            component: torch module to wrap as level
        """
        super().__init__()
        self.component = component

    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Foward batch through level

        Args:
            batch: input batch [N, C, dims]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Both tensors refer to the output
                if of the wrappe module
        """
        out = self.component(batch)
        return (out, out)


class NoOpBackboneLevel(BackboneLevel):
    """
    Forward input
    """

    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pass through batch through level

        Args:
            batch: input batch [N, C, dims]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Both tensors refer to the output
                if of the wrappe module
        """
        return (batch, batch)


class LevelBackbone(AbstractBackbone):
    def __init__(
        self,
        conv: CONV_GENERATOR,
        in_channels: int,
        stem_cfg: Dict,
        level_cfgs: List[Dict],
    ) -> None:
        """
        Provide a structured template to build module backbones in nnDeteciton.
        This template first executes a stem and then uses levels to propagate
        the information. Each level return the information to propagate through
        the network and the information which should be returned to the neck.

        Args:
            conv: generator to build a conv with optional act, norm etc.
            in_channels: number of input channels (usually number of modalities)
            stem_cfg: configuration parameters of stem. If None, an empty
                dict will be passed.
            level_cfgs: configuration for each level. If None, an empty
                dict will be passed.
        """
        super().__init__()
        self.dim = conv.dim
        self.num_levels = len(level_cfgs)

        # caching variables
        self.in_channels: int = in_channels
        self.out_channels: List[int] = []
        self.relative_strides: List[ND_INT] = []

        # build stem
        _start_channels, _stem = self._build_stem(
            conv=conv,
            in_channels=in_channels,
            stem_cfg=stem_cfg,
        )
        self.start_channels: int = _start_channels
        self.stem: Optional[torch.nn.Module] = _stem

        # build levels
        levels = []
        for idx in range(self.num_levels):
            _out_channels, _stride, _level = self._build_level(
                conv=conv,
                level_idx=idx,
                level_cfg=level_cfgs[idx],
            )
            self.out_channels.append(_out_channels)
            self.relative_strides.append(_stride)
            levels.append(_level)
        self.levels: torch.nn.Module = torch.nn.ModuleList(levels)

    def get_channels(self) -> List[int]:
        """
        Compute number of channels for each returned feature map
        inside the forward pass

        Returns
            List[int]: list with number of channels corresponding to
                returned feature maps
        """
        return self.out_channels

    def get_strides(self) -> List[ND_TUPLE_INT]:
        """
        Retrieve absolute strides of the backbone feature maps
        (ordered from lowest to highest strides)

        Returns
            List[List[int]]: defines the absolute stride for each output
                feature map with respect to input size
        """
        out_strides = []
        for level_idx in range(self.num_levels):
            if level_idx == 0:
                out_strides.append(
                    to_nd_tuple(self.relative_strides[level_idx], self.dim)
                )
            else:
                _stride = to_nd_tuple(self.relative_strides[level_idx], self.dim)
                absolute_stride = [
                    prev_stride * rel_stride
                    for prev_stride, rel_stride in zip(
                        out_strides[level_idx - 1],
                        _stride,
                    )
                ]
                out_strides.append(absolute_stride)
        return out_strides

    @abstractmethod
    def _build_stem(
        self,
        conv: CONV_GENERATOR,
        stem_cfg: Dict,
    ) -> Tuple[int, Optional[torch.nn.Module]]:
        """
        Buld the network stem. This is executed befor any levels is executed.

        Args:
            stem_config: configuration of stem

        Returns:
            Tuple[int, torch.nn.Module]: The first output is the number of
                output channels of the stem. The second output is the
                stem itself.
        """
        raise NotImplementedError

    @abstractmethod
    def _build_level(
        self,
        conv: CONV_GENERATOR,
        level_idx: int,
        level_cfg: Dict,
    ) -> Tuple[int, ND_INT, BackboneLevel]:
        """
        Build one backbone level. Each level returns the features to propagate
        deeper through the network and the features which should be returned
        by the backbone.

        Args:
            level_idx: index of level
            level_config: configuration of level

        Returns:
            int: number of output channels
            ND_INT: (relative) stride of level
            BackboneLevel: constructed level
        """
        raise NotImplementedError

    def forward(self, batch: torch.Tensor) -> List[torch.Tensor]:
        """
        Foward batch through all stem and all levels

        Args:
            batch: input batch [N, C, dims]

        Returns:
            List[torch.Tensor]: output features from each level
        """
        if self.stem is not None:
            x = self.stem(batch)
        else:
            x = batch

        all_outs = []
        for level_idx in range(self.num_levels):
            x, out = self.levels[level_idx](x)
            all_outs.append(out)
        return all_outs
