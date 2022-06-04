from typing import Dict, Optional, Sequence, Union

import torch

from nndet.utils.typing import ND_INT, ND_TUPLE_INT


class BodyInfo:
    def __init__(
        self,
        ndim: int,
        strides: Union[Dict[str, ND_INT], Sequence[ND_INT]],
        kernels: Union[Dict[str, ND_INT], Sequence[ND_INT]],
        channels: Optional[Union[Dict[str, int], Sequence[int]]] = None,
    ) -> None:
        """
        Provides a unified container for network information with utility
        functions for access and conver them.

        Args:
            ndim: number of spatial dimensions
            strides: relative strides to use for each level. Can either be
                provided as a list or as a Dict specifying the level P[X].
                Needs to be ordered from P0 (highest res) to PX (lowest res)
            kernels: kernel sizes for each level. Can either be
                provided as a list or as a Dict specifying the level P[X]
                Needs to be ordered from P0 (highest res) to PX (lowest res)
            channels: number of feature channels. Can either be
                provided as a list or as a Dict specifying the level P[X]
                Needs to be ordered from P0 (highest res) to PX (lowest res)
        """
        self.ndim = ndim

        if len(strides) != len(kernels):
            raise ValueError("Need to provide stride for every kernel!")

        self.strides: Dict[str, ND_TUPLE_INT] = strides
        self.kernels: Dict[str, ND_TUPLE_INT] = kernels
        self.channels: Dict[str, int] = channels

        assert len(strides) == len(kernels)
        if self.channels is not None:
            assert len(channels) == self.kernels

    def level(self, idx: Union[str, int]) -> Dict[str, Union[ND_TUPLE_INT, int]]:
        """
        Retrieve information of a single level

        Args:
            idx: index of level. Can either be an integer or string as P[X]

        Returns:
            Dict[str, Union[ND_TUPLE_INT, int]]: info

                ``"stride"`` (ND_TUPLE_INT)
                    (relative) stride of level

                ``"kernel"`` (ND_TUPLE_INT)
                    kernel size

                ``"channels"`` (Optional[int])
                    number of feature channels
        """
        if isinstance(idx, int):
            _idx = f"P{idx}"
        else:
            _idx = idx

        info = {
            "stride": self.strides[_idx],
            "kernel": self.kernels[_idx],
        }
        if self.channels is not None:
            info["channels"] = self.channels[_idx]
        return info

    @property
    def channels(self):
        self._channels

    @channels.setter
    def channels(self, item) -> Dict[str, int]:
        if isinstance(item, Sequence):
            self._channels = item
        else:
            self._channels = [item for _ in range(len(self.strides))]
        assert len(self._channels) == len(self.kernels)

    @property
    def strides(self) -> Dict[str, ND_TUPLE_INT]:
        return self._strides

    @strides.setter
    def strides(self, item: Union[Dict[str, ND_INT], Sequence[ND_INT]]):
        self._strides = self.bring_to_std(item)

    @property
    def kernels(self) -> Dict[str, ND_TUPLE_INT]:
        return self._kernels

    @kernels.setter
    def kernels(self, item: Union[Dict[str, ND_INT], Sequence[ND_INT]]):
        self._kernels = self.bring_to_std(item)

    def bring_to_std(
        self,
        item: Union[Dict[str, ND_INT], Sequence[ND_INT]],
    ) -> Dict[str, ND_TUPLE_INT]:
        """
        Convert input into unified format {P[X]: ND_TUPLE_INT}

        Args:
            item: item to convert into unified format

        Raises:
            ValueError: if a Sequence[Sequence] is provided, the inner
                sequence needs to match the number of spatial dimensions
            ValueError: if a Mapping is provided, the keys needs to be
                of the correct format P[X]

        Returns:
            Dict[str, ND_TUPLE_INT]: standardized item
        """
        level_std = {}
        if isinstance(item, Sequence):
            for level, subitem in enumerate(item):
                if subitem is None:
                    continue

                if not isinstance(subitem, Sequence):
                    level_std[f"P{level}"] = tuple([subitem for _ in range(self.ndim)])
                else:
                    if not len(subitem) == self.ndim:
                        raise ValueError(
                            f"Found different dims with subitem {subitem} and ndim {self.ndim}."
                        )
                    level_std[f"P{level}"] = tuple(subitem)
        else:
            for key, item in item.items():
                if not key.startswith("P"):
                    raise ValueError(
                        f"Found inconsistent key {key}, expected key in format P[X]"
                    )

                if not isinstance(subitem, Sequence):
                    level_std[key] = tuple([subitem for _ in range(self.ndim)])
                else:
                    level_std[key] = tuple(subitem)
        return level_std

    def absolute_strides(self) -> Dict[str, ND_TUPLE_INT]:
        # current_stride = Dict
        for level in range(len(self.strides)):
            pass

    def absolute_strides_seq(self) -> Sequence[ND_TUPLE_INT]:
        pass


class BodyOutput(torch.nn.Module):
    def __init__(self, output: Dict[str, torch.Tensor]) -> None:
        """
        Defined a standardized format to work with output from the different
        bodyparts of the network (e.g. backbone, neck)

        Args:
            output: output which should be wrapped by this container. The
                container needs to be named in the format {P[X]: tensor}
                where X is an integer ranging from 0 (highest resolution)
                to N (lowest resolution)
        """
        super().__init__()

        output_std = {}
        for key, item in output.items():
            if isinstance(key, int):
                output_std[f"P{key}"] = item
            else:
                assert key.startswith("P")
                output_std[key] = item
        self.output = torch.ParameterDict(output)

    def __getitem__(self, key: Union[int, str]) -> torch.Tensor:
        """
        Access individual element

        Args:
            key: return element. If int, it will automatically formatted
                to the P[X] format

        Returns:
            torch.Tensor: selecetd element
        """
        if isinstance(key, int):
            return self.output[f"P{key}"]
        else:
            return self.output[key]

    def first_level(self) -> int:
        """
        First present level (inclusive)

        Returns:
            int: first level
        """
        levels = [int(p[1:]) for p in self.output_std.keys()]
        return min(levels)

    def last_level(self) -> int:
        """
        Last present level (inclusive)

        Returns:
            int: last level
        """
        levels = [int(p[1:]) for p in self.output_std.keys()]
        return max(levels)

    def num_levels(self) -> int:
        """
        Number of levels

        Returns:
            int: number of levels
        """
        return len(self.output_std)
