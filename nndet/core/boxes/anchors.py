# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
#
# Parts of this code are from torchvision (https://github.com/pytorch/vision) licensed under
# SPDX-FileCopyrightText: 2016 Soumith Chintala
# SPDX-License-Identifier: BSD-3-Clause


from abc import abstractmethod, abstractstaticmethod
from itertools import product
from typing import List, Sequence, Tuple, Union

import torch
from loguru import logger


class AnchorGenerator(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_anchors_per_level: List[int] = None

    def forward(
        self,
        images: torch.Tensor,
        feature_maps: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Generate anchors for given feature maps

        Args:
            images: input batch of images in shape [N, C, dims] where
                N is the batch size, C is the number of channels and
                dims are the spatial dimension
            feature_maps: feature maps for which anchors need to be generated
                each of them with size [N, C, dims]

        Returns:
            List[Tensor]: list of anchors (for each image inside the batch)
                List([R, sdims * 2]) where R is the number of anchors per
                image and sdims is the number of spatial dimensions
        """
        # TODO: rework device and dtype handling here
        # TODO: check speed when anchor are generated for each run
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        image_size = images.shape[2:]
        grid_sizes = [feature_map.shape[2:] for feature_map in feature_maps]
        strides = [list((int(i / s) for i, s in zip(image_size, fm_size))) for fm_size in grid_sizes]

        anchors_over_all_feature_maps, num_anchors_per_level = self._grid_anchors(
            grid_sizes=grid_sizes,
            strides=strides,
            device=device,
            dtype=dtype,
        )
        self.num_anchors_per_level = num_anchors_per_level

        anchors: List[List[torch.Tensor]] = []
        for _ in range(images.shape[0]):
            anchors_in_image = [anchors_per_feature_map for anchors_per_feature_map in anchors_over_all_feature_maps]
            anchors.append(anchors_in_image)
        # TODO: optionally remove to device statement
        anchors = [torch.cat(anchors_per_image).to(device) for anchors_per_image in anchors]
        return anchors

    @abstractmethod
    def _grid_anchors(
        self,
        grid_sizes: Sequence[Sequence[int]],
        strides: Sequence[Sequence[int]],
        dtype: torch.dtype,
        device: Union[torch.device, str],
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Distribute anchors over feature maps

        Args:
            grid_sizes: spatial sizes of feature maps
            strides: stride of each feature map
            #TODO: docs

        Returns:
            List[torch.Tensor]: Anchors for each feature maps
            List[int]: number of anchors per level
        """
        raise NotImplementedError

    @abstractstaticmethod
    def generate_anchors(
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate anchors for given sizes

        Args:
            #TODO: docs
            width: sizes along width dimension
            height: sizes along height dimension
            depth: sizes along depth dimension

        Returns:
            Tensor: anchors of shape [n(width) * n(height) * n(depth) , dim * 2]
        """
        raise NotImplementedError

    @abstractmethod
    def num_anchors_per_location(self) -> List[int]:
        """
        Number of anchors per resolution

        Returns:
            List[int]: number of anchors per positions for each resolution
        """
        raise NotImplementedError

    def get_num_anchors_per_level(self) -> List[int]:
        """
        Number of anchors per resolution

        Returns:
            List[int]: number of anchors per positions for each resolution
        """
        if self.num_anchors_per_level is None:
            raise RuntimeError("Need to forward features maps before " "get_num_acnhors_per_level can be called")
        return self.num_anchors_per_level


class AnchorGenerator2D(AnchorGenerator):
    def __init__(
        self,
        width: Sequence[Union[int, Sequence[int]]],
        height: Sequence[Union[int, Sequence[int]]],
        **kwargs,
    ):
        """
        Helper to generate anchors for different input sizes
        Uses a different parametrization of anchors
        (if Sequence[int] is provided it is interpreted as one
        value per feature map size)

        Args:
            width: sizes along width dimension
            height: sizes along height dimension
        """
        # TODO: check width and height statements
        super().__init__()
        if not isinstance(width[0], Sequence):
            width = [(w,) for w in width]
        if not isinstance(height[0], Sequence):
            height = [(h,) for h in height]
        self.width = width
        self.height = height
        assert len(self.width) == len(self.height)
        if kwargs:
            logger.info(f"Discarding anchor generator kwargs {kwargs}")

        self.cell_anchors: List[torch.Tensor] = [
            self.generate_anchors(width=w, height=h) for w, h in zip(self.width, self.height)
        ]

    def _grid_anchors(
        self,
        grid_sizes: Sequence[Sequence[int]],
        strides: Sequence[Sequence[int]],
        dtype: torch.dtype,
        device: Union[torch.device, str],
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Distribute anchors over feature maps

        Args:
            grid_sizes: spatial sizes of feature maps
            strides: stride of each feature map
            # TODO: docs

        Returns:
            List[torch.Tensor]: Anchors for each feature maps
            List[int]: number of anchors per level
        """
        assert len(grid_sizes) == len(strides), "Every fm size needs strides"
        assert len(grid_sizes) == len(self.cell_anchors), "Every fm size needs cell anchors"
        anchors = []
        cell_anchors = [ca.to(device=device, dtype=dtype) for ca in self.cell_anchors]
        assert cell_anchors is not None

        _i = 0
        # modified from torchvision (ordering of axis differs)
        anchor_per_level = []
        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            size0, size1 = size
            stride0, stride1 = stride
            device = base_anchors.device

            shifts_x = torch.arange(0, size0, dtype=torch.float, device=device) * stride0
            shifts_y = torch.arange(0, size1, dtype=torch.float, device=device) * stride1

            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            _anchors = (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            anchors.append(_anchors)
            anchor_per_level.append(_anchors.shape[0])
            logger.debug(
                f"Generated {anchors[_i].shape[0]} anchors and expected "
                f"{size0 * size1 * self.num_anchors_per_location()[_i]} "
                f"anchors on level {_i}."
            )
            _i += 1
        return anchors, anchor_per_level

    @staticmethod
    def generate_anchors(
        width: Tuple[int],
        height: Tuple[int],
    ) -> torch.Tensor:
        """
        Generate anchors for given width, height and depth sizes

        Args:
            width: sizes along width dimension
            height: sizes along height dimension

        Returns:
            Tensor: anchors of shape [n(width) * n(height), dim * 2]
        """
        all_sizes = torch.tensor(list(product(width, height))) / 2
        anchors = torch.stack(
            [-all_sizes[:, 0], -all_sizes[:, 1], all_sizes[:, 0], all_sizes[:, 1]],
            dim=1,
        )
        return anchors

    def num_anchors_per_location(self) -> List[int]:
        """
        Number of anchors per resolution

        Returns:
            List[int]: number of anchors per positions for each resolution
        """
        return [len(w) * len(h) for w, h in zip(self.width, self.height)]


class AnchorGenerator3D(AnchorGenerator):
    def __init__(
        self,
        width: Sequence[Union[int, Sequence[int]]],
        height: Sequence[Union[int, Sequence[int]]],
        depth: Sequence[Union[int, Sequence[int]]],
        **kwargs,
    ):
        """
        Helper to generate anchors for different input sizes
        Uses a different parametrization of anchors
        (if Sequence[int] is provided it is interpreted as one
        value per feature map size)

        Args:
            width: sizes along width dimension
            height: sizes along height dimension
            depth: sizes along depth dimension
        """
        # TODO: check width and height statements
        super().__init__()
        if not isinstance(width[0], Sequence):
            width = [(w,) for w in width]
        if not isinstance(height[0], Sequence):
            height = [(h,) for h in height]
        if not isinstance(depth[0], Sequence):
            depth = [(d,) for d in depth]
        self.width = width
        self.height = height
        self.depth = depth
        assert len(self.width) == len(self.height) == len(self.depth)
        if kwargs:
            logger.info(f"Discarding anchor generator kwargs {kwargs}")

        self.cell_anchors = [self.generate_anchors(w, h, d) for w, h, d in zip(self.width, self.height, self.depth)]

    def _grid_anchors(
        self,
        grid_sizes: Sequence[Sequence[int]],
        strides: Sequence[Sequence[int]],
        dtype: torch.dtype,
        device: Union[torch.device, str],
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Distribute anchors over feature maps

        Args:
            grid_sizes: spatial sizes of feature maps
            strides: stride of each feature map
            #TODO: docs

        Returns:
            List[torch.Tensor]: Anchors for each feature maps
            List[int]: number of anchors per level
        """
        assert len(grid_sizes) == len(strides)
        assert len(grid_sizes) == len(self.cell_anchors)
        anchors = []
        _i = 0
        anchor_per_level = []
        for size, stride, base_anchors in zip(grid_sizes, strides, self.cell_anchors):
            size0, size1, size2 = size
            stride0, stride1, stride2 = stride
            dtype, device = base_anchors.dtype, base_anchors.device

            shifts_x = torch.arange(0, size0, dtype=dtype, device=device) * stride0
            shifts_y = torch.arange(0, size1, dtype=dtype, device=device) * stride1
            shifts_z = torch.arange(0, size2, dtype=dtype, device=device) * stride2

            shift_x, shift_y, shift_z = torch.meshgrid(shifts_x, shifts_y, shifts_z, indexing="ij")
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shift_z = shift_z.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y, shift_z, shift_z), dim=1)

            _anchors = (shifts.view(-1, 1, 6) + base_anchors.view(1, -1, 6)).reshape(-1, 6)
            anchors.append(_anchors)
            anchor_per_level.append(_anchors.shape[0])
            logger.debug(
                f"Generated {_anchors.shape[0]} anchors and expected "
                f"{size0 * size1 * size2 * self.num_anchors_per_location()[_i]} "
                f"anchors on level {_i}."
            )
            _i += 1
        return anchors, anchor_per_level

    @staticmethod
    def generate_anchors(
        width: Tuple[int],
        height: Tuple[int],
        depth: Tuple[int],
        dtype: torch.dtype = torch.float,
        device: Union[torch.device, str] = "cpu",
    ) -> torch.Tensor:
        """
        Generate anchors for given width, height and depth sizes

        Args:
            width: sizes along width dimension
            height: sizes along height dimension
            depth: sizes along depth dimension

        Returns:
            Tensor: anchors of shape [n(width) * n(height) * n(depth) , dim * 2]
        """
        all_sizes = torch.tensor(list(product(width, height, depth)), dtype=dtype, device=device) / 2
        anchors = torch.stack(
            [
                -all_sizes[:, 0],
                -all_sizes[:, 1],
                all_sizes[:, 0],
                all_sizes[:, 1],
                -all_sizes[:, 2],
                all_sizes[:, 2],
            ],
            dim=1,
        )
        return anchors

    def num_anchors_per_location(self) -> List[int]:
        """
        Number of anchors per resolution

        Returns:
            List[int]: number of anchors per positions for each resolution
        """
        return [len(w) * len(h) * len(d) for w, h, d in zip(self.width, self.height, self.depth)]
