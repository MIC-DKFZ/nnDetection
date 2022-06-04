# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
# SPDX-License-Identifier: Apache-2.0
#
# Parts of this code are from torchvision (https://github.com/pytorch/vision) licensed under
# SPDX-FileCopyrightText: 2016 Soumith Chintala
# SPDX-License-Identifier: BSD-3-Clause


import torch
from typing import Callable, Sequence, List, Tuple, TypeVar, Union
from torchvision.models.detection.rpn import AnchorGenerator
from loguru import logger
from itertools import product


AnchorGeneratorType = TypeVar('AnchorGeneratorType', bound=AnchorGenerator)


def get_anchor_generator(dim: int, s_param: bool = False) -> AnchorGenerator:
    """
    Get anchor generator class for corresponding dimension

    Args:
        dim: number of spatial dimensions
        s_param: enable size parametrization

    Returns:
        Callable: class of anchor generator
    """
    normal = {2: AnchorGenerator2D, 3: AnchorGenerator3D}
    sparam = {2: AnchorGenerator2DS, 3: AnchorGenerator3DS}

    if s_param:
        return sparam[dim]
    else:
        return normal[dim]


def compute_anchors_for_strides(anchors: torch.Tensor,
                                strides: Sequence[Union[Sequence[Union[int, float]], Union[int, float]]],
                                cat: bool) -> Union[List[torch.Tensor], torch.Tensor]:
    """
    Compute anchors sizes which follow a given sequence of strides
    
    Args:
        anchors: anchors for stride 0
        strides: sequence of strides to adjust anchors for
        cat: concatenate resulting anchors, if false a Sequence of Anchors
            is returned
    
    Returns:
        Union[List[torch.Tensor], torch.Tensor]: new anchors
    """
    anchors_with_stride = [anchors]
    dim = anchors.shape[1] // 2
    for stride in strides:
        if isinstance(stride, (int, float)):
            stride = [stride] * dim
        
        stride_formatted = [stride[0], stride[1], stride[0], stride[1]]
        if dim == 3:
            stride_formatted.extend([stride[2], stride[2]])
        anchors_with_stride.append(
            anchors * torch.tensor(stride_formatted)[None].float())
    if cat:
        anchors_with_stride = torch.cat(anchors_with_stride, dim=0)
    return anchors_with_stride


class AnchorGenerator2D(torch.nn.Module):
    def __init__(self, sizes: Sequence[Union[int, Sequence[int]]] = (128, 256, 512),
                 aspect_ratios: Sequence[Union[float, Sequence[float]]] = (0.5, 1.0, 2.0),
                 **kwargs):
        """
        Generator for anchors
        Modified from https://github.com/pytorch/vision/blob/master/torchvision/models/detection/rpn.py

        Args:
            sizes (Sequence[Union[int, Sequence[int]]]): anchor sizes for each feature map
                (length should match the number of feature maps)
            aspect_ratios (Sequence[Union[float, Sequence[float]]]): anchor aspect ratios:
                height/width, e.g. (0.5, 1, 2). if Seq[Seq] is provided, it should have
                the same length as sizes
        """
        super().__init__()
        if not isinstance(sizes[0], (list, tuple)):
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)
        assert len(sizes) == len(aspect_ratios)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}

        self.num_anchors_per_level: List[int] = None
        if kwargs:
            logger.info(f"Discarding anchor generator kwargs {kwargs}")

    def cached_grid_anchors(self, grid_sizes: List[List[int]], strides: List[List[int]]) -> List[torch.Tensor]:
        """
        Check if combination was already generated before and return that if possible

        Args:
            grid_sizes (Sequence[Sequence[int]]): spatial sizes of feature maps
            strides (Sequence[Sequence[int]]): stride of each feature map

        Returns:
            List[torch.Tensor]: Anchors for each feature maps
        """
        key = str(grid_sizes + strides)
        if key not in self._cache:
            self._cache[key] = self.grid_anchors(grid_sizes, strides)

        self.num_anchors_per_level = self._cache[key][1]
        return self._cache[key][0]

    def grid_anchors(self, grid_sizes, strides) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Distribute anchors over feature maps

        Args:
            grid_sizes (Sequence[Sequence[int]]): spatial sizes of feature maps
            strides (Sequence[Sequence[int]]): stride of each feature map

        Returns:
            List[torch.Tensor]: Anchors for each feature maps
            List[int]: number of anchors per level
        """
        assert len(grid_sizes) == len(strides), "Every fm size needs strides"
        assert len(grid_sizes) == len(self.cell_anchors), "Every fm size needs cell anchors"
        anchors = []
        cell_anchors = self.cell_anchors
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
            logger.debug(f"Generated {anchors[_i].shape[0]} anchors and expected "
                         f"{size0 * size1 * self.num_anchors_per_location()[_i]} "
                         f"anchors on level {_i}.")
            _i += 1
        return anchors, anchor_per_level

    @staticmethod
    def generate_anchors(scales: Tuple[int],
                         aspect_ratios: Tuple[float],
                         dtype: torch.dtype = torch.float,
                         device: Union[torch.device, str] = "cpu",
                         ) -> torch.Tensor:
        """
        Generate anchors for a pair of scales and ratios

        Args:
            scales (Tuple[int]): scales of anchors, e.g. (32, 64, 128)
            aspect_ratios (Tuple[float]): aspect ratios of height/width, e.g. (0.5, 1, 2)
            dtype (torch.dtype): data type of anchors
            device (Union[torch.device, str]): target device of anchors

        Returns:
            Tensor: anchors of shape [n(scales) * n(ratios), dim * 2]
        """
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()

    def set_cell_anchors(self,  dtype: torch.dtype, device: Union[torch.device, str] = "cpu") -> None:
        """
        Set :para:`self.cell_anchors` if it was not already set

        Args:
            dtype (torch.dtype): data type of anchors
            device (Union[torch.device, str]): target device of anchors

        Returns:
        None
            result is saved into attribute
        """
        if self.cell_anchors is not None:
            return

        cell_anchors = [self.generate_anchors(sizes, aspect_ratios, dtype, device)
                        for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)]
        self.cell_anchors = cell_anchors

    def forward(self, image_list: torch.Tensor, feature_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Generate anchors for given feature maps
        # TODO: update docstring and type
        Args:
            image_list (torch.Tensor): data structure which contains images and their original shapes
            feature_maps (Sequence[torch.Tensor]): feature maps for which anchors need to be generated

        Returns:
            List[Tensor]: list of anchors (for each image inside the batch)
        """
        device = image_list.device
        grid_sizes = list([feature_map.shape[2:] for feature_map in feature_maps])
        image_size = image_list.shape[2:]
        strides = [list((int(i / s) for i, s in zip(image_size, fm_size))) for fm_size in grid_sizes]

        self.set_cell_anchors(dtype=feature_maps[0].dtype, device=feature_maps[0].device)
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)

        anchors = []
        images_shapes = [img.shape for img in image_list.split(1)]
        for i, x in enumerate(images_shapes):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)
        anchors = [torch.cat(anchors_per_image).to(device) for anchors_per_image in anchors]

        # TODO: check with torchvision if this makes sense (if enabled, anchors are newly generated for each run)
        # # Clear the cache in case that memory leaks.
        # self._cache.clear()
        return anchors

    def num_anchors_per_location(self) -> List[int]:
        """
        Number of anchors per resolution

        Returns:
            List[int]: number of anchors per positions for each resolution
        """
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    def get_num_acnhors_per_level(self) -> List[int]:
        """
        Number of anchors per resolution

        Returns:
            List[int]: number of anchors per positions for each resolution
        """
        if self.num_anchors_per_level is None:
            raise RuntimeError("Need to forward features maps before "
                               "get_num_acnhors_per_level can be called")
        return self.num_anchors_per_level


class AnchorGenerator3D(AnchorGenerator2D):
    def __init__(self,
                 sizes: Sequence[Union[int, Sequence[int]]] = (128, 256, 512),
                 aspect_ratios: Sequence[Union[float, Sequence[float]]] = (0.5, 1.0, 2.0),
                 zsizes: Sequence[Union[int, Sequence[int]]] = (4, 4, 4),
                 **kwargs):
        """
        Helper to generate anchors for different input sizes

        Args:
            sizes (Sequence[Union[int, Sequence[int]]]): anchor sizes for each feature map
                (length should match the number of feature maps)
            aspect_ratios (Sequence[Union[float, Sequence[float]]]): anchor aspect ratios:
                height/width, e.g. (0.5, 1, 2). if Seq[Seq] is provided, it should have
                the same length as sizes
            zsizes (Sequence[Union[int, Sequence[int]]]): sizes along z dimension
        """
        super().__init__(sizes, aspect_ratios)
        if not isinstance(zsizes[0], (Sequence, list, tuple)):
            zsizes = (zsizes,) * len(sizes)
        self.zsizes = zsizes
        if kwargs:
            logger.info(f"Discarding anchor generator kwargs {kwargs}")

    def set_cell_anchors(self, dtype: torch.dtype, device: Union[torch.device, str] = "cpu") -> None:
        """
        Compute anchors for all pairs of sclaes and ratios and save them inside :param:`cell_anchors`
        if they were not computed before

        Args:
            dtype (torch.dtype): data type of anchors
            device (Union[torch.device, str]): target device of anchors

        Returns:
            None (result is saved into :param:`self.cell_anchors`)
        """
        if self.cell_anchors is not None:
            return

        cell_anchors = [
            self.generate_anchors(sizes, aspect_ratios, zsizes, dtype, device)
            for sizes, aspect_ratios, zsizes in zip(self.sizes, self.aspect_ratios, self.zsizes)
        ]
        self.cell_anchors = cell_anchors

    @staticmethod
    def generate_anchors(scales: Tuple[int], aspect_ratios: Tuple[float], zsizes: Tuple[int],
                         dtype: torch.dtype = torch.float,
                         device: Union[torch.device, str] = "cpu") -> torch.Tensor:
        """
        Generate anchors for a pair of scales and ratios

        Args:
            scales (Tuple[int]): scales of anchors, e.g. (32, 64, 128)
            aspect_ratios (Tuple[float]): aspect ratios of height/width, e.g. (0.5, 1, 2)
            zsizes (Tuple[int]): scale along z dimension
            dtype (torch.dtype): data type of anchors
            device (Union[torch.device, str]): target device of anchors

        Returns:
            Tensor: anchors of shape [n(scales) * n(ratios) * n(zscales) , dim * 2]
        """
        base_anchors_2d = AnchorGenerator2D.generate_anchors(
            scales, aspect_ratios, dtype=dtype, device=device)
        zanchors = torch.cat(
            [torch.as_tensor([-z, z], dtype=dtype, device=device).repeat(
                base_anchors_2d.shape[0], 1) for z in zsizes], dim=0)
        base_anchors_3d = torch.cat(
            [base_anchors_2d.repeat(len(zsizes), 1), (zanchors / 2.).round()], dim=1)
        return base_anchors_3d

    def grid_anchors(self, grid_sizes: Sequence[Sequence[int]],
                     strides: Sequence[Sequence[int]]) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Distribute anchors over feature maps

        Args:
            grid_sizes (Sequence[Sequence[int]]): spatial sizes of feature maps
            strides (Sequence[Sequence[int]]): stride of each feature map

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
            logger.debug(f"Generated {_anchors.shape[0]} anchors and expected "
                         f"{size0 * size1 * size2 * self.num_anchors_per_location()[_i]} "
                         f"anchors on level {_i}.")
            _i += 1
        return anchors, anchor_per_level

    def num_anchors_per_location(self) -> List[int]:
        """
        Number of anchors per resolution

        Returns:
            List[int]: number of anchors per positions for each resolution
        """
        return [len(s) * len(a) * len(z) for s, a, z in zip(self.sizes, self.aspect_ratios, self.zsizes)]


class AnchorGenerator2DS(AnchorGenerator2D):
    def __init__(self,
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

    def set_cell_anchors(self, dtype: torch.dtype,
                         device: Union[torch.device, str] = "cpu") -> None:
        """
        Compute anchors for all pairs of sclaes and ratios and
        save them inside :param:`cell_anchors`
        if they were not computed before

        Args:
            dtype (torch.dtype): data type of anchors
            device (Union[torch.device, str]): target device of anchors

        Returns:
            None (result is saved into :param:`self.cell_anchors`)
        """
        if self.cell_anchors is not None:
            return

        cell_anchors = [
            self.generate_anchors(w, h, dtype, device)
            for w, h in zip(self.width, self.height)
        ]
        self.cell_anchors = cell_anchors

    @staticmethod
    def generate_anchors(width: Tuple[int],
                         height: Tuple[int],
                         dtype: torch.dtype = torch.float,
                         device: Union[torch.device, str] = "cpu",
                         ) -> torch.Tensor:
        """
        Generate anchors for given width, height and depth sizes

        Args:
            width: sizes along width dimension
            height: sizes along height dimension

        Returns:
            Tensor: anchors of shape [n(width) * n(height), dim * 2]
        """
        all_sizes = torch.tensor(list(product(width, height)),
                                 dtype=dtype, device=device) / 2
        anchors = torch.stack([-all_sizes[:, 0], -all_sizes[:, 1],
                               all_sizes[:, 0], all_sizes[:, 1]], dim=1)
        return anchors

    def num_anchors_per_location(self) -> List[int]:
        """
        Number of anchors per resolution

        Returns:
            List[int]: number of anchors per positions for each resolution
        """
        return [len(w) * len(h) for w, h in zip(self.width, self.height)]


class AnchorGenerator3DS(AnchorGenerator3D):
    def __init__(self,
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

    def set_cell_anchors(self, dtype: torch.dtype, device: Union[torch.device, str] = "cpu") -> None:
        """
        Compute anchors for all pairs of scales and ratios and save them inside :param:`cell_anchors`
        if they were not computed before

        Args:
            dtype (torch.dtype): data type of anchors
            device (Union[torch.device, str]): target device of anchors

        Returns:
            None (result is saved into :param:`self.cell_anchors`)
        """
        if self.cell_anchors is not None:
            return

        cell_anchors = [
            self.generate_anchors(w, h, d, dtype, device)
            for w, h, d in zip(self.width, self.height, self.depth)
        ]
        self.cell_anchors = cell_anchors

    @staticmethod
    def generate_anchors(width: Tuple[int],
                         height: Tuple[int],
                         depth: Tuple[int],
                         dtype: torch.dtype = torch.float,
                         device: Union[torch.device, str] = "cpu") -> torch.Tensor:
        """
        Generate anchors for given width, height and depth sizes

        Args:
            width: sizes along width dimension
            height: sizes along height dimension
            depth: sizes along depth dimension

        Returns:
            Tensor: anchors of shape [n(width) * n(height) * n(depth) , dim * 2]
        """
        all_sizes = torch.tensor(list(product(width, height, depth)),
                                 dtype=dtype, device=device) / 2
        anchors = torch.stack(
            [-all_sizes[:, 0], -all_sizes[:, 1], all_sizes[:, 0], all_sizes[:, 1],
             -all_sizes[:, 2], all_sizes[:, 2]], dim=1
            )
        return anchors

    def num_anchors_per_location(self) -> List[int]:
        """
        Number of anchors per resolution

        Returns:
            List[int]: number of anchors per positions for each resolution
        """
        return [len(w) * len(h) * len(d) 
                for w, h, d in zip(self.width, self.height, self.depth)]
