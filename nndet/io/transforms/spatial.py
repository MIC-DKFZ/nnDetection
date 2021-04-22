"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
from torch import Tensor
from typing import Sequence, List

from nndet.io.transforms.base import AbstractTransform


class Mirror(AbstractTransform):
    def __init__(self, keys: Sequence[str], dims: Sequence[int],
                 point_keys: Sequence[str] = (), box_keys: Sequence[str] = (),
                 grad: bool = False):
        """
        Mirror Transform

        Args:
            keys: keys to mirror (first key must correspond to data for
                shape information) expected shape [N, C, dims]
            dims: dimensions to mirror (starting from the first spatial
                dimension)
            point_keys: keys where points for transformation are located
                [N, dims]
            box_keys: keys where boxes are located; following format
                needs to be used (x1, y1, x2, y2, (z1, z2)) [N, dims * 2]
            grad: enable gradient computation inside transformation
        """
        super().__init__(grad=grad)
        self.dims = dims
        self.keys = keys
        self.point_keys = point_keys
        self.box_keys = box_keys

    def forward(self, **data) -> dict:
        """
        Implement transform functionality here

        Args
            data: dict with data

        Returns
            dict: dict with transformed data
        """
        for key in self.keys:
            data[key] = mirror(data[key], self.dims)

        data_shape = data[self.keys[0]].shape
        data_shapes = [tuple(data_shape[2:])] * data_shape[0]

        for key in self.box_keys:
            points = [boxes2points(b) for b in data[key]]
            points = mirror_points(points, self.dims, data_shapes)
            data[key] = [points2boxes(p) for p in points]

        for key in self.point_keys:
            data[key] = mirror_points(data[key], self.dims, data_shapes)
        return data

    def invert(self, **data) -> dict:
        """
        Revert mirroring

        Args:
            **data: dict with data

        Returns:
            dict with re-transformed data
        """
        return self(**data)


def mirror(data: torch.Tensor, dims: Sequence[int]) -> torch.Tensor:
    """
    Mirror data at dims

    Args
        data: input data [N, C, spatial dims]
        dims: dimensions to mirror starting from spatial dims
            e.g. dim=(0,) mirror the first spatial dimension

    Returns
        torch.Tensor: tensor with mirrored dimensions
    """
    dims = [d + 2 for d in dims]
    return data.flip(dims)


def mirror_points(points: Sequence[torch.Tensor], dims: Sequence[int],
                  data_shapes: Sequence[Sequence[int]]) -> List[torch.Tensor]:
    """
    Mirror points along given dimensions

    Args:
        points: points per batch element [N, dims]
        dims: dimensions to mirror
        data_shapes: shape of data

    Returns:
        Tensor: transformed points [N, dims]
    """
    cartesian_dims = points[0].shape[1]
    homogeneous_points = points_to_homogeneous(points)

    transformed = []
    for points_per_image, data_shape in zip(homogeneous_points, data_shapes):
        matrix = nd_mirror_matrix(cartesian_dims, dims, data_shape).to(points_per_image)
        transformed.append(points_per_image @ matrix.transpose(0, 1))
    return points_to_cartesian(transformed)


def nd_mirror_matrix(cartesian_dims: int, mirror_dims: Sequence[int],
                     data_shape: Sequence[int]) -> torch.Tensor:
    """
    Create n dimensional matrix to for mirroring

    Args:
        cartesian_dims: number of cartesian dimensions
        mirror_dims: dimensions to mirror
        data_shape: shape of image

    Returns:
        Tensor: matrix for mirroring in homogeneous coordinated,
            [cartesian_dims + 1, cartesian_dims + 1]
    """
    mirror_dims = tuple(mirror_dims)
    data_shape = list(data_shape)

    homogeneous_dims = cartesian_dims + 1
    mat = torch.eye(homogeneous_dims, dtype=torch.float)

    # reflection
    mat[[mirror_dims] * 2] = -1

    # add data shape to axis which were reflected
    self_tensor = torch.zeros(cartesian_dims, dtype=torch.float)
    index_tensor = torch.Tensor(mirror_dims).long()
    src_tensor = torch.tensor([1] * len(mirror_dims), dtype=torch.float)
    offset_mask = self_tensor.scatter_(0, index_tensor, src_tensor)
    mat[:-1, -1] = offset_mask * torch.tensor(data_shape)
    return mat


def points_to_homogeneous(points: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """
    Transforms points from cartesian to homogeneous coordinates

    Args:
        points: list of points to transform [N, dims] where N is the number
            of points and dims is the number of spatial dimensions

    Returns
        torch.Tensor: the batch of points in homogeneous coordinates [N, dim + 1]
    """
    return [torch.cat([p, torch.ones(p.shape[0], 1).to(p)], dim=1) for p in points]


def points_to_cartesian(points: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """
    Transforms points in homogeneous coordinates back to cartesian
    coordinates.

    Args:
        points: homogeneous points [N, in_dims], N number of points,
            in_dims number of input dimensions (spatial dimensions + 1)

    Returns:
        List[Tensor]]: cartesian points [N, in_dims] = [N, dims]
    """
    return [p[..., :-1] / p[..., -1][:, None] for p in points]


def boxes2points(boxes: Tensor) -> Tensor:
    """
    Convert boxes to points

    Args:
        boxes: (x1, y1, x2, y2, (z1, z2))[N, dims *2]

    Returns:
        Tensor: points [N * 2, dims]
    """
    if boxes.shape[1] == 4:
        idx0 = [0, 1]
        idx1 = [2, 3]
    else:
        idx0 = [0, 1, 4]
        idx1 = [2, 3, 5]

    points0 = boxes[:, idx0]
    points1 = boxes[:, idx1]
    return torch.cat([points0, points1], dim=0)


def points2boxes(points: Tensor) -> Tensor:
    """
    Convert points to boxes

    Args:
        points: boxes need to be order as specified
            order: [point_box_0, ... point_box_N/2] * 4
            format of points: (x, y(, z)))[N, dims]

    Returns:
        Tensor: bounding boxes [N / 2, dims * 2]
    """
    if points.nelement() > 0:
        points0, points1 = points.split(points.shape[0] // 2)
        boxes = torch.zeros(points.shape[0] // 2, points.shape[1] * 2).to(
            device=points.device, dtype=points.dtype)
        boxes[:, 0] = torch.min(points0[:, 0], points1[:, 0])
        boxes[:, 1] = torch.min(points0[:, 1], points1[:, 1])
        boxes[:, 2] = torch.max(points0[:, 0], points1[:, 0])
        boxes[:, 3] = torch.max(points0[:, 1], points1[:, 1])
        if boxes.shape[1] == 6:
            boxes[:, 4] = torch.min(points0[:, 2], points1[:, 2])
            boxes[:, 5] = torch.max(points0[:, 2], points1[:, 2])
        return boxes
    else:
        return torch.tensor([]).view(-1, points.shape[1] * 2).to(points)
