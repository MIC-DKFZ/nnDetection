import copy

import torch

from nndet.io.transforms.spatial import (
    Mirror,
    boxes2points,
    mirror_points,
    nd_mirror_matrix,
    points2boxes,
    points_to_cartesian,
    points_to_homogeneous,
)


def test_single_mirror_corner_cases():
    trafo = Mirror(
        keys=["data"],
        dims=[0, 1, 2],
        box_keys=["boxes"],
    )
    data = torch.zeros(1, 1, 128, 128, 128)
    boxes = torch.tensor(
        [
            [-1, -1, 1, 1, -1, 1],
            [126, 126, 128, 128, 126, 128],
            [63, 63, 64, 64, 63, 64],  # actually this object can not exist in nnDet
            [63, 63, 65, 65, 63, 65],
        ],
    )

    new_batch = trafo(**{"data": data, "boxes": [boxes]})

    boxes_expteced = torch.tensor(
        [
            [126, 126, 128, 128, 126, 128],
            [-1, -1, 1, 1, -1, 1],
            [63, 63, 64, 64, 63, 64],  # actually this object can not exist in nnDet
            [62, 62, 64, 64, 62, 64],
        ],
    ).float()
    new_boxes = new_batch["boxes"][0]

    assert torch.allclose(new_boxes, boxes_expteced)
    assert new_boxes.shape == boxes_expteced.shape


class TestMirror:
    def test_mirror(self):
        trafo = Mirror(
            keys=["data"],
            # point_keys=["points"],
            box_keys=["boxes"],
            dims=(0, 1),
        )
        data = torch.zeros(10, 10)
        # points = torch.tensor([[2, 4], [4, 5], [1, 8], [6, 2]]).float()
        boxes = torch.tensor([[1.0, 1.0, 3.0, 3.0], [3.0, 3.0, 5.0, 5.0]]).float()

        expected_data = data.flip(0, 1)
        # expected_points = torch.tensor([[8, 6], [6, 5], [9, 2], [4, 8]]).float()
        expected_boxes = torch.tensor([[6, 6, 8, 8], [4, 4, 6, 6]]).float()

        batch = trafo(
            **{
                "data": data[None, None],
                # "points": [points],
                "boxes": [boxes],
            }
        )

        assert batch["data"][0, 0].allclose(expected_data)
        # assert batch["points"][0].allclose(expected_points)
        assert batch["boxes"][0].allclose(expected_boxes)

        batch = trafo.invert(**batch)
        assert batch["data"][0, 0].allclose(data)
        # assert batch["points"][0].allclose(points)
        assert batch["boxes"][0].allclose(boxes)

    def test_box_restoration(self):
        boxes = torch.randint(0, 10, (10000, 6)).float()
        boxes[:, [2, 3, 5]] += 11
        self.check_trafo_forward_backward_boxes((0,), boxes)
        self.check_trafo_forward_backward_boxes((1,), boxes)
        self.check_trafo_forward_backward_boxes((0, 1), boxes)
        self.check_trafo_forward_backward_boxes((2,), boxes)
        self.check_trafo_forward_backward_boxes((0, 1, 2), boxes)

    def check_trafo_forward_backward_boxes(self, dims, boxes):
        _boxes = copy.deepcopy(boxes)
        data = torch.zeros(10, 10, 10)
        trafo = Mirror(keys=["data"], box_keys=["boxes"], dims=dims)
        batch = trafo(**{"data": data[None, None], "boxes": [boxes]})
        assert not _boxes.allclose(batch["boxes"][0])
        batch = trafo.invert(**batch)
        assert _boxes.allclose(batch["boxes"][0])

    def test_mirror_points(self):
        data_shapes = [(10, 10)]
        points = [torch.tensor([[2.0, 4.0]])]
        mirrored_points = mirror_points(points, (0, 1), data_shapes)
        expected_points = torch.tensor([[7.0, 5.0]])
        assert mirrored_points[0].allclose(expected_points)

    def test_nd_mirror_matrix(self):
        data_shape = (10, 10)
        mat = nd_mirror_matrix(2, [1], data_shape)
        expected = torch.tensor([[1, 0, 0], [0, -1, 9], [0, 0, 1]], dtype=torch.float)
        assert expected.allclose(mat)


class TestSpatialFN:
    def test_points_to_homogeneous(self):
        points = [torch.tensor([[1, 2, 3]])]
        hom_points = points_to_homogeneous(points)
        expected = torch.tensor([[1, 2, 3, 1]])
        assert hom_points[0].allclose(expected)

    def test_points_to_cartesian(self):
        points = [torch.tensor([[2, 4, 8, 2]]).float()]
        cat_points = points_to_cartesian(points)
        expected = torch.tensor([[1, 2, 4]]).float()
        assert cat_points[0].allclose(expected)

    def test_points2boxes_2d(self):
        points = torch.cat([torch.tensor([[0.0, 0.0]] * 3), torch.tensor([[1.0, 1.0]] * 3)], dim=0)
        expected_boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]] * 3)

        boxes = points2boxes(points)
        assert expected_boxes.allclose(boxes)

    def test_points2boxes_3d(self):
        points = torch.cat([torch.tensor([[0.0, 0.0, 0.0]] * 3), torch.tensor([[1.0, 1.0, 1.0]] * 3)])
        expected_boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 1.0]] * 3)

        boxes = points2boxes(points)
        assert expected_boxes.allclose(boxes)

    def test_boxes2points_2d(self):
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]] * 3)
        expected_points = torch.cat([torch.tensor([[0.0, 0.0]] * 3), torch.tensor([[1.0, 1.0]] * 3)], dim=0)

        points = boxes2points(boxes)
        assert expected_points.allclose(points)

    def test_boxes2points_3d(self):
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 1.0]] * 3)
        expected_points = torch.cat([torch.tensor([[0.0, 0.0, 0.0]] * 3), torch.tensor([[1.0, 1.0, 1.0]] * 3)])

        points = boxes2points(boxes)
        assert expected_points.allclose(points)
