import numpy as np
import pytest
import torch
from torchvision.ops.boxes import nms as nms_torchvision

from nndet.core.boxes.nms import batched_nms, nms
from nndet.core.boxes.nms import nms_cpu as nms_pytorch
from nndet.core.boxes.nms import nms_gpu


def generate_boxes(
    count, dim=2, h=100, w=100, d=20, normalize=False, on_grid=False, seed=0
):
    """
    generate radnom boxes of format [y1, x1, y2, x2, (z1, z2)]
    :param count: nr of boxes
    :param dim: dimension of boxes (2 or 3)
    :return: boxes in format (n_boxes, 4 or 6), scores
    """
    np.random.seed(seed)
    if on_grid:
        lower_y = np.random.randint(0, h // 2, (count,))
        lower_x = np.random.randint(0, w // 2, (count,))
        upper_y = np.random.randint(h // 2, h, (count,))
        upper_x = np.random.randint(w // 2, w, (count,))
        if dim == 3:
            lower_z = np.random.randint(0, d // 2, (count,))
            upper_z = np.random.randint(d // 2, d, (count,))
    else:
        lower_y = np.random.rand(count) * h / 2.0
        lower_x = np.random.rand(count) * w / 2.0
        upper_y = (np.random.rand(count) + 1.0) * h / 2.0
        upper_x = (np.random.rand(count) + 1.0) * w / 2.0
        if dim == 3:
            lower_z = np.random.rand(count) * d / 2.0
            upper_z = (np.random.rand(count) + 1.0) * d / 2.0

    if dim == 3:
        boxes = np.array(
            list(zip(lower_y, lower_x, upper_y, upper_x, lower_z, upper_z))
        )
        # add an extreme box that tests the boundaries
        boxes = np.concatenate((boxes, np.array([[0.0, 0.0, h, w, 0, d]])))
    else:
        boxes = np.array(list(zip(lower_y, lower_x, upper_y, upper_x)))
        boxes = np.concatenate((boxes, np.array([[0.0, 0.0, h, w]])))

    scores = np.random.rand(count + 1)
    if normalize:
        divisor = np.array([h, w, h, w, d, d]) if dim == 3 else np.array([h, w, h, w])
        boxes = boxes / divisor
    return torch.from_numpy(boxes).float(), torch.from_numpy(scores).float()


def generate_2d_fixed():
    """
    Generate 2d example

    Returns:
        Tensor: boxes (x1, y1, x2, y2)[N, 4]
        Tensor: scores [N]
        Tensor: expected keep [M] (for threshold 0.01 (much smaller than in real application))
    """
    boxes = torch.tensor([[0, 0, 2, 2], [1, 1, 3, 3], [2, 2, 4, 4]]).float()
    scores = torch.tensor([1, 0.8, 0.6]).float()
    expected = torch.tensor([0, 2])
    return boxes, scores, expected


def generate_3d_fixed():
    """
    Generate 2d example

    Returns:
        Tensor: boxes (x1, y1, x2, y2, (z1, z2))[N, 6]
        Tensor: scores [N]
        Tensor: expected keep [M] (for threshold 0.01)
    """
    boxes = torch.tensor(
        [[0, 0, 2, 2, 0, 2], [1, 1, 3, 3, 1, 3], [2, 2, 4, 4, 2, 4]]
    ).float()
    scores = torch.tensor([1, 0.8, 0.6]).float()
    expected = torch.tensor([0, 2])
    return boxes, scores, expected


@pytest.fixture
def th():
    return 0.01


class TestNMS:
    def test_nms_torchvision_2d_cpu(self, th):
        boxes, scores, expected = generate_2d_fixed()
        computed_wrapper = nms(boxes, scores, th)
        computed_vision = nms(boxes, scores, th)
        assert (computed_wrapper == expected).all()
        assert (computed_vision == expected).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda gpu available")
    @pytest.mark.skipif(
        nms_gpu is None, reason="nnDetection was not build with GPU support"
    )
    def test_nms_torchvision_2d_gpu(self, th):
        boxes, scores, expected = generate_2d_fixed()
        boxes, scores, expected = boxes.cuda(), scores.cuda(), expected.cuda()
        computed_wrapper = nms(boxes, scores, th)
        computed_vision = nms(boxes, scores, th)
        assert (computed_wrapper == expected).all()
        assert (computed_vision == expected).all()

    def test_nms_pytorch_2d_fixed(self, th):
        boxes, scores, expected = generate_2d_fixed()
        computed = nms_pytorch(boxes, scores, th)
        assert (computed == expected).all()

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_nms_pytorch_2d_random(self, th, seed):
        np.random.seed(seed)
        boxes, scores = generate_boxes(1000, seed=seed)
        computed_vision = nms_torchvision(boxes, scores, th)
        computed_pytorch = nms_pytorch(boxes, scores, th)
        assert (computed_vision == computed_pytorch).all()

    def test_nms_cuda_3d_fixed_cpu(self, th):
        boxes, scores, expected = generate_3d_fixed()
        boxes, scores, expected = boxes.cpu(), scores.cpu(), expected.cpu()
        computed_cpu = nms(boxes, scores, th)
        assert (computed_cpu == expected).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda gpu available")
    @pytest.mark.skipif(
        nms_gpu is None, reason="nnDetection was not build with GPU support"
    )
    def test_nms_cuda_3d_fixed(self, th):
        boxes, scores, expected = generate_3d_fixed()

        computed_cuda = nms(boxes.cuda(), scores.cuda(), th)
        computed_cpu = nms(boxes.cpu(), scores.cpu(), th)

        assert (computed_cuda == expected.cuda()).all()
        assert (computed_cpu == expected.cpu()).all()

    def test_nms_cuda_3d_random_cpu(self, th):
        np.random.seed(0)
        boxes, scores = generate_boxes(1000, dim=3)
        boxes, scores = boxes.cpu(), scores.cpu()
        computed_cpu = nms(boxes, scores, th)
        computed_pytorch = nms_pytorch(boxes, scores, th)
        assert (computed_cpu == computed_pytorch).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda gpu available")
    @pytest.mark.skipif(
        nms_gpu is None, reason="nnDetection was not build with GPU support"
    )
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_nms_cuda_3d_random(self, th, seed):
        np.random.seed(seed)
        boxes, scores = generate_boxes(1000, dim=3, seed=seed)
        computed_cuda = nms(boxes.cuda(), scores.cuda(), th)
        computed_cpu = nms(boxes.cpu(), scores.cpu(), th)

        computed_pytorch_cuda = nms_pytorch(boxes.cuda(), scores.cuda(), th)
        computed_pytorch_cpu = nms_pytorch(boxes.cpu(), scores.cpu(), th)
        assert (computed_cuda == computed_pytorch_cuda).all()
        assert (computed_cpu == computed_pytorch_cpu).all()
        assert (computed_pytorch_cuda.cpu() == computed_pytorch_cpu).all()

    def test_batched_nms(self, th):
        boxes, scores, _ = generate_2d_fixed()
        groups = torch.tensor([0, 1, 0])
        boxes_res, scores_res, labels_res, _ = batched_nms(boxes, scores, groups, th)

        # no suppression
        assert boxes_res.allclose(boxes)
        assert scores_res.allclose(scores)
        assert labels_res.allclose(groups)
