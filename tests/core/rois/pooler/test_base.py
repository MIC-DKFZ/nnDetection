from unittest.mock import Mock

import pytest
import torch

from nndet.core.rois.pooler.base import RoIPooler


@pytest.fixture
def pooler():
    _pooler = RoIPooler(
        feature_output_size=(7, 7, 7),
        mask_output_size=(8, 8, 8),
        feature_pool_kwargs={},
        mask_pool_kwargs={},
    )
    return _pooler


class TestRoIPooler:
    def test_forward_assert_dims(self, pooler):
        with pytest.raises(AssertionError):
            features = [torch.zeros(1, 1, 28, 28)]
            boxes = torch.zeros(4, 6)
            batch_idx = torch.zeros(4)
            pooler(features, boxes, batch_idx, (128, 128, 128))

    def test_forward_single_feature_map(self, pooler):
        mock = Mock(return_value=torch.zeros(4, 1, 7, 7, 7))
        mock2 = Mock()
        pooler._pool_features = mock
        pooler._find_pyramid_level = mock2

        features = [torch.zeros(1, 1, 32, 32, 32)]
        boxes = torch.zeros(4, 6)
        batch_idx = torch.zeros(4)
        res = pooler(features, boxes, batch_idx, (128, 128, 128))

        mock.assert_called_once()
        assert tuple(res.shape) == (4, 1, 7, 7, 7)
        assert res.allclose(torch.zeros(4, 1, 7, 7, 7))
        mock2.assert_not_called()

    def test_fordward_multiple_feature_maps(self, pooler):
        # return tensor filled with the number of passed proposals
        def dummy_return(fmap, proposal_boxes_batch_idx, spatial_scale):
            n = proposal_boxes_batch_idx.shape[0]
            return torch.ones(n, 1, 7, 7, 7) * n

        mock = Mock(side_effect=dummy_return)
        mock2 = Mock(return_value=torch.tensor([2, 1, 2, 2]))
        pooler._pool_features = mock
        pooler._find_pyramid_level = mock2

        features = [
            torch.zeros(1, 1, 16, 16, 16),
            torch.zeros(1, 1, 32, 32, 32),
            torch.zeros(1, 1, 64, 64, 64),
        ]
        boxes = torch.zeros(4, 6)
        batch_idx = torch.zeros(4)
        res = pooler(features, boxes, batch_idx, (128, 128, 128))

        mock.assert_called()
        assert mock.call_count == 2
        assert tuple(res.shape) == (4, 1, 7, 7, 7)

        expected_result = torch.ones(4, 1, 7, 7, 7)
        expected_result[0] = expected_result[0] * 3
        expected_result[2:] = expected_result[2:] * 3
        assert res.allclose(expected_result)
        mock2.assert_called_once()
