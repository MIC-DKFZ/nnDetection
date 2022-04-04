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
        mock = Mock()
        pooler._pool_features = mock

        features = [torch.zeros(1, 1, 32, 32, 32)]
        boxes = torch.zeros(4, 6)
        batch_idx = torch.zeros(4)
        pooler(features, boxes, batch_idx, (128, 128, 128))

        mock.assert_called_once()

    def test_fordward_multiple_feature_maps(self):
        raise NotImplementedError
