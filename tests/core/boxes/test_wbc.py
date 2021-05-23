import pytest
from dataclasses import dataclass

import torch

from nndet.core.boxes.wbc import batched_wbc


@dataclass
class Example():
    boxes = torch.tensor([[0., 0., 1., 1.],
                          [0., 0., 1., 1.],
                          [2., 2., 3., 3.]])
    scores = torch.tensor([1., 1., 1.])
    labels = torch.tensor([1., 1., 1.])
    weights = torch.tensor([1., 1., 1.])
    n_exp_preds = torch.tensor([2., 2., 1.])


@pytest.fixture
def example():
    return Example()


class TestWBC:
    def test_label_weighted_box_clustering_no_clustering(self, example):
        labels = torch.tensor([1., 2., 1.])
        pboxes, pscores, plabels, _ = batched_wbc(
            example.boxes, example.scores, labels, example.weights,
            0.1, example.n_exp_preds, 0.1)
        assert int(pboxes.shape[0]) == 3

        expected_scores = torch.tensor([0.5, 1., 0.5])
        assert expected_scores.allclose(pscores)

        expected_labels = torch.tensor([1., 1., 2.])
        assert expected_labels.allclose(plabels)

    def test_label_weighted_box_clustering(self, example):
        pboxes, pscores, plabels, _ = batched_wbc(
            example.boxes, example.scores, example.labels, example.weights,
            0.1, example.n_exp_preds, 0.1)
        assert int(pboxes.shape[0]) == 2

        expected_scores = torch.tensor([1., 1.])
        assert expected_scores.allclose(pscores)

        expected_labels = torch.tensor([1., 1.])
        assert expected_labels.allclose(plabels)
