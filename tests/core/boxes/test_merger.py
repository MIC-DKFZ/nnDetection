import pytest
import torch

from nndet.core.boxes.merging import GreedyIoUBoxMerger, VoteLabelGreedyIoUBoxMerger


@pytest.fixture
def boxes_neighboring_track():
    # IoU = 1 => track => 1 3D Box
    return [
        [3.0, 0.0, 4.0, 1.0, 0.0, 1.0],
        [4.0, 0.0, 5.0, 1.0, 0.0, 1.0],
        [5.0, 0.0, 6.0, 1.0, 0.0, 1.0],
        [6.0, 0.0, 7.0, 1.0, 0.0, 1.0],
    ]


@pytest.fixture
def boxes_neighboring_no_track():
    # IoU > 0.5 => no track => 4 3D Boxes
    return [
        [3.0, 0.0, 4.0, 1.0, 0.0, 1.0],
        [4.0, 0.5, 5.0, 1.5, 0.5, 1.5],
        [5.0, 1.0, 6.0, 1.5, 1.0, 1.5],
        [6.0, 1.5, 7.0, 2.0, 1.5, 2.0],
    ]


@pytest.fixture
def scores():
    return [1.0, 0.9, 0.8, 0.7]


@pytest.fixture
def labels():
    return [1.0, 1.0, 1.0, 1.0]


class TestGreedyIoUBoxMerger:
    def test_wrong_neighboring_slices(self, boxes_neighboring_track, scores, labels):
        boxes = torch.Tensor(boxes_neighboring_track)
        scores = torch.Tensor(scores)
        labels = torch.Tensor(labels)
        boxes_2d = boxes[:, [1, 4, 3, 5]]  # [N, 4]
        slices = torch.round(boxes[:, 0]).int()  # [N]

        with pytest.raises(ValueError):
            merger = GreedyIoUBoxMerger(
                boxes=boxes_2d,
                slices=slices,
                scores=scores,
                labels=labels,
                iou_th=0.5,
                neighbor_slices=0,
            )

    def test_wrong_iou_th(self, boxes_neighboring_track, scores, labels):
        boxes = torch.Tensor(boxes_neighboring_track)
        scores = torch.Tensor(scores)
        labels = torch.Tensor(labels)
        boxes_2d = boxes[:, [1, 4, 3, 5]]  # [N, 4]
        slices = torch.round(boxes[:, 0]).int()  # [N]

        with pytest.raises(ValueError):
            merger = GreedyIoUBoxMerger(
                boxes=boxes_2d,
                slices=slices,
                scores=scores,
                labels=labels,
                iou_th=1.1,
                neighbor_slices=1,
            )

    def test_wrong_type(self, boxes_neighboring_track, scores, labels):
        boxes = torch.Tensor(boxes_neighboring_track)
        scores = torch.Tensor(scores)
        labels = torch.Tensor(labels)
        boxes_2d = boxes[:, [1, 4, 3, 5]]  # [N, 4]
        slices = torch.round(boxes[:, 0]).int()  # [N]

        with pytest.raises(ValueError):
            merger = GreedyIoUBoxMerger(
                boxes=[boxes_2d],
                slices=slices,
                scores=scores,
                labels=labels,
                iou_th=0.5,
                neighbor_slices=0,
            )

        with pytest.raises(ValueError):
            merger = GreedyIoUBoxMerger(
                boxes=boxes_2d,
                slices=slices,
                scores=[scores],
                labels=labels,
                iou_th=0.5,
                neighbor_slices=0,
            )

        with pytest.raises(ValueError):
            merger = GreedyIoUBoxMerger(
                boxes=boxes_2d,
                slices=slices,
                scores=scores,
                labels=[labels],
                iou_th=0.5,
                neighbor_slices=0,
            )

    def test_empty_boxes(self):
        boxes = torch.Tensor([[]]).view(-1, 4)
        scores = torch.Tensor([])
        labels = torch.Tensor([])
        slices = torch.Tensor([])

        merger = GreedyIoUBoxMerger(
            boxes=boxes,
            slices=slices,
            scores=scores,
            labels=labels,
            iou_th=0.5,
        )
        boxes_3d, scores_3d, labels_3d = merger.merge()
        assert boxes_3d.numel() == 0
        assert all([a == b for a, b in zip(boxes_3d.shape, (0, 6))])
        assert scores_3d.numel() == 0
        assert labels_3d.numel() == 0

    def test_neighboring_track(self, boxes_neighboring_track, scores, labels):
        """
        Test tracking of direct neighbors
        """
        boxes = torch.Tensor(boxes_neighboring_track)
        scores = torch.Tensor(scores)
        labels = torch.Tensor(labels)
        boxes_2d = boxes[:, [1, 4, 3, 5]]  # [N, 4]
        slices = torch.round(boxes[:, 0]).int()  # [N]
        merger = GreedyIoUBoxMerger(
            boxes=boxes_2d,
            slices=slices,
            scores=scores,
            labels=labels,
            iou_th=0.5,
        )
        boxes_3d, scores_3d, labels_3d = merger.merge()

        expected_boxes = torch.Tensor([[3.0, 0.0, 7.0, 1.0, 0.0, 1.0]])
        expected_scores = torch.Tensor([0.8])
        expected_labels = torch.Tensor([1.0])
        assert boxes_3d.allclose(expected_boxes)
        assert scores_3d.allclose(expected_scores)
        assert labels_3d.allclose(expected_labels)

    def test_neighboring_no_track(self, boxes_neighboring_no_track, scores, labels):
        """
        Test tracking of direct neighbors with insufficient IoU
        """
        boxes = torch.Tensor(boxes_neighboring_no_track)
        scores = torch.Tensor(scores)
        labels = torch.Tensor(labels)
        boxes_2d = boxes[:, [1, 4, 3, 5]]  # [N, 4]
        slices = torch.round(boxes[:, 0]).int()  # [N]
        merger = GreedyIoUBoxMerger(
            boxes=boxes_2d,
            slices=slices,
            scores=scores,
            labels=labels,
            iou_th=0.5,
        )
        boxes_3d, scores_3d, labels_3d = merger.merge()

        assert boxes_3d.allclose(boxes)
        assert scores_3d.allclose(scores)
        assert labels_3d.allclose(labels)

    def test_neighboring_no_track_label(self, boxes_neighboring_track, scores, labels):
        """
        Test tracking of direct neighbors with wrong label inside
        """
        labels[1] = 0.0
        boxes = torch.Tensor(boxes_neighboring_track)
        scores = torch.Tensor(scores)
        labels = torch.Tensor(labels)
        boxes_2d = boxes[:, [1, 4, 3, 5]]  # [N, 4]
        slices = torch.round(boxes[:, 0]).int()  # [N]
        merger = GreedyIoUBoxMerger(
            boxes=boxes_2d,
            slices=slices,
            scores=scores,
            labels=labels,
            iou_th=0.5,
        )
        boxes_3d, scores_3d, labels_3d = merger.merge()

        expected_boxes = torch.Tensor(
            [
                [3.0, 0.0, 4.0, 1.0, 0.0, 1.0],
                [4.0, 0.0, 5.0, 1.0, 0.0, 1.0],
                [5.0, 0.0, 7.0, 1.0, 0.0, 1.0],
            ]
        )
        expected_scores = torch.Tensor([1.0, 0.9, 0.7])
        expected_labels = torch.Tensor([1.0, 0.0, 1.0])
        assert boxes_3d.allclose(expected_boxes)
        assert scores_3d.allclose(expected_scores)
        assert labels_3d.allclose(expected_labels)

    def test_2x_neighboring_track(self, boxes_neighboring_track, scores, labels):
        """
        Track across multiple neighbors
        """
        boxes_neighboring_track.pop(1)
        scores.pop(1)
        labels.pop(1)

        boxes = torch.Tensor(boxes_neighboring_track)
        scores = torch.Tensor(scores)
        labels = torch.Tensor(labels)
        boxes_2d = boxes[:, [1, 4, 3, 5]]  # [N, 4]
        slices = torch.round(boxes[:, 0]).int()  # [N]
        merger = GreedyIoUBoxMerger(
            boxes=boxes_2d,
            slices=slices,
            scores=scores,
            labels=labels,
            iou_th=0.5,
            neighbor_slices=2,
        )
        boxes_3d, scores_3d, labels_3d = merger.merge()

        expected_boxes = torch.Tensor([[3.0, 0.0, 7.0, 1.0, 0.0, 1.0]])
        expected_scores = torch.Tensor([0.8])
        expected_labels = torch.Tensor([1.0])
        assert boxes_3d.allclose(expected_boxes)
        assert scores_3d.allclose(expected_scores)
        assert labels_3d.allclose(expected_labels)

    def test_2x_neighboring_track_label(self, boxes_neighboring_track, scores, labels):
        """
        Track across multiple neighbors
        """
        labels[1] = 0.0
        boxes = torch.Tensor(boxes_neighboring_track)
        scores = torch.Tensor(scores)
        labels = torch.Tensor(labels)
        boxes_2d = boxes[:, [1, 4, 3, 5]]  # [N, 4]
        slices = torch.round(boxes[:, 0]).int()  # [N]
        merger = GreedyIoUBoxMerger(
            boxes=boxes_2d,
            slices=slices,
            scores=scores,
            labels=labels,
            iou_th=0.5,
            neighbor_slices=2,
        )
        boxes_3d, scores_3d, labels_3d = merger.merge()

        expected_boxes = torch.Tensor(
            [
                [3.0, 0.0, 7.0, 1.0, 0.0, 1.0],
                [4.0, 0.0, 5.0, 1.0, 0.0, 1.0],
            ]
        )
        expected_scores = torch.Tensor([0.8, 0.9])
        expected_labels = torch.Tensor([1.0, 0.0])
        assert boxes_3d.allclose(expected_boxes)
        assert scores_3d.allclose(expected_scores)
        assert labels_3d.allclose(expected_labels)

    def test_2x_neighboring_no_track(self, boxes_neighboring_track, scores, labels):
        """
        no tracking across neighboring slices
        """
        boxes_neighboring_track.pop(1)
        scores.pop(1)
        labels.pop(1)

        boxes = torch.Tensor(boxes_neighboring_track)
        scores = torch.Tensor(scores)
        labels = torch.Tensor(labels)
        boxes_2d = boxes[:, [1, 4, 3, 5]]  # [N, 4]
        slices = torch.round(boxes[:, 0]).int()  # [N]
        merger = GreedyIoUBoxMerger(
            boxes=boxes_2d,
            slices=slices,
            scores=scores,
            labels=labels,
            iou_th=0.5,
            neighbor_slices=1,
        )
        boxes_3d, scores_3d, labels_3d = merger.merge()

        expected_boxes = torch.Tensor(
            [[3.0, 0.0, 4.0, 1.0, 0.0, 1.0], [5.0, 0.0, 7.0, 1.0, 0.0, 1.0]]
        )
        expected_scores = torch.Tensor([1.0, 0.7])
        expected_labels = torch.Tensor([1.0, 1.0])
        assert boxes_3d.allclose(expected_boxes)
        assert scores_3d.allclose(expected_scores)
        assert labels_3d.allclose(expected_labels)


class TestVoteLabelGreedyIoUBoxMerger:
    def test_neighboring_track_vote_label(
        self, boxes_neighboring_track, scores, labels
    ):
        """
        Test tracking of direct neighbors with wrong label inside
        """
        labels[1] = 0.0
        boxes = torch.Tensor(boxes_neighboring_track)
        scores = torch.Tensor(scores)
        labels = torch.Tensor(labels)
        boxes_2d = boxes[:, [1, 4, 3, 5]]  # [N, 4]
        slices = torch.round(boxes[:, 0]).int()  # [N]

        merger = VoteLabelGreedyIoUBoxMerger(
            boxes=boxes_2d,
            slices=slices,
            scores=scores,
            labels=labels,
            iou_th=0.5,
        )
        boxes_3d, scores_3d, labels_3d = merger.merge()

        expected_boxes = torch.Tensor([[3.0, 0.0, 7.0, 1.0, 0.0, 1.0]])
        expected_scores = torch.Tensor([0.8])
        expected_labels = torch.Tensor([1.0])
        assert boxes_3d.allclose(expected_boxes)
        assert scores_3d.allclose(expected_scores)
        assert labels_3d.allclose(expected_labels)

    def test_neighboring_track_merging(self, boxes_neighboring_track, scores, labels):
        """
        Test tracking of direct neighbors with wrong label inside
        """
        labels[1] = 0.0
        boxes = torch.Tensor(boxes_neighboring_track)
        scores = torch.Tensor(scores)
        labels = torch.Tensor(labels)
        boxes_2d = boxes[:, [1, 4, 3, 5]]  # [N, 4]
        slices = torch.round(boxes[:, 0]).int()  # [N]
        slices[1] = 3
        slices[2] = 3

        merger = VoteLabelGreedyIoUBoxMerger(
            boxes=boxes_2d,
            slices=slices,
            scores=scores,
            labels=labels,
            iou_th=0.5,
        )
        boxes_3d, scores_3d, labels_3d = merger.merge()

        expected_boxes = torch.Tensor(
            [[3.0, 0.0, 4.0, 1.0, 0.0, 1.0], [6.0, 0.0, 7.0, 1.0, 0.0, 1.0]]
        )
        expected_scores = torch.Tensor([0.8, 0.7])
        expected_labels = torch.Tensor([1.0, 1.0])
        assert boxes_3d.allclose(expected_boxes)
        assert scores_3d.allclose(expected_scores)
        assert labels_3d.allclose(expected_labels)
