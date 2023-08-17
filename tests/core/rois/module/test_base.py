###
# TODO: test. cat_and_index ops_torch
# TODO: test detach all
###


# TODO: cover no ground truth test case


# TODO: integration test box and mask train step
# TODO: integration test box and mask inference step
# TODO: integration test postprocessing

# TODO: test assign and sample
# TODO: add_gt_to_proposals
# TODO assign_targets_to_anchors empty check

import pytest
import torch
from torch import Tensor

from nndet.core.boxes.coder import BoxCoderND
from nndet.core.boxes.matcher import IoUMatcher
from nndet.core.boxes.sampler import HardNegativeSampler
from nndet.core.post.box import CrossLevelBoxPostprocessing
from nndet.core.post.mask import NoMaskPostprocessing
from nndet.core.rois.module.base import BaseRoIModule
from nndet.core.rois.module.single import RoIModule
from nndet.core.rois.pooler.roi_align import RoIAlignNaiveAssign
from nndet.losses.classification.ce import BCELoss
from nndet.nn.heads.classifier.roi import BCEConvRoIClassifier
from nndet.nn.heads.comb.roi import RoIBoxHead
from nndet.nn.heads.masker.roi import BCEAgnosticMasker
from nndet.nn.heads.regressor.roi import L1ConvRoIAgnosticRegressor
from nndet.nn.layers.conv import ConvGroupRelu
from nndet.nn.layers.wrapper import Generator

DEVICES = ["cuda"]


class ZeroL1Regressor(L1ConvRoIAgnosticRegressor):
    """
    No deltas predicted

    Only use with 0 inputs!
    """

    def forward(self, features: Tensor) -> Tensor:
        assert torch.allclose(features, torch.zeros_like(features))
        return torch.zeros_like(super().forward(features))


class FGAllBCEClassifier(BCEConvRoIClassifier):
    """
    Predict all Foreground with 1

    Only use with 0 inputs!
    """

    def forward(self, features: Tensor) -> Tensor:
        assert torch.allclose(features, torch.zeros_like(features))
        return torch.ones_like(super().forward(features))


def boxes1_in_boxes2(boxes1: torch.Tensor, boxes2: torch.Tensor) -> bool:
    _boxes1 = boxes1.unsqueeze(dim=1)  # [N, 1, x]
    _boxes2 = boxes2.unsqueeze(dim=0)  # [1, M, x]
    closeness = torch.isclose(_boxes1, _boxes2)  # N, M, x
    assert closeness.ndim == 3
    # all => all box coordinates need to be close of a single box
    # any => there is at least one box of boxes 1 which is in boxes2
    # all => all boxes in boxes1 have a box in boxes 2
    return closeness.all(dim=2).any(dim=1).all(dim=0)


def example_boxes(device):
    return torch.tensor(
        [
            [0, 0, 1, 1, 0, 1],
            [1, 1, 2, 2, 1, 2],
        ],
        dtype=torch.float32,
        device=device,
    )


def example_modules():
    conv = Generator(ConvGroupRelu, dim=3)
    roi_feature_size = (3, 3, 3)
    roi_mask_size = (7, 7, 7)
    in_channels = 16
    internal_channels = 32
    num_classes = 1

    classifier = FGAllBCEClassifier(
        conv=conv,
        input_size=roi_feature_size,
        in_channels=in_channels,
        internal_channels=internal_channels,
        num_classes=num_classes,
        num_convs=2,
        reduction="sum",
        prior_prob=0.1,
    )
    regressor = ZeroL1Regressor(
        conv=conv,
        input_size=roi_feature_size,
        in_channels=in_channels,
        internal_channels=internal_channels,
        num_classes=num_classes,
        num_convs=2,
        reduction="sum",
    )
    coder = BoxCoderND(weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))

    roi_box_head = RoIBoxHead(
        classifier=classifier,
        regressor=regressor,
        coder=coder,
    )
    roi_box_pooler = RoIAlignNaiveAssign(
        feature_output_size=roi_feature_size,
        mask_output_size=roi_feature_size,
    )
    roi_box_post = CrossLevelBoxPostprocessing(num_classes=num_classes, nms_thresh=0.2)
    roi_matcher = IoUMatcher(low_threshold=0.1, high_threshold=0.2, allow_low_quality_matches=False)
    roi_sampler = HardNegativeSampler(batch_size_per_image=32, positive_fraction=0.5)

    roi_masker = BCEAgnosticMasker(
        conv=conv,
        in_channels=in_channels,
        internal_channels=internal_channels,
        num_classes=num_classes,
        num_convs=2,
    )
    roi_mask_pooler = RoIAlignNaiveAssign(
        feature_output_size=roi_feature_size,
        mask_output_size=roi_mask_size,
    )
    roi_mask_post = NoMaskPostprocessing(num_classes=num_classes)
    return {
        "box_head": roi_box_head,
        "box_pooler": roi_box_pooler,
        "box_post": roi_box_post,
        "matcher": roi_matcher,
        "sampler": roi_sampler,
        "num_classes": num_classes,
        "mask_head": roi_masker,
        "mask_pooler": roi_mask_pooler,
        "mask_post": roi_mask_post,
    }


@pytest.fixture
def base_roi_module():
    return BaseRoIModule(
        **example_modules(),
        decoder_levels=(1, 2, 3),
        gt_to_proposals=True,
        inference_prob_rpn=False,
    )


@pytest.fixture
def roi_module():
    return RoIModule(
        **example_modules(),
        decoder_levels=(1, 2, 3),
        gt_to_proposals=True,
        inference_prob_rpn=False,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda gpu available")
@pytest.mark.parametrize("device", DEVICES)
def test_train_step_boxes(base_roi_module: BaseRoIModule, device: torch.device):
    # inputs
    roi_module = base_roi_module.to(device=device)
    image_size = (32, 32, 32)
    features = [torch.zeros([1, 16] + [4 * (2**i)] * 3, dtype=torch.float32, device=device) for i in range(4)][::-1]

    boxes = example_boxes(device=device)
    proposal_boxes = [torch.clone(boxes)]
    matched_gt_boxes = [torch.clone(boxes)]
    matched_gt_labels = [torch.tensor([1] * boxes.shape[0], dtype=torch.long, device=device)]

    # computation
    losses, preds = roi_module._train_step_boxes(
        features=features,
        image_size=image_size,
        matched_gt_boxes=matched_gt_boxes,
        matched_gt_labels=matched_gt_labels,
        proposal_boxes=proposal_boxes,
        stage=0,
        predict=True,
    )

    # expected outputs
    expected_boxes = torch.clone(boxes)
    p = torch.ones(matched_gt_labels[0].shape, dtype=torch.float32, device=device)[:, None]
    expected_cls_loss = BCELoss(reduction="sum")(p, matched_gt_labels[0].float()) / matched_gt_labels[0].numel()

    # losses
    assert torch.allclose(torch.tensor(0, dtype=torch.float32, device=device), losses["reg"])
    assert torch.allclose(expected_cls_loss, losses["cls"])

    # predictions
    assert len(preds["pred_boxes"]) == 1
    assert len(preds["pred_scores"]) == 1
    assert len(preds["pred_labels"]) == 1
    assert preds["pred_boxes"][0].shape == expected_boxes.shape
    assert boxes1_in_boxes2(preds["pred_boxes"][0], expected_boxes)
    assert torch.allclose(
        preds["pred_scores"][0],
        torch.sigmoid(torch.tensor([1, 1], dtype=torch.long, device=device)),
    )
    assert torch.allclose(preds["pred_labels"][0], torch.tensor([0, 0], dtype=torch.long, device=device))


@pytest.mark.parametrize("device", DEVICES)
def test_train_step_empty(roi_module: BaseRoIModule, device: torch.device):
    # inputs
    roi_module = roi_module.to(device=device)
    images = torch.zeros(1, 3, 32, 32, 32)
    features = [torch.zeros([1, 16] + [4 * (2**i)] * 3, dtype=torch.float32, device=device) for i in range(4)][::-1]
    proposals = {
        "pred_boxes": [torch.tensor([[]], device=device, dtype=torch.float32).view(-1, 6)],
        "pred_labels": [torch.tensor([], device=device, dtype=torch.long)],
        "pred_scores": [torch.tensor([], device=device, dtype=torch.float32)],
    }
    targets = {
        "target_boxes": [torch.tensor([[]], device=device, dtype=torch.float32).view(-1, 6)],
        "target_roi_classes": [torch.tensor([], device=device, dtype=torch.long)],
    }

    # computation
    losses = roi_module.train_step(
        images=images,
        features=features,
        proposals=proposals,
        targets=targets,
    )

    # losses
    assert not losses


def test_inference_step_boxes():
    pass


def test_train_step_masks():
    pass


def test_inference_step_masks():
    pass
