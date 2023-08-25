###
# TODO: test. cat_and_index ops_torch
# TODO: test detach all
###

# TODO: test assign and sample
# TODO: add_gt_to_proposals
# TODO assign_targets_to_anchors empty check

# TODO: extend steps to batch size 2

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
from nndet.core.rois.pooler.roi_align import RoIAlignNaiveAssign, roi_align_3d
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


class ZeroBCEAgnosticMasker(BCEAgnosticMasker):
    """
    All foreground masks
    """

    def forward(self, features: Tensor) -> Tensor:
        o, i = super().forward(features)
        return torch.ones_like(o), torch.ones_like(i)


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
            [0, 0, 2, 2, 0, 2],
            [2, 2, 4, 4, 2, 4],
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

    return {
        "box_head": roi_box_head,
        "box_pooler": roi_box_pooler,
        "box_post": roi_box_post,
        "matcher": roi_matcher,
        "sampler": roi_sampler,
        "num_classes": num_classes,
    }


def example_modules_mask():
    conv = Generator(ConvGroupRelu, dim=3)
    roi_feature_size = (3, 3, 3)
    roi_mask_size = (6, 6, 6)
    in_channels = 16
    internal_channels = 32
    num_classes = 1

    roi_masker = ZeroBCEAgnosticMasker(
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

    modules = example_modules()
    modules["mask_head"] = roi_masker
    modules["mask_pooler"] = roi_mask_pooler
    modules["mask_post"] = roi_mask_post
    return modules


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


@pytest.fixture
def roi_module_mask():
    return RoIModule(
        **example_modules_mask(),
        decoder_levels=(1, 2, 3),
        gt_to_proposals=True,
        inference_prob_rpn=False,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda gpu available")
@pytest.mark.skipif(
    roi_align_3d is None,
    reason="nnDetection was not build with GPU support",
)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda gpu available")
@pytest.mark.skipif(
    roi_align_3d is None,
    reason="nnDetection was not build with GPU support",
)
@pytest.mark.parametrize("device", DEVICES)
def test_inference_step_boxes(base_roi_module: BaseRoIModule, device: torch.device):
    # inputs
    roi_module = base_roi_module.to(device=device)
    image_size = (32, 32, 32)
    features = [torch.zeros([1, 16] + [4 * (2**i)] * 3, dtype=torch.float32, device=device) for i in range(4)][::-1]

    boxes = example_boxes(device=device)
    proposal_boxes = [torch.clone(boxes)]
    proposal_scores = [torch.tensor([1] * boxes.shape[0], dtype=torch.float32, device=device)]

    # computation
    preds = roi_module._inference_step_boxes(
        features=features,
        image_size=image_size,
        proposal_boxes=proposal_boxes,
        proposal_scores=proposal_scores,
        stage=0,
    )

    # predictions
    expected_boxes = torch.clone(boxes)
    expected_scores = torch.sigmoid(torch.tensor([1, 1], dtype=torch.long, device=device))
    expected_labels = torch.tensor([0, 0], dtype=torch.long, device=device)

    assert len(preds["pred_boxes"]) == 1
    assert len(preds["pred_scores"]) == 1
    assert len(preds["pred_labels"]) == 1
    assert preds["pred_boxes"][0].shape == expected_boxes.shape
    assert boxes1_in_boxes2(preds["pred_boxes"][0], expected_boxes)
    assert torch.allclose(preds["pred_scores"][0], expected_scores)
    assert torch.allclose(preds["pred_labels"][0], expected_labels)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda gpu available")
@pytest.mark.skipif(
    roi_align_3d is None,
    reason="nnDetection was not build with GPU support",
)
@pytest.mark.parametrize("device", DEVICES)
def test_train_step_masks(roi_module_mask: BaseRoIModule, device: torch.device):
    # inputs
    roi_module = roi_module_mask.to(device=device)
    image_size = (32, 32, 32)
    features = [torch.zeros([1, 16] + [4 * (2**i)] * 3, dtype=torch.float32, device=device) for i in range(4)][::-1]

    boxes = example_boxes(device=device)
    proposal_boxes = [torch.clone(boxes)]
    matched_gt_idx = [torch.tensor([0, 1], dtype=torch.long, device=device)]
    matched_gt_labels = [torch.tensor([1] * boxes.shape[0], dtype=torch.long, device=device)]
    mask = torch.zeros(2, *image_size, dtype=torch.float32, device=device)
    mask[0, 1, 1, 1] = 1
    mask[1, 3, 3, 3] = 1
    gt_binary_masks = [mask]

    # computation
    losses, preds = roi_module._train_step_masks(
        features=features,
        matched_gt_labels=matched_gt_labels,
        matched_gt_idx=matched_gt_idx,
        proposal_boxes=proposal_boxes,
        gt_binary_masks=gt_binary_masks,
        image_size=image_size,
        stage=0,
        predict=True,
    )

    # TODO: finalise test once roi_align design is finalised

    # predictions
    assert preds is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda gpu available")
@pytest.mark.skipif(
    roi_align_3d is None,
    reason="nnDetection was not build with GPU support",
)
@pytest.mark.parametrize("device", DEVICES)
def test_inference_step_masks(roi_module_mask: BaseRoIModule, device: torch.device):
    # inputs
    roi_module = roi_module_mask.to(device=device)
    image_size = (32, 32, 32)
    features = [torch.zeros([1, 16] + [4 * (2**i)] * 3, dtype=torch.float32, device=device) for i in range(4)][::-1]

    boxes = example_boxes(device=device)
    pred_boxes = [boxes]
    pred_scores = [torch.tensor([1] * boxes.shape[0], dtype=torch.float32, device=device)]
    pred_labels = [torch.tensor([0] * boxes.shape[0], dtype=torch.long, device=device)]

    # computation
    preds = roi_module._inference_step_masks(
        features=features,
        image_size=image_size,
        pred_boxes=pred_boxes,
        pred_probs=pred_scores,
        pred_labels=pred_labels,
        stage=0,
    )

    # checks
    expected_masks = torch.sigmoid(torch.ones((2, 6, 6, 6), dtype=torch.float32, device=device))
    expected_scores = torch.tensor([1, 1], dtype=torch.float32, device=device)
    expected_labels = torch.tensor([0, 0], dtype=torch.long, device=device)

    assert len(preds["pred_masks"]) == 1
    assert len(preds["pred_mask_scores"]) == 1
    assert len(preds["pred_mask_labels"]) == 1
    assert preds["pred_image_spatial_size"] == (32, 32, 32)

    assert torch.allclose(preds["pred_masks"][0], expected_masks)
    assert torch.allclose(preds["pred_mask_scores"][0], expected_scores)
    assert torch.allclose(preds["pred_mask_labels"][0], expected_labels)


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


@pytest.mark.parametrize("device", DEVICES)
def test_inference_step_empty(roi_module: BaseRoIModule, device: torch.device):
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
    preds = roi_module.inference_step(
        images=images,
        features=features,
        proposals=proposals,
    )

    # predictions
    assert len(preds["pred_boxes"]) == 1
    assert preds["pred_boxes"][0].numel() == 0
    assert preds["pred_boxes"][0].ndim == 2
    assert preds["pred_boxes"][0].shape[1] == 6

    assert len(preds["pred_scores"]) == 1
    assert preds["pred_scores"][0].numel() == 0
    assert preds["pred_scores"][0].ndim == 1

    assert len(preds["pred_labels"]) == 1
    assert preds["pred_labels"][0].numel() == 0
    assert preds["pred_labels"][0].ndim == 1


def test_add_gt_to_proposal(base_roi_module: BaseRoIModule):
    proposal_boxes = [
        torch.zeros(2, 6),
        torch.zeros(3, 6),
    ]
    proposal_scores = [torch.zeros(2), torch.zeros(3)]
    proposal_labels = [torch.zeros(2), torch.zeros(3)]

    target_boxes = [torch.ones(2, 6), torch.ones(3, 6)]
    target_roi_classes = [torch.ones(2), torch.ones(3)]

    new_proposals = base_roi_module.add_gt_to_proposals(
        proposals={
            "pred_boxes": proposal_boxes,
            "pred_scores": proposal_scores,
            "pred_labels": proposal_labels,
        },
        targets={
            "target_boxes": target_boxes,
            "target_roi_classes": target_roi_classes,
        },
    )

    expected_box_shapes = [(4, 6), (6, 6)]
    expected_score_shapes = [(4,), (6,)]
    expected_label_shapes = [(4,), (6,)]

    assert len(expected_box_shapes) == len(new_proposals["pred_boxes"])
    assert len(expected_score_shapes) == len(new_proposals["pred_scores"])
    assert len(expected_label_shapes) == len(new_proposals["pred_labels"])

    for exp, new in zip(expected_box_shapes, new_proposals["pred_boxes"]):
        assert exp == tuple(new.shape)
    for exp, new in zip(expected_score_shapes, new_proposals["pred_scores"]):
        assert exp == tuple(new.shape)
    for exp, new in zip(expected_label_shapes, new_proposals["pred_labels"]):
        assert exp == tuple(new.shape)


def test_assign_and_sample():
    pass
