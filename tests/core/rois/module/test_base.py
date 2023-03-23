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

import pytest
import torch

from nndet.core.boxes.coder import BoxCoderND
from nndet.core.boxes.matcher import IoUMatcher
from nndet.core.boxes.sampler import HardNegativeSampler
from nndet.core.post.box import CrossLevelBoxPostprocessing
from nndet.core.post.mask import NoMaskPostprocessing
from nndet.core.rois.module.base import BaseRoIModule
from nndet.core.rois.pooler.roi_align import RoIAlignNaiveAssign
from nndet.nn.heads.classifier.roi import BCEConvRoIClassifier
from nndet.nn.heads.comb.roi import RoIBoxHead
from nndet.nn.heads.masker.roi import BCEAgnosticMasker
from nndet.nn.heads.regressor.roi import L1ConvRoIAgnosticRegressor
from nndet.nn.layers.conv import ConvGroupRelu
from nndet.nn.layers.wrapper import Generator


@pytest.fixture
def roi_module():
    conv = Generator(ConvGroupRelu, dim=3)
    roi_feature_size = (3, 3, 3)
    roi_mask_size = (7, 7, 7)
    in_channels = 16
    internal_channels = 32
    num_classes = 2

    classifier = BCEConvRoIClassifier(
        conv=conv,
        input_size=roi_feature_size,
        in_channels=in_channels,
        internal_channels=internal_channels,
        num_classes=num_classes,
        num_convs=2,
        reduction="sum",
        prior_prob=0.1,
    )
    regressor = L1ConvRoIAgnosticRegressor(
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
    return BaseRoIModule(
        box_head=roi_box_head,
        box_pooler=roi_box_pooler,
        box_post=roi_box_post,
        matcher=roi_matcher,
        sampler=roi_sampler,
        num_classes=num_classes,
        decoder_levels=(1, 2, 3),
        gt_to_proposals=True,
        mask_head=roi_masker,
        mask_pooler=roi_mask_pooler,
        mask_post=roi_mask_post,
        inference_prob_rpn=False,
    )


@pytest.fixture
def example_boxes_bs2():
    pass


@pytest.fixture
def example_boxes_bs2_empty():
    pass


def test_train_step_boxes(roi_module: BaseRoIModule):
    image_size = (32, 32, 32)
    features = [torch.zeros([4 * (2**i)] * 3) for i in range(4)][::-1]

    matched_gt_boxes = []
    matched_gt_labels = []
    proposal_boxes = []


def test_inference_step_boxes():
    pass


def test_train_step_masks():
    pass


def test_inference_step_masks():
    pass
