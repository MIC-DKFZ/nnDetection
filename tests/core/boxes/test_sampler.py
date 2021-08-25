import pytest

import torch

from nndet.core.boxes import *


def create_hard_negative_example():
    """
    Example values for hard negative example mining
    By limiting pool size to 1 the worst/"hardest" examples are mined from the example

    expected result masks should:
    positive: [0, 0, 1, 1, 0, 0]
    negative: [1, 1, 0, 0, 0, 0]
    """
    sampler = HardNegativeSampler(
        batch_size_per_image=4, positive_fraction=0.5, pool_size=1
    )
    target_labels = torch.tensor([0, 0, 1, 1, 0, 0])
    fg_probs = torch.tensor([1, 1, 1, 1, 0, 0])
    img_labels = torch.zeros_like(target_labels)
    return sampler, target_labels, img_labels, fg_probs


def test_negative_sampler():
    sampler = NegativeSampler(batch_size_per_image=4, positive_fraction=0.5)
    target_labels = [torch.tensor([0, 0, 1, 1, 0, 1])]
    fg_probs = None
    pos_mask, neg_mask = sampler(target_labels, fg_probs)
    assert target_labels[0][pos_mask[0].bool()].sum() == 2
    assert target_labels[0][neg_mask[0].bool()].sum() == 0


def test_hard_negative_sampler_select_positives():
    sampler, target_labels, img_labels, fg_probs = create_hard_negative_example()
    positive_idx = torch.where(target_labels >= 1)[0]
    pos_mask = sampler.select_positives(positive_idx, 2, img_labels, fg_probs)
    assert pos_mask.allclose(torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.uint8))


def test_hard_negative_sampler_select_negatives():
    sampler, target_labels, img_labels, fg_probs = create_hard_negative_example()
    negative_idx = torch.where(target_labels == 0)[0]
    neg_mask = sampler.select_negatives(negative_idx, 2, img_labels, fg_probs)
    assert neg_mask.allclose(torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.uint8))


def test_hard_negative_sampler():
    sampler, target_labels, _, fg_probs = create_hard_negative_example()
    pos_mask, neg_mask = sampler([target_labels], fg_probs)
    assert neg_mask[0].allclose(torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.uint8))
    assert pos_mask[0].allclose(torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.uint8))


def test_hard_negative_samler_fg_all():
    sampler = HardNegativeSamplerFgAll(pool_size=1)
    target_labels = torch.tensor([0, 0, 1, 1, 0, 1])
    fg_probs = torch.tensor([1, 1, 1, 1, 0.5, 0])
    pos_mask, neg_mask = sampler([target_labels], fg_probs)
    assert neg_mask[0].allclose(torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.uint8))
    assert pos_mask[0].allclose(torch.tensor([0, 0, 1, 1, 0, 1], dtype=torch.uint8))
