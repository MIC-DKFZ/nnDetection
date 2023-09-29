import math

import torch

from nndet.nn.layers.pos_embed.sine import PositionEmbeddingSine


def test_sine_2d_no_norm():
    inp = torch.zeros((1, 3, 2, 2))
    pos_embedding = PositionEmbeddingSine(dim=2, num_pos_feats=8)
    pos = pos_embedding(inp)

    temp = 10000
    d = 4

    expected = torch.tensor(
        [
            # k = 0
            [
                [math.sin(1 / (temp ** (2 * 0 / d))), math.sin(1 / (temp ** (2 * 0 / d)))],
                [math.sin(2 / (temp ** (2 * 0 / d))), math.sin(2 / (temp ** (2 * 0 / d)))],
            ],
            [
                [math.cos(1 / (temp ** (2 * 0 / d))), math.cos(1 / (temp ** (2 * 0 / d)))],
                [math.cos(2 / (temp ** (2 * 0 / d))), math.cos(2 / (temp ** (2 * 0 / d)))],
            ],
            # k = 1
            [
                [math.sin(1 / (temp ** (2 * 1 / d))), math.sin(1 / (temp ** (2 * 1 / d)))],
                [math.sin(2 / (temp ** (2 * 1 / d))), math.sin(2 / (temp ** (2 * 1 / d)))],
            ],
            [
                [math.cos(1 / (temp ** (2 * 1 / d))), math.cos(1 / (temp ** (2 * 1 / d)))],
                [math.cos(2 / (temp ** (2 * 1 / d))), math.cos(2 / (temp ** (2 * 1 / d)))],
            ],
            # k = 0
            [
                [math.sin(1 / (temp ** (2 * 0 / d))), math.sin(2 / (temp ** (2 * 0 / d)))],
                [math.sin(1 / (temp ** (2 * 0 / d))), math.sin(2 / (temp ** (2 * 0 / d)))],
            ],
            [
                [math.cos(1 / (temp ** (2 * 0 / d))), math.cos(2 / (temp ** (2 * 0 / d)))],
                [math.cos(1 / (temp ** (2 * 0 / d))), math.cos(2 / (temp ** (2 * 0 / d)))],
            ],
            # k = 1
            [
                [math.sin(1 / (temp ** (2 * 1 / d))), math.sin(2 / (temp ** (2 * 1 / d)))],
                [math.sin(1 / (temp ** (2 * 1 / d))), math.sin(2 / (temp ** (2 * 1 / d)))],
            ],
            [
                [math.cos(1 / (temp ** (2 * 1 / d))), math.cos(2 / (temp ** (2 * 1 / d)))],
                [math.cos(1 / (temp ** (2 * 1 / d))), math.cos(2 / (temp ** (2 * 1 / d)))],
            ],
        ]
    )

    assert tuple(pos.shape)[2:] == tuple(inp.shape)[2:]
    assert pos.shape[1] == 8
    assert pos.shape[0] == 1
    assert torch.allclose(expected, pos[0])


def test_sine_3d_no_norm():
    inp = torch.zeros((1, 3, 2, 2, 2))
    pos_embedding = PositionEmbeddingSine(dim=3, num_pos_feats=6)
    pos = pos_embedding(inp)

    temp = 10000
    d = 6

    expected = torch.tensor(
        [
            # k = 0
            [
                [
                    [math.sin(1 / (temp ** (2 * 0 / d))), math.sin(1 / (temp ** (2 * 0 / d)))],
                    [math.sin(1 / (temp ** (2 * 0 / d))), math.sin(1 / (temp ** (2 * 0 / d)))],
                ],
                [
                    [math.sin(2 / (temp ** (2 * 0 / d))), math.sin(2 / (temp ** (2 * 0 / d)))],
                    [math.sin(2 / (temp ** (2 * 0 / d))), math.sin(2 / (temp ** (2 * 0 / d)))],
                ],
            ],
            [
                [
                    [math.cos(1 / (temp ** (2 * 0 / d))), math.cos(1 / (temp ** (2 * 0 / d)))],
                    [math.cos(1 / (temp ** (2 * 0 / d))), math.cos(1 / (temp ** (2 * 0 / d)))],
                ],
                [
                    [math.cos(2 / (temp ** (2 * 0 / d))), math.cos(2 / (temp ** (2 * 0 / d)))],
                    [math.cos(2 / (temp ** (2 * 0 / d))), math.cos(2 / (temp ** (2 * 0 / d)))],
                ],
            ],
            [
                [
                    [math.sin(1 / (temp ** (2 * 0 / d))), math.sin(1 / (temp ** (2 * 0 / d)))],
                    [math.sin(2 / (temp ** (2 * 0 / d))), math.sin(2 / (temp ** (2 * 0 / d)))],
                ],
                [
                    [math.sin(1 / (temp ** (2 * 0 / d))), math.sin(1 / (temp ** (2 * 0 / d)))],
                    [math.sin(2 / (temp ** (2 * 0 / d))), math.sin(2 / (temp ** (2 * 0 / d)))],
                ],
            ],
            [
                [
                    [math.cos(1 / (temp ** (2 * 0 / d))), math.cos(1 / (temp ** (2 * 0 / d)))],
                    [math.cos(2 / (temp ** (2 * 0 / d))), math.cos(2 / (temp ** (2 * 0 / d)))],
                ],
                [
                    [math.cos(1 / (temp ** (2 * 0 / d))), math.cos(1 / (temp ** (2 * 0 / d)))],
                    [math.cos(2 / (temp ** (2 * 0 / d))), math.cos(2 / (temp ** (2 * 0 / d)))],
                ],
            ],
            [
                [
                    [math.sin(1 / (temp ** (2 * 0 / d))), math.sin(2 / (temp ** (2 * 0 / d)))],
                    [math.sin(1 / (temp ** (2 * 0 / d))), math.sin(2 / (temp ** (2 * 0 / d)))],
                ],
                [
                    [math.sin(1 / (temp ** (2 * 0 / d))), math.sin(2 / (temp ** (2 * 0 / d)))],
                    [math.sin(1 / (temp ** (2 * 0 / d))), math.sin(2 / (temp ** (2 * 0 / d)))],
                ],
            ],
            [
                [
                    [math.cos(1 / (temp ** (2 * 0 / d))), math.cos(2 / (temp ** (2 * 0 / d)))],
                    [math.cos(1 / (temp ** (2 * 0 / d))), math.cos(2 / (temp ** (2 * 0 / d)))],
                ],
                [
                    [math.cos(1 / (temp ** (2 * 0 / d))), math.cos(2 / (temp ** (2 * 0 / d)))],
                    [math.cos(1 / (temp ** (2 * 0 / d))), math.cos(2 / (temp ** (2 * 0 / d)))],
                ],
            ],
        ]
    )

    assert tuple(pos.shape)[2:] == tuple(inp.shape)[2:]
    assert pos.shape[1] == 6
    assert pos.shape[0] == 1
    assert torch.allclose(expected, pos[0])
