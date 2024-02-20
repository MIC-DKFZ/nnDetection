import math

import torch

from nndet.nn.transformer.layers.conditional_detr import gen_sine_embed_for_position


def test_gen_sine_embed_for_position():
    temp = 10000
    num_pos_feats = 18
    d = 2 * num_pos_feats / (2 * 3)

    pos_tensor = torch.arange(6).reshape(2, 3).unsqueeze_(0)
    pos_embed = gen_sine_embed_for_position(
        pos_tensor,
        temperature=temp,
        num_pos_feats=num_pos_feats,
    )

    p1_pos_embed = torch.stack(
        [
            # x
            torch.sin(0 * 2 * math.pi / (torch.tensor(temp) ** (2 * 0 / d))),
            torch.cos(0 * 2 * math.pi / (torch.tensor(temp) ** (2 * 0 / d))),
            torch.sin(0 * 2 * math.pi / (torch.tensor(temp) ** (2 * 1 / d))),
            torch.cos(0 * 2 * math.pi / (torch.tensor(temp) ** (2 * 1 / d))),
            torch.sin(0 * 2 * math.pi / (torch.tensor(temp) ** (2 * 2 / d))),
            torch.cos(0 * 2 * math.pi / (torch.tensor(temp) ** (2 * 2 / d))),
            # y
            torch.sin(1 * 2 * math.pi / (torch.tensor(temp) ** (2 * 0 / d))),
            torch.cos(1 * 2 * math.pi / (torch.tensor(temp) ** (2 * 0 / d))),
            torch.sin(1 * 2 * math.pi / (torch.tensor(temp) ** (2 * 1 / d))),
            torch.cos(1 * 2 * math.pi / (torch.tensor(temp) ** (2 * 1 / d))),
            torch.sin(1 * 2 * math.pi / (torch.tensor(temp) ** (2 * 2 / d))),
            torch.cos(1 * 2 * math.pi / (torch.tensor(temp) ** (2 * 2 / d))),
            # z
            torch.sin(2 * 2 * math.pi * 1 / (torch.tensor(temp) ** (2 * 0 / d))),
            torch.cos(2 * 2 * math.pi * 1 / (torch.tensor(temp) ** (2 * 0 / d))),
            torch.sin(2 * 2 * math.pi * 1 / (torch.tensor(temp) ** (2 * 1 / d))),
            torch.cos(2 * 2 * math.pi * 1 / (torch.tensor(temp) ** (2 * 1 / d))),
            torch.sin(2 * 2 * math.pi * 1 / (torch.tensor(temp) ** (2 * 2 / d))),
            torch.cos(2 * 2 * math.pi * 1 / (torch.tensor(temp) ** (2 * 2 / d))),
        ]
    )
    p2_pos_embed = torch.stack(
        [
            # x
            torch.sin(3 * 2 * math.pi * 1 / (torch.tensor(temp) ** (2 * 0 / d))),
            torch.cos(3 * 2 * math.pi * 1 / (torch.tensor(temp) ** (2 * 0 / d))),
            torch.sin(3 * 2 * math.pi * 1 / (torch.tensor(temp) ** (2 * 1 / d))),
            torch.cos(3 * 2 * math.pi * 1 / (torch.tensor(temp) ** (2 * 1 / d))),
            torch.sin(3 * 2 * math.pi * 1 / (torch.tensor(temp) ** (2 * 2 / d))),
            torch.cos(3 * 2 * math.pi * 1 / (torch.tensor(temp) ** (2 * 2 / d))),
            # y
            torch.sin(4 * 2 * math.pi * 1 / (torch.tensor(temp) ** (2 * 0 / d))),
            torch.cos(4 * 2 * math.pi * 1 / (torch.tensor(temp) ** (2 * 0 / d))),
            torch.sin(4 * 2 * math.pi * 1 / (torch.tensor(temp) ** (2 * 1 / d))),
            torch.cos(4 * 2 * math.pi * 1 / (torch.tensor(temp) ** (2 * 1 / d))),
            torch.sin(4 * 2 * math.pi * 1 / (torch.tensor(temp) ** (2 * 2 / d))),
            torch.cos(4 * 2 * math.pi * 1 / (torch.tensor(temp) ** (2 * 2 / d))),
            # z
            torch.sin(5 * 2 * math.pi * 1 / (torch.tensor(temp) ** (2 * 0 / d))),
            torch.cos(5 * 2 * math.pi * 1 / (torch.tensor(temp) ** (2 * 0 / d))),
            torch.sin(5 * 2 * math.pi * 1 / (torch.tensor(temp) ** (2 * 1 / d))),
            torch.cos(5 * 2 * math.pi * 1 / (torch.tensor(temp) ** (2 * 1 / d))),
            torch.sin(5 * 2 * math.pi * 1 / (torch.tensor(temp) ** (2 * 2 / d))),
            torch.cos(5 * 2 * math.pi * 1 / (torch.tensor(temp) ** (2 * 2 / d))),
        ]
    )
    expected_pos_embed = torch.stack([torch.tensor(p1_pos_embed), torch.tensor(p2_pos_embed)], dim=0)

    assert pos_embed.shape == (1, 2, num_pos_feats)
    assert torch.allclose(pos_embed[0], expected_pos_embed, atol=2e-6)
