from typing import Callable, Tuple

import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf.omegaconf import OmegaConf

from nndet.ptmodule.rcnn.base import BoxCascadeRCNN, BoxRCNN
from nndet.ptmodule.rcnn.dev.cascademc001 import CascadeMaskURCNNC001
from nndet.ptmodule.rcnn.dev.fc001 import FasterRCNNC001
from nndet.ptmodule.rcnn.dev.mc001 import MaskRCNNC001

# base modules
from nndet.ptmodule.retinanet.base import RetinaNetModule

# specific modules
from nndet.ptmodule.retinanet.dev import RetinaNetC001, RetinaNetC001Focal
from nndet.ptmodule.retinaunet.base import RetinaUNetModule
from nndet.ptmodule.retinaunet.v001 import RetinaUNetCV001Focal, RetinaUNetV001


@pytest.fixture
def example_plan():
    plan = {
        "patch_size": (32, 32, 32),
        "num_modalities": 1,
        "architecture": {
            "dim": 3,
            "in_channels": 1,
            "classifier_classes": 2,
            "seg_classes": 2,
            "start_channels": 4,
            "fpn_channels": 16,
            "head_channels": 16,
            "decoder_levels": (2, 3, 4),
            "conv_kernels": [3, 3, 3, 3, 3],
            "strides": [2, 2, 2, 2],
        },
        "anchors": {
            "width": [
                [
                    3,
                ],
                [
                    3,
                ],
                [
                    3,
                ],
            ],
            "height": [
                [
                    3,
                ],
                [
                    3,
                ],
                [
                    3,
                ],
            ],
            "depth": [
                [
                    3,
                ],
                [
                    3,
                ],
                [
                    3,
                ],
            ],
        },
    }
    return plan


def example_batch(in_channels, patch_size, device):
    data = torch.zeros(in_channels, *patch_size, dtype=torch.float, device=device)

    mask = torch.zeros(1, *patch_size, dtype=torch.float, device=device)
    mask[0, 4:8, 4:8, 4:8] = 1.0
    mask[0, 10:14, 4:8, 4:8] = 3.0

    mapping = {"1": "0", "2": "0", "3": "1", "4": "0"}

    # batch size = 1
    return {
        "data": data[None],
        "target": mask[None],
        "instance_mapping": [mapping],
    }


CASES = [
    (RetinaNetC001, "v001"),
    (RetinaNetC001Focal, "c014_focal"),
    (RetinaUNetV001, "v001"),
    (RetinaUNetCV001Focal, "c014_focal"),
    (FasterRCNNC001, "frcnn_c001"),
    (MaskRCNNC001, "mrcnn_c001"),
    (CascadeMaskURCNNC001, "cascmrcnn_c001"),
]


@pytest.mark.parametrize("step", ["train", "val"])
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        # TODO: gpu tests
        # pytest.param(
        #     "cuda",
        #     marks=pytest.mark.skipif(
        #         not torch.cuda.is_available(), reason="No cuda gpu available"
        #     ),
        # ),
    ],
)
def test_step_smoke(example_plan, step: str, case: Tuple[Callable, str], device):
    # Skip RCNN CPU tests ...
    if "rcnn" in case[1] and device == "cpu":
        return

    module_cls, ov = case

    GlobalHydra.instance().clear()  # clear hydra
    initialize_config_module(config_module="nndet.conf")
    cfg = compose("config.yaml", overrides=[f"train={ov}"])
    OmegaConf.set_struct(cfg, False)
    cfg["task"] = "Task000_TEST"

    module = module_cls(
        model_cfg=OmegaConf.to_container(cfg["model_cfg"], resolve=True),
        trainer_cfg=OmegaConf.to_container(cfg["trainer_cfg"], resolve=True),
        plan=example_plan,
    )
    module.to(device)

    batch = example_batch(
        in_channels=example_plan["architecture"]["in_channels"],
        patch_size=example_plan["patch_size"],
        device=device,
    )
    if step == "train":
        module.training_step(batch=batch, batch_idx=0)
    elif step == "val":
        module.validation_step(batch=batch, batch_idx=0)
    else:
        raise ValueError(f"Step {step} is unknown")
    module.cpu()
