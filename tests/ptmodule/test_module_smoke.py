from typing import Callable, Tuple

import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf.omegaconf import OmegaConf

from nndet.ptmodule.detr.dev.c002 import BoxDETRC002

# specific modules
from nndet.ptmodule.retinanet.rnv002 import (
    RetinaNetFocalResV002,
    RetinaNetFocalV002,
    RetinaNetHNMV002,
)

# base modules
from nndet.ptmodule.retinaunet.runv001 import RetinaUNetV001
from nndet.ptmodule.retinaunet.runv002 import (
    RetinaUNetFocalResV002,
    RetinaUNetFocalV002,
    RetinaUNetHNMV002,
)

# from nndet.ptmodule.frcnn.dev.fc001 import FasterRCNNC001
# from nndet.ptmodule.mrcnn.dev.cmc001 import CascadeMaskURCNNC001
# from nndet.ptmodule.mrcnn.dev.mc001 import MaskRCNNC001, MaskURCNNC001


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
            "max_channels": 320,
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


def example_empty_batch(in_channels, patch_size, device):
    data = torch.zeros(in_channels, *patch_size, dtype=torch.float, device=device)
    mask = torch.zeros(1, *patch_size, dtype=torch.float, device=device)
    return {
        "data": data[None],
        "target": mask[None],
        "instance_mapping": [{}],
    }


CASES = [
    # Base Models
    (RetinaUNetV001, "retinaunet_v001"),
    (RetinaUNetV001, "retinaunet_v001_mod"),
    (RetinaUNetHNMV002, "retinaunet_hnm_v002"),
    (RetinaUNetFocalV002, "retinaunet_focal_v002"),
    (RetinaUNetFocalResV002, "retinaunet_focal_v002"),
    (RetinaNetHNMV002, "retinaunet_hnm_v002"),
    (RetinaNetFocalV002, "retinaunet_focal_v002"),
    (RetinaNetFocalResV002, "retinaunet_focal_v002"),
    # Dev Models
    (BoxDETRC002, "detr_c002"),
    # (RetinaNetC001, "v001"),
    # (RetinaNetC001Focal, "c014_focal"),
    # (RetinaUNetCV001Focal, "c014_focal"),
    # (FasterRCNNC001, "frcnn_c001"),
    # (MaskRCNNC001, "mrcnn_c001"),
    # (CascadeMaskURCNNC001, "cascmrcnn_c001"),
]


DEVICES = ["cpu"]
# TODO: gpu tests
# pytest.param(
#     "cuda",
#     marks=pytest.mark.skipif(
#         not torch.cuda.is_available(), reason="No cuda gpu available"
#     ),
# ),


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("step", ["train", "val"])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("empty_batch", [True, False])
def test_model_step_smoke(
    example_plan,
    case: Tuple[Callable, str],
    step: str,
    device: str,
    empty_batch: bool,
):
    # Skip RCNN CPU tests ...
    if "rcnn" in case[1] and device == "cpu":
        return

    module_cls, ov = case

    GlobalHydra.instance().clear()  # clear hydra
    initialize_config_module(config_module="nndet.conf", version_base="1.1")
    cfg = compose("config.yaml", overrides=[f"train={ov}"])
    OmegaConf.set_struct(cfg, False)
    cfg["task"] = "Task000_TEST"

    module = module_cls(
        model_cfg=OmegaConf.to_container(cfg["model_cfg"], resolve=True),
        trainer_cfg=OmegaConf.to_container(cfg["trainer_cfg"], resolve=True),
        plan=example_plan,
    )
    module.to(device)

    _batch_fn = example_empty_batch if empty_batch else example_batch
    batch = _batch_fn(
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
