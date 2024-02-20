from typing import Callable, Tuple

import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf.omegaconf import OmegaConf

from nndet.core.rois.pooler.roi_align import roi_align_3d
from nndet.ptmodule.detr.dev.c002 import BoxCDETRC002, BoxDETRC002, BoxDETRCEC002
from nndet.ptmodule.detr.dev.def_detr_c002 import BoxDeformableDETRC002
from nndet.ptmodule.retinanet.rn_v002 import RetinaNetFocalV002, RetinaNetHNMV002
from nndet.ptmodule.retinaunet2sm.run2sm_v002 import RetinaNet2SMV002, RetinaUNet2SMV002
from nndet.ptmodule.retinaunet.run_v001 import RetinaUNetV001
from nndet.ptmodule.retinaunet.run_v002 import RetinaUNetFocalV002, RetinaUNetHNMV002


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
    # Single Stage
    (RetinaUNetV001, "retinaunet_v001"),
    (RetinaUNetV001, "retinaunet_v001_mod"),
    (RetinaUNetHNMV002, "retinaunet_hnm_v002"),
    (RetinaUNetFocalV002, "retinaunet_focal_v002"),
    (RetinaNetHNMV002, "retinaunet_hnm_v002"),
    (RetinaNetFocalV002, "retinaunet_focal_v002"),
    # Set Prediction
    (BoxDETRCEC002, "detr_softm_c002"),
    (BoxDETRC002, "detr_sigm_c002"),
    (BoxCDETRC002, "detr_sigm_c002"),
    (BoxDeformableDETRC002, "def_detr_c002"),
]


if roi_align_3d is not None and torch.cuda.is_available():
    # Two Stage
    CASES.append((RetinaNet2SMV002, "mrcnn_v002"))
    CASES.append((RetinaUNet2SMV002, "mrcnn_v002"))


DEVICES = ["cpu"]


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
