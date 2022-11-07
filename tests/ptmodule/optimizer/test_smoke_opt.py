import pytest
import torch

from nndet.ptmodule.optimizer import OPTIMIZER_REGISTRY
from nndet.training.learning_rate import LinearWarmupPolyLR

base_cfg = {
    "initial_lr": 0.01,
    "weight_decay": 0.1,
    "num_train_batches_per_epoch": 250,
}
default_scheduler = {
    "warm_iterations": 250,
    "warm_lr": 0.00001,
    "poly_gamma": 0.9,
}


TEST_CASES = [
    (
        "SGDLWPoly",
        {
            **base_cfg,
            **default_scheduler,
            "sgd_momentum": 0.9,
            "sgd_nesterov": True,
        },
        (torch.optim.SGD, LinearWarmupPolyLR),
    ),
    (
        "AdamWLWPoly",
        {
            **base_cfg,
            **default_scheduler,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
            "amsgrad": True,
        },
        (torch.optim.AdamW, LinearWarmupPolyLR),
    ),
]

# Optional test cases if `torch_optimizer` is installed
try:
    import torch_optimizer as torch_external_optim

    TEST_CASES.append(
        (
            "RAdamLWPoly",
            {
                **base_cfg,
                **default_scheduler,
            },
            (torch_external_optim.RAdam, LinearWarmupPolyLR),
        )
    )
    TEST_CASES.append(
        (
            "RangerLWPoly",
            {
                **base_cfg,
                **default_scheduler,
            },
            (torch_external_optim.Ranger, LinearWarmupPolyLR),
        )
    )
except ImportError:
    pass

# Optional test cases if `ranger21` is installed
try:
    from ranger21 import Ranger21

    TEST_CASES.append(
        (
            "Ranger21",
            {
                **base_cfg,
                **default_scheduler,
            },
            Ranger21,
        )
    )
except ImportError:
    pass

# Optional test cases if `madgrad` is installed
try:
    from madgrad import MADGRAD

    TEST_CASES.append(
        (
            "MadgradLWPoly",
            {
                **base_cfg,
                **default_scheduler,
                "momentum": 0.9,
            },
            (MADGRAD, LinearWarmupPolyLR),
        )
    )
except ImportError:
    pass


class DummyModule(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.train_epochs = 100
        self.trainer_cfg = cfg
        self.layer = torch.nn.Conv2d(10, 10, 3)


@pytest.mark.parametrize("opt_str,cfg,opt_expected_cls", TEST_CASES)
def test_optim_mixin_smoke(opt_str, cfg, opt_expected_cls):
    module = DummyModule(cfg)
    result = OPTIMIZER_REGISTRY[opt_str].configure_optimizers(module)  # get optimizers

    if isinstance(result, tuple):
        optimizer, scheduler = result
        assert isinstance(optimizer[0], opt_expected_cls[0])  # check for correct class
        assert isinstance(scheduler["scheduler"], opt_expected_cls[1])  # check for correct class
    else:
        optimizer = result
        assert isinstance(optimizer, opt_expected_cls)  # check for correct class
