import numpy as np
import pytest
import torch

from nndet.io.augmentation.monai import MonaiTransform
from nndet.io.augmentation.torchio import TIOTransform

try:
    import torchio as tio
except ImportError:
    tio = None

try:
    import monai
except ImportError:
    monai = None


@pytest.fixture
def example_np():
    size = (4, 1, 64, 64, 64)
    return {"data": np.random.rand(*size), "label": np.random.randint(0, 2, size=size)}


@pytest.fixture
def example_torch():
    size = (4, 1, 64, 64, 64)
    return {"data": torch.rand(*size), "label": torch.randint(0, 2, size=size)}


@pytest.mark.skipif(tio is None, reason="TorchIO is not available")
def test_tio_smoke(example_np, example_torch):
    transforms = [
        tio.transforms.RandomBiasField(p=1.0),
        tio.transforms.RandomSpike(p=1.0),
    ]

    trafo = TIOTransform(trafo=tio.transforms.Compose(transforms), data_key="data")

    numpy_result = trafo(**example_np)
    torch_result = trafo(**example_torch)


@pytest.mark.skipif(monai is None, reason="Monai is not available")
def test_monai_smoke(example_np, example_torch):
    transforms = [
        monai.transforms.RandBiasFieldD(prob=1.0, keys=["data"]),
        monai.transforms.RandKSpaceSpikeNoiseD(prob=1.0, keys=["data"]),
    ]

    trafo = MonaiTransform(
        trafo=monai.transforms.Compose(transforms),
        data_key="data",
        label_key=None,
    )

    # numpy_result = trafo(**example_np)
    torch_result = trafo(**example_torch)
