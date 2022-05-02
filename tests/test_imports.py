from nndet.utils.info import SuppressPrint


def test_nevergrad_import():
    import nevergrad as ng


def test_batchgenerators_import():
    import batchgenerators


def test_pytorch_lightning_import():
    import pytorch_lightning as pl


def test_nnunet_import():
    with SuppressPrint():
        import nnunet.preprocessing.preprocessing as nn_preprocessing
