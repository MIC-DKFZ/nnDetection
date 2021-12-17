import pytest
from hydra import compose, initialize_config_module


def test_basic_compose():
    with initialize_config_module(config_module="nndet.conf"):
        cfg = compose("config.yaml", overrides=["train=v001_mod"])

        # check for keys
        assert cfg["exp"]["fold"] == 0
        assert cfg["exec"]["mode"] == "overwrite"

        assert cfg["module"] is not None
        assert cfg["plan"] is not None
        assert cfg["planner"] is not None

        assert cfg["augment_cfg"]["name"] is not None
        assert cfg["io_cfg"]["multiprocessing"] is not None
        assert cfg["trainer_cfg"]["gpus"] is not None
        assert cfg["model_cfg"]["backbone_kwargs"] is not None


def test_batch_size_overwrite():
    with initialize_config_module(config_module="nndet.conf"):
        cfg = compose(
            "config.yaml", overrides=["train=v001_mod", "+io_cfg.batch_size=42"]
        )

        assert cfg["io_cfg"]["batch_size"] == 42


def test_patch_size_overwrite():
    with initialize_config_module(config_module="nndet.conf"):
        cfg = compose(
            "config.yaml", overrides=["train=v001_mod", "+io_cfg.patch_size=42"]
        )

        assert cfg["io_cfg"]["patch_size"] == 42


def test_splits_overwrite():
    with initialize_config_module(config_module="nndet.conf"):
        cfg = compose(
            "config.yaml", overrides=["train=v001_mod", "+io_cfg.splits=custom"]
        )

        assert cfg["io_cfg"]["splits"] == "custom"


def test_v001():
    with initialize_config_module(config_module="nndet.conf"):
        cfg_backup = compose("config.yaml", overrides=["train=v001"])
    with initialize_config_module(config_module="nndet.conf"):
        cfg_mod = compose("config.yaml", overrides=["train=v001_mod"])
    assert cfg_backup == cfg_mod


@pytest.mark.parametrize("name", ["c014", "c014_focal"])
def test_backups(name):
    with initialize_config_module(config_module="nndet.conf"):
        cfg_backup = compose("config.yaml", overrides=[f"train={name}_backup"])
    with initialize_config_module(config_module="nndet.conf"):
        cfg_mod = compose("config.yaml", overrides=[f"train={name}"])
    assert cfg_backup == cfg_mod
