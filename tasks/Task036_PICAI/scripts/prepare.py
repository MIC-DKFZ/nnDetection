import os
from pathlib import Path

from nndet.io import load_json, save_json, save_pickle
from nndet.utils.check import env_guard


@env_guard
def main():
    # create and check folder structure
    det_data_dir = Path(os.getenv("det_data"))
    task_data_dir = det_data_dir / "Task036_PICAI"

    # crate split
    preprocessed_dir = task_data_dir / "preprocessed"
    splits = load_json(preprocessed_dir / "splits")
    assert len(splits) == 5
    save_pickle(splits, preprocessed_dir / "splits")
    save_json(splits, preprocessed_dir / "splits_final")
    save_pickle(splits, preprocessed_dir / "splits_final")


if __name__ == "__main__":
    main()
