import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from nndet.utils.check import env_guard
from nndet.io import get_case_ids_from_dir, save_pickle


@env_guard
def main():
    det_data_dir = Path(os.getenv('det_data'))
    task_data_dir = det_data_dir / "Task019FG_ADAM"

    target_label_dir = task_data_dir / "raw_splitted" / "labelsTr"
    splits_file_dir = task_data_dir / "preprocessed"
    splits_file_dir.mkdir(parents=True, exist_ok=True)
    splits_file = splits_file_dir / "splits_final.pkl"

    case_ids = sorted(get_case_ids_from_dir(target_label_dir, remove_modality=False))
    case_ids_pat = [c if c.isdigit() else c[:-1] for c in case_ids]
    case_ids_pat_unique = list(set(case_ids_pat))
    print(f"Found {len(case_ids_pat_unique)} unique patient ids.")

    splits = []
    kfold = GroupKFold(n_splits=5)
    for i, (train_idx, test_idx) in enumerate(kfold.split(case_ids, groups=case_ids_pat)):
        train_keys = np.array(case_ids)[train_idx]
        test_keys = np.array(case_ids)[test_idx]

        splits.append(OrderedDict())
        splits[-1]['train'] = train_keys
        splits[-1]['val'] = test_keys
        print(f"Generated split: {splits[-1]}")
    save_pickle(splits, splits_file)
   

if __name__ == '__main__':
    main()
