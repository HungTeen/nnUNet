from typing import List

import numpy as np
from sklearn.model_selection import KFold

from pangteen import config


def generate_crossval_split(train_identifiers: List[str], seed=12345, n_splits=5) -> List[dict[str, List[str]]]:
    splits = []
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (train_idx, test_idx) in enumerate(kfold.split(train_identifiers)):
        train_keys = np.array(train_identifiers)[train_idx]
        test_keys = np.array(train_identifiers)[test_idx]
        splits.append({})
        splits[-1]['train'] = list(train_keys)
        splits[-1]['val'] = list(test_keys)
    return splits


def generate_lock_split(train_identifiers: List[str], seed=12345, n_splits=5) -> List[dict[str, List[str]]]:
    """
    生成指定的验证集划分。
    """
    splits = []
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    valid_set = config.val_id_set
    for s in range(n_splits):
        train_keys = []
        val_keys = []
        for i in range(len(train_identifiers)):
            case_id = int(train_identifiers[i].split('_')[1].split('.')[0])
            if case_id in valid_set:
                val_keys.append(train_identifiers[i])
            else:
                train_keys.append(train_identifiers[i])
        splits.append({})
        splits[-1]['train'] = train_keys
        splits[-1]['val'] = val_keys
    return splits