import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold


def get_StratifiedKFold(y_train, n_splits=5, shuffle=True, random_state=71):

    folds = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    fold_ids = []

    for train_index, valid_index in folds.split(y_train, y_train):
        fold_ids.append([train_index, valid_index])

    return fold_ids


def get_KFold(y_train, n_splits=5, shuffle=True, random_state=71):

    folds = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    fold_ids = []

    for train_index, valid_index in folds.split(y_train, y_train):
        fold_ids.append([train_index, valid_index])

    return fold_ids
