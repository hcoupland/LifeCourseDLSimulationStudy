#pylint: disable=invalid-name
"""Data loading functions."""
from collections import Counter
from pathlib import Path
from pprint import pprint
from types import SimpleNamespace

import numpy as np
import yaml

from tsai.data.validation import get_splits


def load_yaml_config(file_path):
    """Read a yml file into namedtuple

    Parameters
    ----------
    file_path : str, path object, file-like object implementing a write() function
        Path to config file

    Returns
    -------
    types.SimpleNamespace
        namespace version of the dict constructed by yaml.safe_load
    """
    cfg = yaml.safe_load(Path(file_path).read_text(encoding='utf-8'))
    pprint(cfg)
    return SimpleNamespace(**cfg)


def load_data(cfg):
    X_raw = np.load(Path(cfg.repo_path, 'data', cfg.x_file)).astype(np.float32)

    y_raw = np.load(Path(cfg.repo_path, 'data', cfg.y_file))

    y = np.expand_dims(y_raw[:, -1].astype(np.int64), -1)

    return X_raw, y


def train_test_split(cfg, X, y):
    # FIXME: valid_size = cfg.test_size doesn't look right?
    splits = get_splits(
        y,
        valid_size=cfg.test_size,
        stratify=True,
        shuffle=True,
        test_size=0,
        show_plot=False,
        random_state=cfg.seed
    )

    X_train, X_test = X[splits[0]], X[splits[1]]
    y_train, y_test = y[splits[0]], y[splits[1]]

    print(f'y: {Counter(y.flatten())}')
    print(f'y_train: {Counter(y_train.flatten())}')
    print(f'y_test: {Counter(y_test.flatten())}')

    return X_train, X_test, y_train, y_test
