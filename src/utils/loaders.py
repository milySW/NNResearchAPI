from __future__ import annotations

import importlib

from pathlib import Path
from typing import Any, Generator, Tuple

import numpy as np

from torch.utils.data.dataloader import DataLoader

import configs


def load_variable(variable_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(variable_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, variable_name)


def load_custom_sets(data_path: Path, *args, **kwargs) -> Tuple[np.array, ...]:
    return NotImplemented


def load_default_sets(path_to_data: Path) -> Tuple[np.array, ...]:
    X_train = np.expand_dims(np.load(path_to_data / "X_train.npy"), axis=1)
    y_train = np.load(path_to_data / "y_train.npy")

    X_test = np.expand_dims(np.load(path_to_data / "X_test.npy"), axis=1)
    y_test = np.load(path_to_data / "y_test.npy")

    X_val = np.load(path_to_data / "X_val.npy")
    y_val = np.load(path_to_data / "y_val.npy")

    train, test, valid = (X_train, y_train), (X_test, y_test), (X_val, y_val)
    sets = dict(train=train, test=test, valid=valid)

    return sets


def get_loader(
    x_data: np.array, labels: np.array, shuffle: bool, batch_size: int
) -> DataLoader:
    data = list(zip(x_data, labels))
    loader = DataLoader(data, shuffle=shuffle, batch_size=batch_size)
    return loader


def get_loaders(
    path_to_data: Path, config: configs.DefaultConfig
) -> Generator[DataLoader]:

    loading_func = config.training.loader_func
    bs = config.training.batch_size

    sets = loading_func(path_to_data)
    for key, data_set in sets.items():
        shuffle = True if key == "train" else False
        loader = get_loader(*data_set, shuffle=shuffle, batch_size=bs)

        yield loader
