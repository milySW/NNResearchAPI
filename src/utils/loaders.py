from __future__ import annotations

import importlib

from pathlib import Path
from typing import Any, Generator, List, Tuple

import numpy as np
import torch

import configs

from src.loarders import DataLoader


def load_variable(variable_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(variable_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, variable_name)


def load_custom_sets(
    path: Path, dtype=torch.dtype, *args, **kwargs
) -> Tuple[np.array, ...]:
    return NotImplemented


def load_custom_set(
    path: Path, dtype=torch.dtype, *args, **kwargs
) -> Tuple[np.array, ...]:
    return NotImplemented


def load_default_sets(path: Path, dtype=torch.dtype) -> Tuple[np.array, ...]:
    X_train, y_train = load_set(path, ["X_train.npy", "y_train.npy"], dtype)
    X_val, y_val = load_set(path, ["X_val.npy", "y_val.npy"], dtype)
    X_test, y_test = load_set(path, ["X_test.npy", "y_test.npy"], dtype)

    train, test, valid = (X_train, y_train), (X_val, y_val), (X_test, y_test)
    sets = dict(train=train, test=test, valid=valid)

    return sets


def load_set(
    path: Path, set_names: List[str], dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor]:

    X = load_x(path / set_names[0])
    y = load_y(path / set_names[1])

    return X, y


def load_x(path: Path, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(np.expand_dims(np.load(path), axis=1), dtype=dtype)


def load_y(path: Path, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(np.load(path), dtype=dtype)


def get_loader(
    x_data: np.array,
    labels: np.array,
    config: configs.DefaultConfig,
    dataset_type: str = "train",
) -> DataLoader:

    data = [x_data, labels]
    loader = DataLoader(dataset=data, config=config, dataset_type=dataset_type)
    return loader


def get_loaders(
    path_to_data: Path, config: configs.DefaultConfig,
) -> Generator[DataLoader, Path, None]:

    loading_func = config.training.loader_func
    dtype = config.training.dtype

    sets = loading_func(path_to_data, dtype)
    for key, data_set in sets.items():
        loader = get_loader(*data_set, config=config, dataset_type=key)
        yield loader
