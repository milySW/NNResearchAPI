import importlib

from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch

from torchvision import transforms
from torchvision.datasets import ImageFolder


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


def load_image_sets(
    path: Path, dtype: torch.dtype, preprocessors, *args, **kwargs
) -> Tuple[np.array, ...]:

    data_transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor(), *preprocessors]
    )

    dataset = ImageFolder(path / "omatko", transform=data_transform)
    length = len(dataset)

    train_size, valid_size = int(0.7 * length), int(0.15 * length)
    sizes = (train_size, valid_size, length - train_size - valid_size)

    train, valid, test = torch.utils.data.random_split(dataset, sizes)
    sets = dict(train=train, test=test, valid=valid)

    return sets


def load_default_sets(
    path: Path, dtype=torch.dtype, preprocessors=None
) -> Tuple[np.array, ...]:

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


def load_custom_x(path: Path, dtype: torch.dtype = torch.float32):
    return torch.tensor(np.expand_dims(np.load(path), axis=1), dtype=dtype)


def load_y(path: Path, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(np.load(path), dtype=dtype)
