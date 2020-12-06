from typing import Any, Iterable

from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageFolder


def set_param(previous: Any, current: Any, name: str) -> Any:
    if previous is None or previous == current:
        return current

    else:
        raise ValueError(f"{name} parameter is not same for all configs")


def image_folder(data: Iterable) -> bool:
    is_dataset = isinstance(data, Dataset)
    return is_dataset and isinstance(data.dataset, ImageFolder)
