from typing import Iterable

import torch

from src.base.transformations import BaseTransformation


class Flatten(BaseTransformation):
    __doc__ = torch.flatten.__doc__

    def __init__(self, start_dim, end_dim, **kwargs):
        self.dims = dict(start_dim=start_dim, end_dim=end_dim)
        super().__init__(**kwargs)

    def transformation(self, data: Iterable) -> torch.Tensor:
        return self.core_transofmation(data, **self.dims)

    @staticmethod
    def core_transofmation(data: Iterable, start_dim, end_dim) -> torch.Tensor:
        data = data.permute(0, 1, 3, 2)
        return torch.flatten(data, start_dim=start_dim, end_dim=end_dim)
