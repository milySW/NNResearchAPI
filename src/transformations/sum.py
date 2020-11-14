from typing import Iterable

import torch

from src.base.transformations import BaseTransformation


class Sum(BaseTransformation):
    __doc__ = torch.sum.__doc__

    def __init__(self, dim, keepdim, **kwargs):
        self.dim = dim
        self.kdim = keepdim

        super().__init__(**kwargs)

    def transformation(self, data: Iterable) -> torch.Tensor:
        return self.core_transofmation(data, dim=self.dim, keepdim=self.kdim)

    @staticmethod
    def core_transofmation(data: Iterable, dim, keepdim) -> torch.Tensor:
        return torch.sum(data, dim=dim, keepdim=keepdim)
