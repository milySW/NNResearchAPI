from typing import Iterable

import torch

from src.base.transformation import BaseTransformation


class ArgMax(BaseTransformation):
    __doc__ = torch.argmax.__doc__.replace("input", "data")

    def __init__(self, dim: int, keepdim: bool, **kwargs):
        self.dim = dim
        self.keepdim = keepdim
        super().__init__(**kwargs)

    def transformation(self, data: Iterable) -> torch.Tensor:
        return self.core_transofmation(data, d=self.dim, keep=self.keepdim)

    @staticmethod
    def core_transofmation(data: Iterable, d: int, keep: bool) -> torch.Tensor:
        return torch.argmax(data, dim=d, keepdim=keep)
