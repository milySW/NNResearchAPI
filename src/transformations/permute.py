from typing import Iterable

import torch

from src.base.transformation import BaseTransformation


class Permute(BaseTransformation):
    __doc__ = torch.Tensor.permute.__doc__

    def __init__(self, dims, **kwargs):
        self.dims = dims
        super().__init__(**kwargs)

    def transformation(self, data: Iterable) -> torch.Tensor:
        return self.core_transofmation(data, dims=self.dims)

    @staticmethod
    def core_transofmation(data: Iterable, dims) -> torch.Tensor:
        return data.permute(*dims)
