from typing import Iterable, List

import torch

from src.base.transformations import BaseTransformation


class Flip(BaseTransformation):
    __doc__ = torch.flip.__doc__.replace("input", "data")

    def __init__(self, dims: List[int], **kwargs):
        self.dims = dims
        super().__init__(**kwargs)

    def transformation(self, data: Iterable) -> torch.Tensor:
        return self.core_transofmation(data, dims=self.dims)

    @staticmethod
    def core_transofmation(data: Iterable, dims: List[int]) -> torch.Tensor:
        return torch.flip(data, dims=dims)
