from typing import Iterable, List

import torch

from src.transformations.base import BaseTransformation


class Flip(BaseTransformation):
    def __init__(self, dims: List[int], **kwargs):
        self.dims = dims
        super().__init__(**kwargs)

    def transformation(self, data: Iterable):
        return torch.flip(data, dims=self.dims)
