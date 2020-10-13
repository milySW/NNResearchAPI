from typing import Iterable

import torch

from src.transformations.base import BaseTransformation


class ArgMax(BaseTransformation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transformation(self, data: Iterable):
        return self.core_transofmation(data)

    @staticmethod
    def core_transofmation(data: Iterable):
        return torch.argmax(data).unsqueeze(dim=0)
