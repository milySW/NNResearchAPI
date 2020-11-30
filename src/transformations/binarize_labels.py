from typing import Iterable

import torch

from src.base.transformations import BaseTransformation


class LabelBinarize(BaseTransformation):
    def __init__(self, main_class, **kwargs):
        self.main_class = main_class
        super().__init__(**kwargs)

    def transformation(self, data: Iterable) -> torch.Tensor:
        return self.core_transofmation(data, main_class=self.main_class)

    @staticmethod
    def core_transofmation(data: Iterable, main_class) -> torch.Tensor:
        flat = (data.argmax(dim=1) == main_class).long()
        return torch.eye(2)[flat, :]
