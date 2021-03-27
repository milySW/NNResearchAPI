from typing import Iterable

import torch

from src.base.transformation import BaseTransformation


class GaussianNoise(BaseTransformation):
    def __init__(self, mean: float, std: float, **kwargs):
        self.mean = mean
        self.std = std
        super().__init__(**kwargs)

    def transformation(self, data: Iterable) -> torch.Tensor:
        return self.core_transofmation(data, mean=self.mean, std=self.std)

    @staticmethod
    def core_transofmation(
        data: Iterable, mean: float = 0, std: float = 1
    ) -> torch.Tensor:

        mean = torch.ones(data.shape) * mean
        std = torch.ones(data.shape) * std

        return data + torch.normal(mean, std).to(data.device)
