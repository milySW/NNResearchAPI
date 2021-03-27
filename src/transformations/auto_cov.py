from typing import Iterable

import torch

from src.base.transformation import BaseTransformation


class AutoCov(BaseTransformation):
    """
    Auto-covariance function.

    Parameters:
        int size: the amount of time by which the signal has been shifted (lag)
    """

    def __init__(self, size: int, **kwargs):
        self.size = size
        super().__init__(**kwargs)

    def transformation(self, data: Iterable) -> torch.Tensor:
        return self.core_transofmation(data, size=self.size)

    @staticmethod
    def core_transofmation(data: Iterable, size: int) -> torch.Tensor:
        N = data.shape[-1]
        shape = (data.shape[0], 1, *data.shape[-2:])

        auto_cov_data = torch.zeros(size=shape)

        for row, vector in enumerate(data):
            mean = vector.mean()

            var = (vector[0, :, size:] - mean) * (vector[0, :, :-size] - mean)
            auto_cov_data[row][0][:, : N - size] = var

        tensor = (1 / (N - 1)) * auto_cov_data
        tensor = torch.cat([data, tensor], dim=1)

        return tensor
