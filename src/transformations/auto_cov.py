from typing import Iterable

import torch

from src.base.transformations import BaseTransformation


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
        shape = (*data.shape[:-1], data.shape[-1] - size)
        data = data.squeeze()
        N = data.shape[-1]

        auto_cov_data = torch.zeros(size=(*data.shape[:-1], N - size))

        for row, vector in enumerate(data):
            mean = vector.mean()

            var = (vector[size:] - mean) * (vector[:-size] - mean)
            auto_cov_data[row] = var

        return (1 / (N - 1)) * auto_cov_data.reshape(shape)
