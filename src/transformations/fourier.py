from typing import Iterable

import torch

from src.base.transformation import BaseTransformation


class Fourier(BaseTransformation):
    __doc__ = torch.fft.__doc__.replace("input", "data")

    def __init__(self, signal_ndim: int, **kwargs):
        self.signal_ndim = signal_ndim
        super().__init__(**kwargs)

    def transformation(self, data: Iterable) -> torch.Tensor:
        return self.core_transofmation(data, signal_ndim=self.signal_ndim)

    @staticmethod
    def core_transofmation(data: Iterable, signal_ndim: int) -> torch.Tensor:
        return torch.rfft(data, signal_ndim=signal_ndim)
