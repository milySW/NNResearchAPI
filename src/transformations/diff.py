from typing import Iterable

import torch

from src.base.transformation import BaseTransformation


class Diff(BaseTransformation):
    def __init__(self, lag: int = 1, new_channel: bool = False, **kwargs):
        self.lag = lag
        self.new_channel = new_channel
        super().__init__(**kwargs)

    def transformation(self, data: Iterable) -> torch.Tensor:
        return self.core_transofmation(
            data, lag=self.lag, new_channel=self.new_channel,
        )

    @staticmethod
    def core_transofmation(
        data: Iterable, lag: int, new_channel: bool,
    ) -> torch.Tensor:

        tensor = torch.ones_like(data)
        diff = data[:, :, lag:] - data[:, :, :-lag]

        tensor[:, :, :-lag] = diff

        if new_channel:
            tensor = torch.cat([data, tensor], dim=1)

        return tensor
