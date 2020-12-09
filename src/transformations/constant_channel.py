from typing import Callable, Iterable

import torch

from src.base.transformation import BaseTransformation


class ConstantChannel(BaseTransformation):
    def __init__(self, func: Callable, new_channel: bool = False, **kwargs):
        self.func = func
        self.new_channel = new_channel
        super().__init__(**kwargs)

    def transformation(self, data: Iterable) -> torch.Tensor:
        return self.core_transofmation(data, self.func, self.new_channel)

    @staticmethod
    def core_transofmation(
        data: Iterable, func: Callable, new_channel: bool
    ) -> torch.Tensor:

        tensor = torch.ones_like(data)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                tensor[i][j] *= func(data[i][j])

        if new_channel:
            tensor = torch.cat([data, tensor], dim=1)

        return tensor
