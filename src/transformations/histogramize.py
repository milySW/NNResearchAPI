from typing import Iterable

import torch

from src.base.transformation import BaseTransformation


class Histogramize(BaseTransformation):
    def __init__(self, bins: int = 100, new_channel: bool = False, **kwargs):
        self.bins = bins
        self.new_channel = new_channel
        super().__init__(**kwargs)

    def transformation(self, data: Iterable) -> torch.Tensor:
        return self.core_transofmation(
            data, bins=self.bins, new_channel=self.new_channel,
        )

    @staticmethod
    def core_transofmation(
        data: Iterable, bins: int, new_channel: bool,
    ) -> torch.Tensor:

        tensor = torch.ones_like(data)

        for batch_i in range(data.shape[0]):
            for channel_i in range(data.shape[1]):
                hist = torch.histc(data[batch_i][channel_i], bins=bins)
                tensor[batch_i, channel_i, ..., : hist.shape.numel()] = hist

        if new_channel:
            tensor = torch.cat([data, tensor], dim=1)

        return tensor
