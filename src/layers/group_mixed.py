from typing import Iterable, List

import torch


class PoolMixed1d(torch.nn.Module):
    def __init__(self, flattened_size: Iterable[int], pool: torch.nn.Module):
        super().__init__()

        self.flattened_size = flattened_size
        self.pool = pool

    @staticmethod
    def reorder(x: torch.Tensor, shapes: List[Iterable[int]]) -> torch.Tensor:
        reshaped = torch.reshape(x, shapes[0])
        permuted = reshaped.permute(0, 1, 3, 2)
        flattened = torch.reshape(permuted, shapes[1])

        return flattened

    def get_shape(self, shape: Iterable[int], before: bool) -> Iterable[int]:
        if len(shape) > 3 and before:
            return shape
        elif len(shape) > 3 and not before:
            return shape[:-2] + (shape[-1], shape[-2])

        if before:
            new_shape = (shape[-1] // self.flattened_size, self.flattened_size)
        elif not before:
            new_shape = (self.flattened_size, shape[-1] // self.flattened_size)

        return shape[:-1] + new_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)
        # current_shape = x.shape
        # new_shape = self.get_shape(shape=current_shape, before=True)

        # reordered = self.reorder(x=x, shapes=[new_shape, current_shape])
        # pooled = self.pool(reordered)

        # current_shape = pooled.shape
        # new_shape = self.get_shape(shape=current_shape, before=False)

        # ordered = self.reorder(x=pooled, shapes=[new_shape, current_shape])

        # return ordered
