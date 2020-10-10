import torch

from src.hooks.base import BaseHook
from src.optimizers import BaseOptim


class SelectiveBackprop(BaseHook):
    @staticmethod
    def backward(
        trainer, loss: torch.Tensor, optim: BaseOptim, optimizer_idx: int
    ) -> None:
        NotImplemented
