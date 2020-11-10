from typing import Generator

from torch.nn.parameter import Parameter
from torch.optim import SGD as TorchSGD

from src.base.optimizers import BaseOptim


class SGD(BaseOptim, TorchSGD):
    __doc__ = TorchSGD.__doc__

    def __init__(self, params: Generator[Parameter, None, None], **kwargs):
        super().__init__(params, **kwargs)
