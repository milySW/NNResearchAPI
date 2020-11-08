from typing import Generator

from torch.nn.parameter import Parameter
from torch.optim import Adagrad as TorchAdagrad

from src.optimizers import BaseOptim


class Adagrad(BaseOptim, TorchAdagrad):
    __doc__ = TorchAdagrad.__doc__

    def __init__(self, params: Generator[Parameter, None, None], **kwargs):
        super().__init__(params, **kwargs)
