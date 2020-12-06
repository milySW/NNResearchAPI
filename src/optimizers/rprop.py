from typing import Generator

from torch.nn.parameter import Parameter
from torch.optim import Rprop as TorchRprop

from src.base.optimizer import BaseOptim


class Rprop(BaseOptim, TorchRprop):
    __doc__ = TorchRprop.__doc__

    def __init__(self, params: Generator[Parameter, None, None], **kwargs):
        super().__init__(params, **kwargs)
