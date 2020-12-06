from typing import Generator

from torch.nn.parameter import Parameter
from torch.optim import Adamax as TorchAdamax

from src.base.optimizer import BaseOptim


class Adamax(BaseOptim, TorchAdamax):
    __doc__ = TorchAdamax.__doc__

    def __init__(self, params: Generator[Parameter, None, None], **kwargs):
        super().__init__(params, **kwargs)
