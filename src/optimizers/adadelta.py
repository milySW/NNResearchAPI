from typing import Generator

from torch.nn.parameter import Parameter
from torch.optim import Adadelta as TorchAdadelta

from src.base.optimizer import BaseOptim


class Adadelta(BaseOptim, TorchAdadelta):
    __doc__ = TorchAdadelta.__doc__

    def __init__(self, params: Generator[Parameter, None, None], **kwargs):
        super().__init__(params, **kwargs)
