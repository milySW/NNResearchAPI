from typing import Generator

from torch.nn.parameter import Parameter
from torch.optim import RMSprop as TorchRMSprop

from src.optimizers import BaseOptim


class RMSprop(BaseOptim, TorchRMSprop):
    __doc__ = TorchRMSprop.__doc__

    def __init__(self, params: Generator[Parameter, None, None], **kwargs):
        super().__init__(params, **kwargs)
