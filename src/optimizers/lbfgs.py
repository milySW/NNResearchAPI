from typing import Generator

from torch.nn.parameter import Parameter
from torch.optim import LBFGS as TorchLBFGS

from src.base.optimizer import BaseOptim


class LBFGS(BaseOptim, TorchLBFGS):
    __doc__ = TorchLBFGS.__doc__

    def __init__(self, params: Generator[Parameter, None, None], **kwargs):
        super().__init__(params, **kwargs)
