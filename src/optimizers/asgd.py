from typing import Generator

from torch.nn.parameter import Parameter
from torch.optim import ASGD as TorchASGD

from src.optimizers import BaseOptim


class ASGD(BaseOptim, TorchASGD):
    __doc__ = TorchASGD.__doc__

    def __init__(self, params: Generator[Parameter, None, None], **kwargs):
        super().__init__(params, **kwargs)
