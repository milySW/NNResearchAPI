from typing import Generator

from torch.nn.parameter import Parameter
from torch.optim import Adam as TorchAdam

from src.optimizers import BaseOptim


class Adam(BaseOptim, TorchAdam):
    __doc__ = TorchAdam.__doc__

    def __init__(self, params: Generator[Parameter, None, None], **kwargs):
        super().__init__(params, **kwargs)
