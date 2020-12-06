from typing import Generator

from torch.nn.parameter import Parameter
from torch.optim import AdamW as TorchAdamW

from src.base.optimizer import BaseOptim


class AdamW(BaseOptim, TorchAdamW):
    __doc__ = TorchAdamW.__doc__

    def __init__(self, params: Generator[Parameter, None, None], **kwargs):
        super().__init__(params, **kwargs)
