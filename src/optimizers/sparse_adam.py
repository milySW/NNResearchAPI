from typing import Generator

from torch.nn.parameter import Parameter
from torch.optim import SparseAdam as TorchSparseAdam

from src.optimizers import BaseOptim


class SparseAdam(BaseOptim, TorchSparseAdam):
    __doc__ = TorchSparseAdam.__doc__

    def __init__(self, params: Generator[Parameter, None, None], **kwargs):
        super().__init__(params, **kwargs)
