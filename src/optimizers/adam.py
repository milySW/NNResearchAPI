from torch.optim import Adam as TorchAdam
from src.optimizers import BaseOptim


class Adam(BaseOptim, TorchAdam):
    __doc__ = TorchAdam.__doc__

    def __init__(self, params, *args, **kwargs):
        super().__init__(params, *args, **kwargs)
