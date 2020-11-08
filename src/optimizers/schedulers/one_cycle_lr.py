from torch.optim.lr_scheduler import OneCycleLR as TorchOneCycleLr

from src.optimizers.schedulers.base import BaseScheduler


class OneCycleLR(BaseScheduler, TorchOneCycleLr):
    __doc__ = TorchOneCycleLr.__doc__

    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)
