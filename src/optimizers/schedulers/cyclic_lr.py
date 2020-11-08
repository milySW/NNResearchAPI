from torch.optim.lr_scheduler import CyclicLR as TorchCyclicLr

from src.optimizers.schedulers.base import BaseScheduler


class CyclicLR(BaseScheduler, TorchCyclicLr):
    __doc__ = TorchCyclicLr.__doc__

    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)
