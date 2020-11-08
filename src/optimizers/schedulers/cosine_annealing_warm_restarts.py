from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts as TorchCAWR

from src.optimizers.schedulers.base import BaseScheduler


class CosineAnnealingWarmRestarts(BaseScheduler, TorchCAWR):
    __doc__ = TorchCAWR.__doc__

    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)
