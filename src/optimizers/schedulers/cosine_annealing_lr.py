from torch.optim.lr_scheduler import CosineAnnealingLR as TCosineAnnealingLR

from src.base.scheduler import BaseScheduler


class CosineAnnealingLR(BaseScheduler, TCosineAnnealingLR):
    __doc__ = TCosineAnnealingLR.__doc__

    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)
