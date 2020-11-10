from torch.optim.lr_scheduler import ReduceLROnPlateau as TReduceLROnPlateau

from src.base.schedulers import BaseScheduler


class ReduceLROnPlateau(BaseScheduler, TReduceLROnPlateau):
    __doc__ = TReduceLROnPlateau.__doc__

    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)
