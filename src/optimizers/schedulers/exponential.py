from torch.optim.lr_scheduler import ExponentialLR as TorchExponentialLR

from src.base.schedulers import BaseScheduler


class ExponentialLR(BaseScheduler, TorchExponentialLR):
    __doc__ = TorchExponentialLR.__doc__

    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)
