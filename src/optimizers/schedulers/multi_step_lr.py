from torch.optim.lr_scheduler import MultiStepLR as TorchMultiStepLR

from src.base.schedulers import BaseScheduler


class MultiStepLR(BaseScheduler, TorchMultiStepLR):
    __doc__ = TorchMultiStepLR.__doc__

    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)
