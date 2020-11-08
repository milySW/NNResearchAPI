from torch.optim.lr_scheduler import MultiplicativeLR as TorchMultiplicativeLR

from src.optimizers.schedulers.base import BaseScheduler


class MultiplicativeLR(BaseScheduler, TorchMultiplicativeLR):
    __doc__ = TorchMultiplicativeLR.__doc__

    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)
