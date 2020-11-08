from torch.optim.lr_scheduler import StepLR as TorchStepLR

from src.optimizers.schedulers.base import BaseScheduler


class StepLR(BaseScheduler, TorchStepLR):
    __doc__ = TorchStepLR.__doc__

    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)
