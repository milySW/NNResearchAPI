from src.optimizers.schedulers.base import BaseScheduler
from torch.optim.lr_scheduler import ExponentialLR as TorchExponentialLR


class ExponentialLR(BaseScheduler, TorchExponentialLR):
    def __init__(self, optimizer, *args, **kwargs):
        super().__init__(optimizer, *args, **kwargs)
