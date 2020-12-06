from torch.optim.lr_scheduler import LambdaLR as TorchLambdaLR

from src.base.scheduler import BaseScheduler


class LambdaLR(BaseScheduler, TorchLambdaLR):
    __doc__ = TorchLambdaLR.__doc__

    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)
