from torch.nn import CrossEntropyLoss as TorchCrossEntropyLoss

from src.base.losses import BaseLoss


class CrossEntropyLoss(BaseLoss, TorchCrossEntropyLoss):
    __doc__ = TorchCrossEntropyLoss.__doc__

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
