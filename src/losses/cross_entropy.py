from torch.nn import CrossEntropyLoss as TorchCrossEntropyLoss

from src.losses import BaseLoss


class CrossEntropyLoss(BaseLoss, TorchCrossEntropyLoss):
    __doc__ = TorchCrossEntropyLoss.__doc__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
