from torch.nn import CrossEntropyLoss as TorchCrossEntropyLoss

from src.losses import BaseLoss


class CrossEntropyLoss(BaseLoss, TorchCrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
