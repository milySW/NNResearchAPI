from torch.nn import MSELoss as TorchMSELoss

from src.base.losses import BaseLoss


class MSELoss(BaseLoss, TorchMSELoss):
    __doc__ = TorchMSELoss.__doc__

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
