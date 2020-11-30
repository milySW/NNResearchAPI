from torch import Tensor, empty_like, eye
from torch.nn import MSELoss as TorchMSELoss
from torch.nn.functional import mse_loss

from src.base.losses import BaseLoss


class MSELoss(BaseLoss, TorchMSELoss):
    __doc__ = TorchMSELoss.__doc__

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        classes = target.unique().numel()
        tensor = empty_like(target).float()

        target = eye(n=classes, out=tensor)[target, :]
        return mse_loss(input, target, reduction=self.reduction)
