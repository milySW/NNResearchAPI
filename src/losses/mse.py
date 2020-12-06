from torch import Tensor, empty_like, eye
from torch.nn import MSELoss as TorchMSELoss
from torch.nn.functional import mse_loss

from src.base.loss import BaseLoss


class MSELoss(BaseLoss, TorchMSELoss):
    __doc__ = TorchMSELoss.__doc__

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        classes = inputs.shape[-1]
        tensor = empty_like(targets).float()

        targets = eye(n=classes, out=tensor)[targets, :]
        return mse_loss(inputs, targets, reduction=self.reduction)
