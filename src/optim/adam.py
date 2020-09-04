from torch.optim import Adam as TorchAdam
from src.optim import BaseOptim


class Adam(BaseOptim, TorchAdam):
    def __init__(self, params, *args, **kwargs):
        super().__init__(params, *args, **kwargs)
