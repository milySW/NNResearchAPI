from typing import Tuple

import torch.nn as nn
import torch

from src.models.base import LitModel
from configs import DefaultModel


class DefaultResnet(DefaultModel):
    name: str = "xresnet18"
    expansion: int = 1
    layers: Tuple = ()
    model: LitModel = LitModel()
    depth: int = 4
    in_channels: int = 1
    out_channels: int = 4
    bias: float = None
    activation: torch.nn.Module = nn.ReLU(inplace=True)
