from dataclasses import dataclass
from typing import Tuple

from src.models.lightning import LitModel
import torch.nn as nn
import torch
from configs.models.model_template import DefaultModel


@dataclass
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
