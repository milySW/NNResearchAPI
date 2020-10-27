from typing import Tuple

import torch
import torch.nn as nn

from configs.tunable.models.model_template import DefaultModel
from src.models.base import LitModel


class DefaultResnet(DefaultModel):
    """
    Config responsible for setting parameters for :class:`ResNet` architecture.

    Parameters:

        str name: Name of the architecture
        int expansion: Model expantion
        tuple layers: TOADD
        LitModel model: Base model with implemented logic
        int depth: Model depth
        int in_channels: Number of input channels
        int out_channels: Number of output channels
        bool bias: TOADD
        torch.nn.Module activation: Model activation function

    """

    name: str = "xresnet18"
    expansion: int = 1
    layers: Tuple = ()
    model: LitModel = LitModel()
    depth: int = 4
    in_channels: int = 1
    out_channels: int = 4
    f_maps: int = 64
    bias: bool = None
    activation: torch.nn.Module = nn.ReLU(inplace=True)
