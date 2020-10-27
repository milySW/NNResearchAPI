import torch
import torch.nn as nn

from configs.tunable.models.model_template import DefaultModel


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
        bool bias: Flag responsible for adding a learnable bias to the output
        torch.nn.Module activation: Model activation function

    """

    name: str = "xresnet18"
    expansion: int = 1
    depth: int = 4
    in_channels: int = 1
    out_channels: int = 4
    kernel_size: int = 3
    f_maps: int = 64
    bias: bool = True
    activation: torch.nn.Module = nn.ReLU(inplace=True)
