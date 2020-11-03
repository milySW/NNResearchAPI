import torch
import torch.nn as nn

from configs.tunable.models.model_template import DefaultModel


class DefaultResnet(DefaultModel):
    """
    Config responsible for setting parameters for :class:`ResNet` architecture.

    Parameters:

        str name: Name of the architecture
        int depth: Model depth
        int in_channels: Number of input channels
        int out_channels: Number of output channels
        int kernel_size: Size of defaut kernel used in architecture
        int f_maps: Default number of feature maps used in architecture
        bool bias: Flag responsible for adding a learnable bias to the output
        torch.nn.Module activation: Model activation function
        int freezing_start: Layer where freezing starts
        int freezing_stop:: Layer where freezing ends
        bool xresnet: Flag responsible for using additional tweaks suggested
            by Jeremy Howard (co-founder of fast.ai)

    """

    name: str = "resnet34"

    # Architecture
    depth: int = 3
    in_channels: int = 1
    out_channels: int = 4
    kernel_size: int = 3
    f_maps: int = 64
    bias: bool = False
    activation: torch.nn.Module = nn.ReLU(inplace=True)

    # Additional features
    xresnet: bool = True
    dropout: float = 0.2

    # Pretrained weights
    freezing_start: int = 9 if xresnet else 3
    freezing_stop: int = -2
