import torch
import torch.nn as nn

from configs.tunable.models.model_template import DefaultModel


class DefaultResnet(DefaultModel):
    """
    Config responsible for setting parameters for :class:`ResNet` architecture.

    Parameters:

        str name: Name of the architecture
        int data_dim: Dimension of provided data. Supprored: ["1D", "2D", "3D"]

        int depth: Model depth
        int in_channels: Number of input channels
        int out_channels: Number of output channels
        int kernel_size: Size of defaut kernel used in architecture
        int f_maps: Default number of feature maps used in architecture
        bool bias: Flag responsible for adding a learnable bias to the output
        torch.nn.Module activation: Model activation function

        float dropout: probability of an element to be zeroed by dropout layer
        int additional_dense_layers: Number of additional
            (dense + dropout) block at the and of the layer
        bool xresnet: Flag responsible for using additional tweaks suggested
            by Jeremy Howard (co-founder of fast.ai)

        int freezing_start: Layer where freezing starts
        int freezing_stop:: Layer where freezing ends

    """

    name: str = "resnet34"
    data_dim: str = "1D"

    # Architecture
    depth: int = 3
    in_channels: int = 1
    out_channels: int = 4
    kernel_size: int = 16
    f_maps: int = 64
    bias: bool = True
    activation: torch.nn.Module = nn.ReLU(inplace=True)

    # Additional features
    dropout: float = 0.5
    additional_dense_layers: int = 1
    xresnet: bool = True

    # Pretrained weights (Supported only for 2d)
    freezing_start: int = 9 if xresnet else 3
    freezing_stop: int = -2
