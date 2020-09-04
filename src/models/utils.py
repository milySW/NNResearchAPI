import torch
import torch.nn as nn


def conv_layer(
    n_inputs: int,
    n_filters: int,
    kernel_size: int = 3,
    stride: int = 1,
    zero_batch_norm: bool = False,
    use_activation: bool = True,
    activation: torch.nn.Module = nn.ReLU(inplace=True),
    bias: bool = True,
) -> nn.Sequential:
    """Creates a convolution block for `XResNet`."""

    batch_norm = nn.BatchNorm2d(n_filters)
    # initializer batch normalization to 0 if its the final conv layer
    nn.init.constant_(batch_norm.weight, 0.0 if zero_batch_norm else 1.0)
    layers = torch.nn.ModuleList(
        [
            nn.Conv2d(
                in_channels=n_inputs,
                out_channels=n_filters,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=bias,
            ),
            batch_norm,
        ]
    )
    if use_activation:
        layers.append(activation)
    return nn.Sequential(*layers)
