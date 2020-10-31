from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

pytorch_prefix = "https://download.pytorch.org/models"
model_urls = dict(
    resnet18=f"{pytorch_prefix}/resnet18-5c106cde.pth",
    resnet34=f"{pytorch_prefix}/resnet34-333f7ec4.pth",
    resnet50=f"{pytorch_prefix}/resnet50-19c8e357.pth",
    resnet101=f"{pytorch_prefix}/resnet101-5d3b4d8f.pth",
    resnet152=f"{pytorch_prefix}/resnet152-b121ed2d.pth",
    resnext50_32x4d=f"{pytorch_prefix}/resnext50_32x4d-7cdf4587.pth",
    resnext101_32x8d=f"{pytorch_prefix}/resnext101_32x8d-8ba56ff5.pth",
    wide_resnet50_2=f"{pytorch_prefix}/wide_resnet50_2-95faca4d.pth",
    wide_resnet101_2=f"{pytorch_prefix}/wide_resnet101_2-32ee1156.pth",
)


def conv_layer(
    n_inputs: int,
    n_filters: int,
    kernel_size: int,
    bias: bool,
    activation: Optional[torch.nn.Module],
    stride: int = 1,
    zero_batch_norm: bool = False,
    use_activation: bool = True,
) -> nn.Sequential:
    """Creates a convolution block for `ResNet`.

    Arguments:
        int n_inputs: Number of inputs
        int n_filters: Number of filters
        int kernel_size: Size of convolutional  kernel
        int stride: controls the stride for the cross-correlation
        bool zero_batch_norm: TOADD
        torch.nn.Module activation: Model activation function
        bool bias: Flag responsible for adding a learnable bias to the output

    """

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
    if activation:
        layers.append(activation)
    return nn.Sequential(*layers)


def save_prediction(predictions: np.ndarray, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, predictions.data.cpu().numpy())
