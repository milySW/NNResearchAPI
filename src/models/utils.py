from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from torch.hub import load_state_dict_from_url

from src.base.model import LitModel

pytorch_prefix = "https://download.pytorch.org/models"
model_urls = dict(
    resnet18=f"{pytorch_prefix}/resnet18-5c106cde.pth",
    resnet34=f"{pytorch_prefix}/resnet34-333f7ec4.pth",
    resnet50=f"{pytorch_prefix}/resnet50-19c8e357.pth",
    resnet101=f"{pytorch_prefix}/resnet101-5d3b4d8f.pth",
    resnet152=f"{pytorch_prefix}/resnet152-b121ed2d.pth",
    # Not supported yet
    resnext50_32x4d=f"{pytorch_prefix}/resnext50_32x4d-7cdf4587.pth",
    resnext101_32x8d=f"{pytorch_prefix}/resnext101_32x8d-8ba56ff5.pth",
    wide_resnet50_2=f"{pytorch_prefix}/wide_resnet50_2-95faca4d.pth",
    wide_resnet101_2=f"{pytorch_prefix}/wide_resnet101_2-32ee1156.pth",
)


@dataclass
class LayersMap:
    AdaptiveAvgPool: torch.nn.Module
    MaxPool: torch.nn.Module
    AvgPool: torch.nn.Module
    Conv: torch.nn.Module
    BatchNorm: torch.nn.Module


def conv_layer(
    n_inputs: int,
    n_filters: int,
    kernel_size: int,
    bias: bool,
    activation: Optional[torch.nn.Module],
    layers_map: LayersMap,
    stride: int = 1,
    zero_batch_norm: bool = False,
) -> nn.Sequential:
    """Creates a convolution block for `ResNet`.

    Arguments:
        int n_inputs: Number of inputs
        int n_filters: Number of filters
        int kernel_size: Size of convolutional kernel
        bool bias: Flag responsible for adding a learnable bias to the output
        torch.nn.Module activation: Model activation function
        int stride: controls the stride for the cross-correlation
        bool zero_batch_norm: Flag responsible for initializing
            batch normalization weights to 0

    """

    batch_norm = layers_map.BatchNorm(n_filters)
    # initializer batch normalization to 0 if its the final conv layer
    nn.init.constant_(batch_norm.weight, 0.0 if zero_batch_norm else 1.0)
    layers = torch.nn.ModuleList(
        [
            layers_map.Conv(
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


def load_state_dict(model: LitModel) -> Any:
    pretrained_dict = load_state_dict_from_url(model_urls[model.name])
    model_dict = model.state_dict()

    model_dict, layers = model.unify_keys(pretrained_dict, model_dict)

    model.load_state_dict(model_dict)
    model.update_pretrained_layers(layers)
