from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from torch.hub import load_state_dict_from_url

from src.models.base import LitModel

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


def load_state_dict(model: LitModel) -> Any:
    pretrained_dict = load_state_dict_from_url(model_urls[model.name])
    model_dict = model.state_dict()

    model_dict, layers = unify_keys(pretrained_dict, model_dict, model)

    model.load_state_dict(model_dict)
    model.update_pretrained_layers(layers)
    model.freeze_pretrained_layers(freeze=True)


def group_dict(dictionary):
    subgroups = unique_keys([split_name(i, 0, 3) for i in dictionary.keys()])
    groups = unique_keys([split_name(i, 0, -1) for i in subgroups])
    keys = dictionary.keys()

    return groups, subgroups, keys


def unify_keys(pretrained_dict, model_dict, model: LitModel):
    pre_groups, pre_subgroups, pre_keys = group_dict(pretrained_dict)
    model_groups, model_subgroups, model_keys = group_dict(model_dict)
    unique = unique_keys([split_name(i, -1, None) for i in model_dict.keys()])

    if model.xresnet:
        model_groups = model_groups[3:-1]
    elif not model.xresnet:
        model_groups = model_groups[1:-1]

    pre_groups = pre_groups[2:-1]
    pre_groups = tune_with_depth(model_groups, pre_groups, model.depth)

    pre_subgroups = filter_list(pre_subgroups, pre_groups)
    pre_keys = filter_list(pre_keys, pre_groups)

    model_subgroups = filter_list(model_subgroups, model_groups)
    model_keys = filter_list(model_keys, model_groups)

    pretrained_layers = []
    for suffix in unique:
        model_layer_keys = [layer for layer in model_keys if suffix in layer]
        pre_layer_keys = [layer for layer in pre_keys if suffix in layer]

        weights = [pretrained_dict[layer] for layer in pre_layer_keys]
        zipped = zip(model_layer_keys, weights)

        weights_dict = {layer: weight for layer, weight in zipped}
        model_dict.update(weights_dict)

        pretrained_layers.extend(list(weights_dict.keys()))

    return model_dict, pretrained_layers


def tune_with_depth(model_groups, pre_groups, depth):
    indices = slice(1, 5 - depth)
    del model_groups[indices]

    main_groups = unique_keys([split_name(i, 0, -1) for i in pre_groups])
    del main_groups[indices]

    return filter_list(pre_groups, main_groups)


def check_prefix(key, prefix_list):
    return any([key.startswith(prefix) for prefix in prefix_list])


def filter_list(key_list, prefix_list):
    return [key for key in key_list if check_prefix(key, prefix_list)]


def split_name(name, start, stop):
    return ".".join(name.split(".")[start:stop])


def unique_keys(keys):
    return list(OrderedDict.fromkeys(keys))
