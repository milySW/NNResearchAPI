from __future__ import annotations

from typing import Tuple

import torch

from torch import nn

import configs

from src.models.base import LitModel
from src.models.utils import conv_layer


class XResNetBlock(LitModel):
    """
    Creates the standard `XResNet` block.

    Parameters:

        int expansion: Model expantion
        int n_inputs: Number of inputs
        int n_filters: Number of filters
        int stride: controls the stride for the cross-correlation
        torch.nn.Module activation: Model activation function

    """

    def __init__(
        self,
        config: configs.DefaultConfig,
        expansion: int,
        n_inputs: int,
        n_filters: int,
        stride: int = 1,
        activation: torch.nn.Module = nn.ReLU(inplace=True),
    ):
        super().__init__()
        self.config = config

        n_inputs = n_inputs * expansion
        n_filters = n_filters * expansion

        # convolution path
        if expansion == 1:
            layers = [
                conv_layer(
                    n_inputs=n_inputs,
                    n_filters=n_filters,
                    kernel_size=3,
                    stride=stride,
                    activation=activation,
                ),
                conv_layer(
                    n_inputs=n_filters,
                    n_filters=n_filters,
                    kernel_size=3,
                    activation=activation,
                ),
            ]
        else:
            layers = [
                conv_layer(
                    n_inputs=n_inputs,
                    n_filters=n_filters,
                    kernel_size=1,
                    activation=activation,
                ),
                conv_layer(
                    n_inputs=n_filters,
                    n_filters=n_filters,
                    kernel_size=3,
                    stride=stride,
                    activation=activation,
                ),
                conv_layer(
                    n_inputs=n_filters,
                    n_filters=n_filters,
                    kernel_size=1,
                    activation=activation,
                ),
            ]

        self.convs = nn.Sequential(*layers)

        # identity path
        if n_inputs == n_filters:
            self.id_conv = nn.Identity()
        else:
            self.id_conv = conv_layer(
                n_inputs=n_inputs,
                n_filters=n_filters,
                kernel_size=1,
                use_activation=False,
            )
        if stride == 1:
            self.pool = nn.Identity()
        else:
            self.pool = nn.AvgPool2d(2, ceil_mode=True)

        self.activation = activation

    def forward(self, x):
        return self.activation(self.convs(x) + self.id_conv(self.pool(x)))


class XResNet(LitModel):
    """
    Creates the standard `XResNet` model.

    Parameters:

        int expansion: Model expantion
        tuple layer: TOADD
        DefaultConfig config: config with model parameters

    """

    def __init__(
        self, expansion: int, layers_list: Tuple, config: configs.DefaultConfig
    ):
        assert (
            config.model == configs.DefaultResnet
        ), "Passed config is not for RESNET architecutre!"
        self.config = config
        model = config.model

        # create the stem of the network
        super().__init__()
        n_filters = [model.in_channels, (model.in_channels + 1) * 8, 64, 64]

        stem = []
        for i in range(3):
            stride = 2 if i == 0 else 1
            layer = conv_layer(
                n_inputs=n_filters[i],
                n_filters=n_filters[i + 1],
                stride=stride,
                activation=model.activation,
            )
            stem.append(layer)

        # create `XResNet` blocks
        n_filters = [64 // expansion, 64, 128, 256, 512]

        self.res_layers = [
            self._make_layer(
                expansion=expansion,
                n_inputs=n_filters[i],
                n_filters=n_filters[i + 1],
                n_blocks=layer,
                stride=1 if i == 0 else 2,
                activation=model.activation,
                config=self.config,
            )
            for i, layer in enumerate(layers_list)
        ]

        self.x_res_net = nn.ModuleList(
            [
                *stem,
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                *self.res_layers,
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(n_filters[-1] * expansion, model.out_channels),
            ]
        )

        for module in self.x_res_net:
            if getattr(module, "bias", None) is not None:
                nn.init.constant_(module.bias, 0)
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)

    def forward(self, x):
        for layer in self.x_res_net:
            x = layer(x)
        return x

    @staticmethod
    def _make_layer(
        expansion: int,
        n_inputs: int,
        n_filters: int,
        n_blocks: nn.Module,
        stride: int,
        activation: nn.Module,
        config: configs.DefaultConfig,
    ) -> nn.Sequential:
        return nn.Sequential(
            *[
                XResNetBlock(
                    expansion=expansion,
                    n_inputs=n_inputs if i == 0 else n_filters,
                    n_filters=n_filters,
                    stride=stride if i == 0 else 1,
                    activation=activation,
                    config=config,
                )
                for i in range(n_blocks)
            ]
        )
