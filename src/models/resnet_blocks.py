from typing import Tuple

import torch

from torch import nn

from configs import DefaultConfig, DefaultResnet
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
        expansion: int,
        n_inputs: int,
        n_filters: int,
        kernel_size: int,
        stride: int,
        bias: bool,
        activation: torch.nn.Module,
    ):
        super().__init__()

        n_inputs = n_inputs * expansion
        n_filters = n_filters * expansion

        # convolution path
        if expansion == 1:
            layer_1 = conv_layer(
                n_inputs=n_inputs,
                n_filters=n_filters,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
                activation=activation,
            )

            layer_2 = conv_layer(
                n_inputs=n_filters,
                n_filters=n_filters,
                kernel_size=kernel_size,
                bias=bias,
                activation=None,
            )

            layers = [layer_1, layer_2]

        else:
            layer_1 = conv_layer(
                n_inputs=n_inputs,
                n_filters=n_filters,
                kernel_size=1,
                bias=bias,
                activation=activation,
            )

            layer_2 = conv_layer(
                n_inputs=n_filters,
                n_filters=n_filters,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
                activation=activation,
            )

            layer_3 = conv_layer(
                n_inputs=n_filters,
                n_filters=n_filters,
                kernel_size=1,
                bias=bias,
                activation=None,
            )

            layers = [layer_1, layer_2, layer_3]

        self.convs = nn.Sequential(*layers)

        # identity path
        if n_inputs == n_filters:
            self.id_conv = nn.Identity()
        else:
            self.id_conv = conv_layer(
                n_inputs=n_inputs,
                n_filters=n_filters,
                kernel_size=1,
                bias=bias,
                activation=None,
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
        Tuple[int] layers: Tuple with number of blocks stacked
        DefaultConfig config: config with model parameters

    """

    def __init__(self, expansion: int, layers: Tuple, config: DefaultConfig):

        self.model_check(config.model, DefaultResnet, "ResNet")

        super().__init__()
        self.set_params(config)
        self.layers = self.set_layers(layers)

        # create the stem of the network
        n_filters = [
            self.in_channels,
            self.f_maps // 8 * (self.in_channels + 1),
            self.f_maps,
            self.f_maps,
        ]

        stem = []
        for i in range(3):
            stride = 2 if i == 0 else 1

            layer = conv_layer(
                n_inputs=n_filters[i],
                n_filters=n_filters[i + 1],
                kernel_size=self.ks,
                stride=stride,
                bias=self.bias,
                activation=self.activation,
            )

            stem.append(layer)

        # create `XResNet` blocks
        n_filters = [self.get_filters(index, expansion) for index in range(5)]
        n_filters = n_filters[: self.depth + 1]

        res_layers = [
            self._make_layer(
                expansion=expansion,
                n_inputs=n_filters[i],
                n_filters=n_filters[i + 1],
                kernel_size=self.ks,
                n_blocks=layer,
                stride=1 if i == 0 else 2,
                bias=self.bias,
                activation=self.activation,
            )
            for i, layer in enumerate(self.layers)
        ]

        self.x_res_net = nn.ModuleList(
            [
                *stem,
                nn.MaxPool2d(kernel_size=self.ks, stride=1, padding=1),
                *res_layers,
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(n_filters[-1] * expansion, self.out_channels),
            ]
        )

        for module in self.x_res_net:
            if getattr(module, "bias", None) is not None:
                nn.init.constant_(module.bias, 0)
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)

    def set_params(self, config: DefaultConfig):
        self.config = config
        self.in_channels = config.model.in_channels
        self.activation = config.model.activation
        self.out_channels = config.model.out_channels
        self.f_maps = config.model.f_maps
        self.bias = config.model.bias
        self.depth = config.model.depth
        self.ks = config.model.kernel_size

    def set_layers(self, layers: Tuple[int]) -> Tuple[int]:
        self.check_depth()

        if self.depth < 4:
            del layers[slice(1, 5 - self.depth)]
        return layers

    def check_depth(self):
        info = f"depth {self.depth} not supported, 4 or less"
        assert self.depth <= 4, info

    def get_filters(self, i: int, exp: int):
        return self.f_maps // exp if i == 0 else self.f_maps * 2 ** (i - 1)

    def forward(self, x):
        for layer in self.x_res_net:
            x = layer(x)
        return x

    @staticmethod
    def _make_layer(
        expansion: int,
        n_inputs: int,
        n_filters: int,
        kernel_size: int,
        n_blocks: nn.Module,
        stride: int,
        bias: bool,
        activation: nn.Module,
    ) -> nn.Sequential:

        resnet_blocks = []

        for number in range(n_blocks):
            resnet_block = XResNetBlock(
                expansion=expansion,
                n_inputs=n_inputs if number == 0 else n_filters,
                n_filters=n_filters,
                kernel_size=kernel_size,
                stride=stride if number == 0 else 1,
                bias=bias,
                activation=activation,
            )

            resnet_blocks.append(resnet_block)

        return nn.Sequential(*resnet_blocks)
