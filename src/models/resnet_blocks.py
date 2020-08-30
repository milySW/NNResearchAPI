from typing import Any
import torch
from torch import nn

from src.models.utils import conv_layer
from src.models.lightning import LitModel
from configs.config_template import DefaultConfig
from configs.models import DefaultResnet


class XResNetBlock(LitModel):
    """Creates the standard `XResNet` block."""

    def __init__(
        self,
        expansion: int,
        n_inputs: int,
        n_filters: int,
        stride: int = 1,
        activation: torch.nn.Module = nn.ReLU(inplace=True),
    ):
        super().__init__()

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
    def __init__(self, expansion: int, layers: list, config: DefaultConfig):
        assert isinstance(
            config.model, DefaultResnet
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
            )
            for i, layer in enumerate(layers)
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

    @property
    def loss_function(self):
        return self.config.training.loss

    @property
    def metrics(self):
        return self.config.training.metrics

    @staticmethod
    def _make_layer(
        expansion: int,
        n_inputs: int,
        n_filters: int,
        n_blocks: nn.Module,
        stride: int,
        activation: nn.Module,
    ) -> nn.Sequential:
        return nn.Sequential(
            *[
                XResNetBlock(
                    expansion=expansion,
                    n_inputs=n_inputs if i == 0 else n_filters,
                    n_filters=n_filters,
                    stride=stride if i == 0 else 1,
                    activation=activation,
                )
                for i in range(n_blocks)
            ]
        )

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
