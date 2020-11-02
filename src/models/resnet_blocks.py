from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch

from torch import nn

from configs import DefaultConfig, DefaultResnet
from src.models.base import LitModel
from src.models.utils import conv_layer, load_state_dict
from src.utils.collections import filter_by_prefix, split, unique_keys
from src.utils.logging import get_logger

logger = get_logger("ResNet")


class ResNetBlock(pl.LightningModule):
    """
    Creates the standard `ResNet` block.

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
        xresnet: bool,
        activation: torch.nn.Module,
    ):
        super().__init__()

        n_inputs = n_inputs * expansion
        n_filters = n_filters * expansion

        # convolution path
        if expansion == 1:
            # Residual block
            # info: https://paperswithcode.com/method/residual-block
            # paper: https://arxiv.org/pdf/1512.03385v1.pdf

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
            # Bottleneck residual layer --> Path A
            # info: https://paperswithcode.com/method/bottleneck-residual-block
            # paper: https://arxiv.org/pdf/1512.03385v1.pdf
            # stride condition is connected with xresnet tweaks -> ResNet-B

            layer_1 = conv_layer(
                n_inputs=n_inputs,
                n_filters=n_filters,
                kernel_size=1,
                stride=stride if not self.xresnet else 1,
                bias=bias,
                activation=activation,
            )

            layer_2 = conv_layer(
                n_inputs=n_filters,
                n_filters=n_filters,
                kernel_size=kernel_size,
                stride=stride if self.xresnet else 1,
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

        # identity path / skip connection
        if n_inputs == n_filters:
            # Not downsampling block
            self.id_conv = nn.Identity()
            self.pool = nn.Identity()
        elif not n_inputs == n_filters and xresnet:
            # If downsampling block in XResNet

            self.id_conv = conv_layer(
                n_inputs=n_inputs,
                n_filters=n_filters,
                kernel_size=1,
                bias=bias,
                activation=None,
            )

            # Add AvgPool because of XResNet tweaks --> ResNet-D
            # info: https://towardsdatascience.com
            # /xresnet-from-scratch-in-pytorch-e64e309af722

            self.pool = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)

        elif not n_inputs == n_filters and not xresnet:
            # If downsampling block in ResNet

            self.id_conv = conv_layer(
                n_inputs=n_inputs,
                n_filters=n_filters,
                kernel_size=1,
                stride=stride,
                bias=bias,
                activation=None,
            )

            self.pool = nn.Identity()

        self.activation = activation

    def forward(self, x):
        return self.activation(self.convs(x) + self.id_conv(self.pool(x)))


class ResNet(LitModel):
    """
    Creates the standard `ResNet` model.

    Parameters:

        int expansion: Model expantion
        List[int] layers: List with number of blocks stacked
        DefaultConfig config: config with model parameters

    """

    def __init__(self, expansion: int, layers: List, config: DefaultConfig):

        self.model_check(config.model, DefaultResnet, "ResNet")

        super().__init__(config)
        self.set_params(config)
        self.layers = self.set_layers(layers)

        # create the stem

        stem = []

        if self.xresnet:
            # Add other stem because of the xresnet tweaks --> ResNet-C
            # info: https://towardsdatascience.com
            # /xresnet-from-scratch-in-pytorch-e64e309af722

            n_filters = [
                self.in_channels,
                self.f_maps // 8 * (self.in_channels + 1),
                self.f_maps,
                self.f_maps,
            ]

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

        elif not self.xresnet:
            layer = conv_layer(
                n_inputs=self.in_channels,
                n_filters=self.f_maps,
                kernel_size=7,
                stride=2,
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
                xresnet=self.xresnet,
                activation=self.activation,
            )
            for i, layer in enumerate(self.layers)
        ]

        self.x_res_net = nn.ModuleList(
            [
                *stem,
                nn.MaxPool2d(kernel_size=self.ks, stride=2, padding=1),
                *res_layers,
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(
                    in_features=n_filters[-1] * expansion,
                    out_features=self.out_channels,
                ),
            ]
        )

        for module in self.x_res_net:
            if getattr(module, "bias", None) is not None:
                nn.init.constant_(module.bias, 0)
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)

        if self.pretrained and self.bias:
            self.log_pretrained_weights_bias_warning()

        elif self.pretrained and not (expansion == 1 or not self.xresnet):
            self.log_pretrained_weights_xresnet_warning()

        elif self.pretrained and (expansion == 1 or not self.xresnet):
            load_state_dict(self)
            self.freeze_pretrained_layers(freeze=True)

    @staticmethod
    def _make_layer(
        expansion: int,
        n_inputs: int,
        n_filters: int,
        kernel_size: int,
        n_blocks: nn.Module,
        stride: int,
        bias: bool,
        xresnet: bool,
        activation: nn.Module,
    ) -> nn.Sequential:

        resnet_blocks = []

        for number in range(n_blocks):
            resnet_block = ResNetBlock(
                expansion=expansion,
                n_inputs=n_inputs if number == 0 else n_filters,
                n_filters=n_filters,
                kernel_size=kernel_size,
                stride=stride if number == 0 else 1,
                bias=bias,
                xresnet=xresnet,
                activation=activation,
            )

            resnet_blocks.append(resnet_block)

        return nn.Sequential(*resnet_blocks)

    def log_pretrained_weights_bias_warning(self):
        info = "Setting pretrained weights will be ommited."
        cause = f"pretrained weights for XResNet with bias == {self.bias}"
        logger.warn(f"{info} Using {cause}  is not supported.")

    def log_pretrained_weights_xresnet_warning(self):
        info = "Setting pretrained weights will be ommited."
        cause = "pretrained weights for XResNet with expansion > 1"
        logger.warn(f"{info} Using {cause}  is not supported.")

    def set_params(self, config: DefaultConfig):
        self.in_channels = config.model.in_channels
        self.activation = config.model.activation
        self.out_channels = config.model.out_channels
        self.f_maps = config.model.f_maps
        self.bias = config.model.bias
        self.depth = config.model.depth
        self.ks = config.model.kernel_size
        self.xresnet = config.model.xresnet

    def set_layers(self, layers: List[int]) -> List[int]:
        self.check_depth()
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

    def tune_with_depth(
        self, model_groups: List[str], pre_groups: List[str]
    ) -> List[str]:

        indices = slice(1, 5 - self.depth)
        del model_groups[indices]

        main_groups = unique_keys([split(i, 0, -1) for i in pre_groups])
        del main_groups[indices]

        return filter_by_prefix(pre_groups, main_groups)

    def group_dict(self, data: Dict[str, torch.Tensor]) -> Tuple[List[str]]:
        subgroups = unique_keys([split(i, 0, 3) for i in data.keys()])
        groups = unique_keys([split(i, 0, -1) for i in subgroups])
        keys = data.keys()

        return groups, subgroups, keys

    def unify_keys(
        self,
        pretrained_dict: Dict[str, torch.Tensor],
        model_dict: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], List[str]]:

        model_keys = model_dict.keys()

        pre_groups, pre_subgroups, pre_keys = self.group_dict(pretrained_dict)
        model_groups, model_subgroups, model_keys = self.group_dict(model_dict)
        unique = unique_keys([split(i, -1, None) for i in model_keys])

        if self.xresnet:
            model_groups = model_groups[3:-1]
        elif not self.xresnet:
            model_groups = model_groups[1:-1]

        pre_groups = pre_groups[2:-1]
        pre_groups = self.tune_with_depth(model_groups, pre_groups)

        pre_subgroups = filter_by_prefix(pre_subgroups, pre_groups)
        pre_keys = filter_by_prefix(pre_keys, pre_groups)

        model_subgroups = filter_by_prefix(model_subgroups, model_groups)
        model_keys = filter_by_prefix(model_keys, model_groups)

        pretrained_layers = []
        for suffix in unique:
            model_layers = [layer for layer in model_keys if suffix in layer]
            pre_layers = [layer for layer in pre_keys if suffix in layer]

            weights = [pretrained_dict[layer] for layer in pre_layers]
            zipped = zip(model_layers, weights)

            weights_dict = {layer: weight for layer, weight in zipped}
            model_dict.update(weights_dict)

            pretrained_layers.extend(list(weights_dict.keys()))

        return model_dict, pretrained_layers
