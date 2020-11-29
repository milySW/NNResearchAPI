from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch

from torch import nn

from configs import DefaultConfig, DefaultResnet
from src.base.models import LitModel
from src.layers import PoolMixed1d
from src.models.utils import LayersMap, conv_layer, load_state_dict
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
        int kernel_size: Size of convolutional kernel
        int stride: controls the stride for the cross-correlation
        bool bias: Flag responsible for adding a learnable bias layer
        bool xresnet: Flag responsible for using XResNet architecture
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
        layers_map: LayersMap,
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
                layers_map=layers_map,
            )

            layer_2 = conv_layer(
                n_inputs=n_filters,
                n_filters=n_filters,
                kernel_size=kernel_size,
                bias=bias,
                activation=None,
                layers_map=layers_map,
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
                stride=stride if not xresnet else 1,
                bias=bias,
                activation=activation,
                layers_map=layers_map,
            )

            layer_2 = conv_layer(
                n_inputs=n_filters,
                n_filters=n_filters,
                kernel_size=kernel_size,
                stride=stride if xresnet else 1,
                bias=bias,
                activation=activation,
                layers_map=layers_map,
            )

            layer_3 = conv_layer(
                n_inputs=n_filters,
                n_filters=n_filters,
                kernel_size=1,
                bias=bias,
                activation=None,
                layers_map=layers_map,
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
                layers_map=layers_map,
            )

            # Add AvgPool because of XResNet tweaks --> ResNet-D
            # info: https://towardsdatascience.com
            # /xresnet-from-scratch-in-pytorch-e64e309af722

            pool_params = dict(kernel_size=2, stride=2, ceil_mode=True)
            pool = layers_map.AvgPool(**pool_params)
            self.pool = PoolMixed1d(flattened_size=2, pool=pool)

        elif not n_inputs == n_filters and not xresnet:
            # If downsampling block in ResNet

            self.id_conv = conv_layer(
                n_inputs=n_inputs,
                n_filters=n_filters,
                kernel_size=1,
                stride=stride,
                bias=bias,
                activation=None,
                layers_map=layers_map,
            )

            self.pool = nn.Identity()

        self.activation = activation

    def forward(self, x):
        identity = self.id_conv(self.pool(x))
        shape = identity.shape

        conv = nn.functional.interpolate(self.convs(x), shape[-1])
        return self.activation(conv + identity)


class ResNet(LitModel):
    """
    Creates the standard `ResNet` model.

    Parameters:

        int expansion: Model expantion
        List[int] blocks: List with number of blocks stacked
        DefaultConfig config: config with model parameters

    """

    def __init__(self, expansion: int, blocks: List, config: DefaultConfig):

        self.model_check(config.model, DefaultResnet, "ResNet")

        super().__init__(config)
        self.set_params(config)
        self.blocks = self.set_blocks(blocks)

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
                    layers_map=self.layers_map,
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
                layers_map=self.layers_map,
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
                n_blocks=number_of_blocks,
                stride=1 if i == 0 else 2,
                bias=self.bias,
                xresnet=self.xresnet,
                activation=self.activation,
                layers_map=self.layers_map,
            )
            for i, number_of_blocks in enumerate(self.blocks)
        ]

        if self.additional_dense_layers:
            first_in = n_filters[-1] * expansion
            last_in = first_in // 2
            final_layers = []

            for i in range(self.additional_dense_layers):
                final_layers.append(nn.Dropout(p=self.dropout))
                final_layers.append(
                    nn.Linear(
                        in_features=first_in if i == 0 else last_in,
                        out_features=last_in,
                    )
                )

        else:
            last_in = n_filters[-1] * expansion
            final_layers = []

        max_pool = self.layers_map.MaxPool(kernel_size=3, stride=2, padding=1)
        self.layers = nn.ModuleList(
            [
                *stem,
                PoolMixed1d(flattened_size=2, pool=max_pool),
                *res_layers,
                self.layers_map.AdaptiveAvgPool(1),
                nn.Flatten(),
                *final_layers,
                nn.Dropout(p=self.dropout),
                nn.Linear(in_features=last_in, out_features=self.out_channels),
            ]
        )

        for module in self.layers:
            if getattr(module, "bias", None) is not None:
                nn.init.constant_(module.bias, 0)
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)

        if self.pretrained and self.bias:
            self.log_pretrained_weights_bias_warning()

        elif self.pretrained and self.f_maps != 64:
            self.log_pretrained_weights_f_maps_warning()

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
        layers_map: LayersMap,
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
                layers_map=layers_map,
            )

            resnet_blocks.append(resnet_block)

        return nn.Sequential(*resnet_blocks)

    @property
    def layers_map(self):
        layers_dim_map = {
            "1D": LayersMap(
                AdaptiveAvgPool=torch.nn.AdaptiveAvgPool1d,
                MaxPool=torch.nn.MaxPool1d,
                AvgPool=torch.nn.AvgPool1d,
                Conv=torch.nn.Conv1d,
                BatchNorm=torch.nn.BatchNorm1d,
            ),
            "2D": LayersMap(
                AdaptiveAvgPool=torch.nn.AdaptiveAvgPool2d,
                MaxPool=torch.nn.MaxPool2d,
                AvgPool=torch.nn.AvgPool2d,
                Conv=torch.nn.Conv2d,
                BatchNorm=torch.nn.BatchNorm2d,
            ),
            "3D": LayersMap(
                AdaptiveAvgPool=torch.nn.AdaptiveAvgPool3d,
                MaxPool=torch.nn.MaxPool3d,
                AvgPool=torch.nn.AvgPool3d,
                Conv=torch.nn.Conv3d,
                BatchNorm=torch.nn.BatchNorm3d,
            ),
        }

        return layers_dim_map[self.data_dim]

    def log_pretrained_weights_bias_warning(self):
        info = "Setting pretrained weights will be ommited."
        cause = f"pretrained weights for XResNet with bias == {self.bias}"
        logger.warn(f"{info} Using {cause} are not supported.")

    def log_pretrained_weights_f_maps_warning(self):
        info = "Setting pretrained weights will be ommited."
        cause = "pretrained weights for XResNet with f_maps != 64"
        logger.warn(f"{info} Using {cause} are not supported.")

    def log_pretrained_weights_xresnet_warning(self):
        info = "Setting pretrained weights will be ommited."
        cause = "pretrained weights for XResNet with expansion > 1"
        logger.warn(f"{info} Using {cause} are not supported.")

    def set_params(self, config: DefaultConfig):
        self.in_channels = config.model.in_channels
        self.activation = config.model.activation
        self.out_channels = config.model.out_channels
        self.f_maps = config.model.f_maps
        self.bias = config.model.bias
        self.depth = config.model.depth
        self.ks = config.model.kernel_size
        self.xresnet = config.model.xresnet
        self.dropout = config.model.dropout
        self.additional_dense_layers = config.model.additional_dense_layers

    def set_blocks(self, blocks: List[int]) -> List[int]:
        self.check_depth()
        del blocks[slice(1, 5 - self.depth)]

        return blocks

    def check_depth(self):
        info = f"depth {self.depth} not supported, 4 or less"
        assert self.depth <= 4, info

    def get_filters(self, i: int, exp: int):
        return self.f_maps // exp if i == 0 else self.f_maps * 2 ** (i - 1)

    def forward(self, x):
        for layer in self.layers:
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
