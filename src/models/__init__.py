from pathlib import Path

import torch

from configs import DefaultConfig
from src.models.resnet_blocks import XResNet


def get_model(config: DefaultConfig):
    name = config.model.name

    models = dict(
        xresnet18=XResNet,
        xresnet34=XResNet,
        xresnet50=XResNet,
        xresnet101=XResNet,
        xresnet152=XResNet,
    )

    kwargs = dict(
        xresnet18=dict(expansion=1, layers_list=(2, 2, 2, 2), config=config),
        xresnet34=dict(expansion=1, layers_list=(3, 4, 6, 3), config=config),
        xresnet50=dict(expansion=4, layers_list=(3, 4, 6, 3), config=config),
        xresnet101=dict(expansion=4, layers_list=(3, 4, 23, 3), config=config),
        xresnet152=dict(expansion=4, layers_list=(3, 8, 36, 3), config=config),
    )

    assert name in models.keys(), "Could not find model name"
    return models[name](**kwargs[name])


def load_model(config, model_path: Path):
    model = get_model(config)
    model.load_state_dict(torch.load(model_path)["state_dict"])
    return model
