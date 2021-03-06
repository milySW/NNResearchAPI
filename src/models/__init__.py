from pathlib import Path

import torch

from configs import DefaultConfig
from src.models.resnet_blocks import ResNet
from src.models.rnn import RNNModel


def get_model(config: DefaultConfig):
    name = config.model.name

    models = dict(
        resnet18=ResNet,
        resnet34=ResNet,
        resnet50=ResNet,
        resnet101=ResNet,
        resnet152=ResNet,
        rnn=RNNModel,
    )

    kwargs = dict(
        resnet18=dict(expansion=1, blocks=[2, 2, 2, 2], config=config),
        resnet34=dict(expansion=1, blocks=[3, 4, 6, 3], config=config),
        resnet50=dict(expansion=4, blocks=[3, 4, 6, 3], config=config),
        resnet101=dict(expansion=4, blocks=[3, 4, 23, 3], config=config),
        resnet152=dict(expansion=4, blocks=[3, 8, 36, 3], config=config),
        rnn=dict(config=config),
    )

    assert name in models.keys(), "Could not find model name"
    return models[name](**kwargs[name])


def load_model(config, model_path: Path):
    model = get_model(config)
    model.load_state_dict(torch.load(model_path)["state_dict"])
    return model
