from src.models.resnet_blocks import XResNet


def get_model(config):
    name = config.model.name

    models = dict(
        xresnet18=XResNet(expansion=1, layers=(2, 2, 2, 2), config=config),
        xresnet34=XResNet(expansion=1, layers=(3, 4, 6, 3), config=config),
        xresnet50=XResNet(expansion=4, layers=(3, 4, 6, 3), config=config),
        xresnet101=XResNet(expansion=4, layers=(3, 4, 23, 3), config=config),
        xresnet152=XResNet(expansion=4, layers=(3, 8, 36, 3), config=config),
    )

    assert name in models.keys(), "Could not find model name"
    return models[name]
