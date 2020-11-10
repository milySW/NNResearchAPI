from configs.base.base import BaseConfig


class DefaultModel(BaseConfig):
    """
    Main config responsible for setting parameters for all architectures.

    Parameters:

        str name: Name of the architecture `(field is abstract)`
        bool pretrained: Flag responsible for using pretrained weights
        int unfreezing_epoch: Epoch after which the layers will be unfrozen
        int freezing_start: Layer where freezing starts `(field is abstract)`
        int freezing_stop: Layer where freezing ends `(field is abstract)`

    """

    name: str

    # Pretrained weights
    pretrained: bool = True
    unfreezing_epoch: int = 3
    freezing_start: int
    freezing_stop: int
