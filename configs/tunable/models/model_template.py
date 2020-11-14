from configs.base.base import BaseConfig


class DefaultModel(BaseConfig):
    """
    Main config responsible for setting parameters for all architectures.

    Parameters:

        str name: Name of the architecture `(field is abstract)`
        int data_dim: Dimension of provided data.

        bool pretrained: Flag responsible for using pretrained weights
        int unfreezing_epoch: Epoch after which the layers will be unfrozen
        int freezing_start: Layer where freezing starts `(field is abstract)`
        int freezing_stop: Layer where freezing ends `(field is abstract)`

    """

    name: str
    data_dim: str

    # Pretrained weights
    pretrained: bool = False
    unfreezing_epoch: int = 3
    freezing_start: int
    freezing_stop: int
