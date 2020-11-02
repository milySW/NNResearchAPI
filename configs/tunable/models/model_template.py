from configs.base.base import BaseConfig


class DefaultModel(BaseConfig):
    """
    Main config responsible for setting parameters for all architectures.

    Parameters:

        str name: Name of the architecture (this field is abstract)
        bool pretrained: Flag responsible for using pretrained weights
        int unfreezing_epoch: Epoch after which the layers will be unfrozen

    """

    name: str

    # Additional features
    pretrained: bool = True
    unfreezing_epoch: int = 5
