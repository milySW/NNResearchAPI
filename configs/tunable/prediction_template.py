from configs.base.base import BaseConfig


class DefaultPrediction(BaseConfig):
    """
    Config responsible for setting prediction parameters.

    Parameters:

        int batch_size: Number of elements in one batch
    """

    batch_size: int = 500
