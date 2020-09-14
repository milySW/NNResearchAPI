from typing import Callable

from configs import BaseConfig
from src.losses import BaseLoss, CrossEntropyLoss
from src.utils.loaders import load_default_sets


class DefaultTraining(BaseConfig):
    """
    Config responsible for setting parameters common for every architecture.

    Parameters:

        int epochs: Number of epochs in training
        int batch_size: Number of elements in one batch
        bool checkpoint_callback: Parameter responsible for saving model
        int seed: Random seed for whole project
        Callable loader_func: Function loading data from folder ".data"
        BaseLoss loss: Name of loss function

    """

    epochs: int = 2
    batch_size: int = 128
    checkpoint_callback: bool = False
    seed: int = 42
    loader_func: Callable = load_default_sets
    loss: BaseLoss = CrossEntropyLoss()
