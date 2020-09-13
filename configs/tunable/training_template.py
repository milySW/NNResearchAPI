from typing import Callable

from configs import BaseConfig
from src.losses import BaseLoss, CrossEntropyLoss
from src.utils.loaders import load_default_sets


class DefaultTraining(BaseConfig):
    epochs: int = 2
    batch_size: int = 128
    checkpoint_callback: bool = False
    seed: int = 42
    loader_func: Callable = load_default_sets
    loss: BaseLoss = CrossEntropyLoss()
