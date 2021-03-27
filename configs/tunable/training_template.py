from typing import Callable, List, Optional, Union

import torch

from pytorch_lightning.profiler import BaseProfiler

from configs.base.base import BaseConfig
from src.base.loss import BaseLoss
from src.callbacks import ModelCheckpoint
from src.losses import CrossEntropyLoss, MSELoss  # noqa
from src.utils.loaders import load_default_sets, load_image_sets  # noqa


class DefaultTraining(BaseConfig):
    """
    Config responsible for setting parameters common for every architecture.

    Parameters:

        int epochs: Number of epochs in training
        int batch_size: Number of elements in one batch
        int seed: Random seed for whole project
        torch.dtype dtype: Type of dats used with model
        gpus:  Which GPUs to train on.
        Callable loader_func: Function loading data from folder ".data"
        BaseLoss loss: Name of loss function
        str experiments_dir: Path to root directory of model experiments
        bool save: Parameter responsible for saving model
        bool test: Flag responsible for calculating test set

        profiler: To profile individual steps during training
            and assist in identifying bottlenecks.

        bool torchsummary: Print a summary from torchvision module
        int summary_depth: depth of layers summary

    """

    # Lengths
    epochs: int = 100
    batch_size: int = 512

    # Environment
    seed: int = 42
    dtype: torch.dtype = torch.float32
    gpus: Optional[Union[List[int], str, int]] = -1

    # Functions
    loader_func: Callable = load_default_sets
    loss: BaseLoss = MSELoss()
    # Saving
    experiments_dir: str = ".data/models"
    save: bool = ModelCheckpoint(monitor="val_accuracy", mode="max")

    # Additional features
    test: bool = True
    profiler: Optional[Union[BaseProfiler, bool]] = True

    # Summary
    torchsummary: bool = True
    summary_depth: int = 4
