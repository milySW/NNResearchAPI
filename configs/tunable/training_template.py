from typing import Callable, List, Optional, Union

import torch

from pytorch_lightning.profiler import BaseProfiler

from configs.base.base import BaseConfig
from src.losses import BaseLoss, CrossEntropyLoss
from src.utils.loaders import load_default_sets


class DefaultTraining(BaseConfig):
    """
    Config responsible for setting parameters common for every architecture.

    Parameters:

        int epochs: Number of epochs in training
        int batch_size: Number of elements in one batch
        bool save: Parameter responsible for saving model
        int seed: Random seed for whole project
        torch.dtype dtype: Type of dats used with model
        Callable loader_func: Function loading data from folder ".data"
        BaseLoss loss: Name of loss function
        str experiments_dir: Path to root directory of model experiments
        bool test: Flag responsible for calculating test set

        str weigths_summary: Prints a summary of the weights
            when training begins. Supported options:

            - `top`: only the top-level modules will be recorded
            - `full`: summarizes all layers and their submodules

        gpus:  Which GPUs to train on.

        profiler: To profile individual steps during training
            and assist in identifying bottlenecks.


    """

    epochs: int = 30
    batch_size: int = 512
    save: bool = True
    seed: int = 42
    dtype: torch.dtype = torch.float32
    loader_func: Callable = load_default_sets
    loss: BaseLoss = CrossEntropyLoss()
    experiments_dir: str = ".data/models"
    test: bool = True
    weights_summary: Optional[str] = "full"
    gpus: Optional[Union[List[int], str, int]] = -1
    profiler: Optional[Union[BaseProfiler, bool]] = True
