from typing import Callable, List, Optional, Union

from pytorch_lightning.profiler import BaseProfiler

import configs

from src.losses import BaseLoss, CrossEntropyLoss
from src.utils.loaders import load_default_sets


class DefaultTraining(configs.BaseConfig):
    """
    Config responsible for setting parameters common for every architecture.

    Parameters:

        int epochs: Number of epochs in training
        int batch_size: Number of elements in one batch
        bool checkpoint_callback: Parameter responsible for saving model
        int seed: Random seed for whole project
        Callable loader_func: Function loading data from folder ".data"
        BaseLoss loss: Name of loss function
        str experiments_dir: path to root directory of model experiments

        str weigths_summary: Prints a summary of the weights
            when training begins. Supported options:

            - `top`: only the top-level modules will be recorded
            - `full`: summarizes all layers and their submodules

        gpus:  Which GPUs to train on.

        profiler: To profile individual steps during training
            and assist in identifying bottlenecks.


    """

    epochs: int = 2
    batch_size: int = 128
    checkpoint_callback: bool = False
    seed: int = 42
    loader_func: Callable = load_default_sets
    loss: BaseLoss = CrossEntropyLoss()
    experiments_dir: str = ".data/models"
    weights_summary: Optional[str] = "full"
    gpus: Optional[Union[List[int], str, int]] = -1
    profiler: Optional[Union[BaseProfiler, bool]] = True
