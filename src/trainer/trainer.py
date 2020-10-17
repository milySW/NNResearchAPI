import os

from pathlib import Path

from pytorch_lightning import Trainer as PLTrainer
from pytorch_lightning.callbacks.progress import ProgressBar

from configs import DefaultConfig
from src.utils.collections import filter_class


class Trainer(PLTrainer):
    """
    Customize every aspect of training via flags

    Parameters:
        int max_epochs: Number of epochs in training
        str default_root_dir: Path to root directory of model experiments
        bool checkpoint_callback: Parameter responsible for saving model
        list callbacks: List of callbacks used with training
        bool logger: Flag responsible for default progress bar
        gpus:  Which GPUs to train on.

        profiler: To profile individual steps during training
            and assist in identifying bottlenecks.

        str weigths_summary: Prints a summary of the weights
            when training begins. Supported options:

            - `top`: only the top-level modules will be recorded
            - `full`: summarizes all layers and their submodules
    """

    def __init__(self, config: DefaultConfig, **kwargs):
        self.setup_trainer(config)

        super().__init__(
            max_epochs=self.epochs,
            default_root_dir=self.root_dir,
            checkpoint_callback=self.checkpoint_callback,
            callbacks=self.callbacks,
            logger=self.logger,
            gpus=self.gpus,
            profiler=True,
            weights_summary=self.weights_summary,
            **kwargs
        )

    def setup_trainer(self, config: DefaultConfig):
        self.set_params(config)
        config.save_configs(self.root_dir)

    @property
    def set_logger(self):
        return filter_class(self.callbacks, ProgressBar)

    def root(self, config: DefaultConfig) -> Path:
        return Path(config.training.experiments_dir) / config.model.name

    def set_params(self, config: DefaultConfig):
        self.gpus = config.training.gpus
        self.weights_summary = config.training.weights_summary
        self.epochs = config.training.epochs
        self.checkpoint_callback = config.training.save
        self.callbacks = config.callbacks.value_list()
        self.logger = None if self.set_logger else True
        self.root_dir = self.create_save_path(self.root(config))

    def create_save_path(self, root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        indices = [int(i) for i in os.listdir(root) if i.isdigit()] or [0]
        model_index = max(indices) + 1
        model_path = root / str(model_index)
        return model_path
