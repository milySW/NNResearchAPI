from __future__ import annotations

import os

from pathlib import Path

from pytorch_lightning import Trainer as PLTrainer
from pytorch_lightning.callbacks.progress import ProgressBar

import configs

from src.utils.collections import filter_class


class Trainer(PLTrainer):
    def __init__(self, config: configs.DefaultConfig, *args, **kwargs):
        self.setup_trainer(config)

        super().__init__(
            logger=self.logger,
            max_epochs=self.epochs,
            default_root_dir=self.root_dir,
            checkpoint_callback=self.checkpoint_callback,
            weights_summary=self.weights_summary,
            gpus=self.gpus,
            profiler=True,
            callbacks=self.callbacks,
            *args,
            **kwargs
        )

    def setup_trainer(self, config: configs.DefaultConfig):
        self.set_params(config)
        config.save_configs(self.root_dir)

    @property
    def set_logger(self):
        return filter_class(self.callbacks, ProgressBar)

    def root(self, config: configs.DefaultConfig) -> Path:
        return Path(config.training.experiments_dir) / config.model.name

    def set_params(self, config: configs.DefaultConfig):
        self.gpus = config.training.gpus
        self.weights_summary = config.training.weights_summary
        self.epochs = config.training.epochs
        self.checkpoint_callback = config.training.checkpoint_callback
        self.callbacks = config.callbacks.value_list()
        self.logger = None if self.set_logger else True
        self.root_dir = self.create_save_path(self.root(config))

    def create_save_path(self, root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        indices = [int(i) for i in os.listdir(root) if i.isdigit()] or [0]
        model_index = max(indices) + 1
        model_path = root / str(model_index)
        return model_path
