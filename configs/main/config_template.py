import inspect

from pathlib import Path
from shutil import copy
from typing import Optional

from configs import (
    BaseConfig,
    DefaultCallbacks,
    DefaultEvaluations,
    DefaultMetrics,
    DefaultModel,
    DefaultOptimizersAndSchedulers,
    DefaultTraining,
)


class DefaultConfig(BaseConfig):
    def __init__(
        self,
        model: DefaultModel,
        training: DefaultTraining,
        optimizers: DefaultOptimizersAndSchedulers,
        metrics: DefaultMetrics,
        callbacks: DefaultCallbacks,
        evaluations: Optional[DefaultEvaluations],
    ):
        self.model = model
        self.training = training
        self.optimizers = optimizers
        self.metrics = metrics
        self.callbacks = callbacks
        self.evaluations = evaluations

    def save_configs(self, output_dir: Path):
        for name, attribute in self.__dict__.items():
            config_path = inspect.getfile(attribute)

            configs_dir = output_dir / "configs"
            configs_dir.mkdir(parents=True, exist_ok=True)

            copy(config_path, configs_dir / (f"{name}.py"))
