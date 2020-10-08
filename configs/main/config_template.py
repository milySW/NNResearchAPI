import inspect

from pathlib import Path
from shutil import copy
from typing import Optional

from configs.base.base import BaseConfig
from configs.base.base_optimizer_config import DefaultOptimizersAndSchedulers
from configs.tunable.callbacks_template import DefaultCallbacks
from configs.tunable.evaluation_template import DefaultEvaluation
from configs.tunable.metrics_template import DefaultMetrics
from configs.tunable.models.model_template import DefaultModel
from configs.tunable.prediction_template import DefaultPrediction
from configs.tunable.training_template import DefaultTraining


class DefaultConfig(BaseConfig):
    def __init__(
        self,
        model: DefaultModel,
        training: DefaultTraining,
        optimizers: DefaultOptimizersAndSchedulers,
        metrics: DefaultMetrics,
        callbacks: DefaultCallbacks,
        prediction: Optional[DefaultPrediction],
        evaluations: Optional[DefaultEvaluation],
    ):
        self.model = model
        self.training = training
        self.optimizers = optimizers
        self.metrics = metrics
        self.callbacks = callbacks
        self.prediction = prediction
        self.evaluations = evaluations

    def save_configs(self, output_dir: Path):
        for name, attribute in self.__dict__.items():
            config_path = inspect.getfile(attribute)

            configs_dir = output_dir / "configs"
            configs_dir.mkdir(parents=True, exist_ok=True)

            copy(config_path, configs_dir / (f"{name}.py"))
