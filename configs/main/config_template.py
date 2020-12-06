import inspect

from pathlib import Path
from shutil import copy

from configs.base.base import BaseConfig
from configs.base.base_hooks_config import DefaultBindedHooks
from configs.base.base_optimizer_config import DefaultOptimizersAndSchedulers
from configs.tunable.augmentations_template import DefaultAugmentations
from configs.tunable.callbacks_template import DefaultCallbacks
from configs.tunable.evaluation_template import DefaultEvaluations
from configs.tunable.metrics_template import DefaultMetrics
from configs.tunable.models.model_template import DefaultModel
from configs.tunable.postprocessors_template import DefaultPostprocessors
from configs.tunable.prediction_template import DefaultPrediction
from configs.tunable.preprocessors_template import DefaultPreprocessors
from configs.tunable.training_template import DefaultTraining


class DefaultConfig(BaseConfig):
    def __init__(
        self,
        model: DefaultModel,
        training: DefaultTraining,
        optimizers: DefaultOptimizersAndSchedulers,
        metrics: DefaultMetrics,
        callbacks: DefaultCallbacks,
        hooks: DefaultBindedHooks,
        preprocessors: DefaultPreprocessors,
        augmentations: DefaultAugmentations,
        postprocessors: DefaultPostprocessors,
        prediction: DefaultPrediction,
        evaluations: DefaultEvaluations,
    ):
        self.model = model
        self.training = training
        self.optimizers = optimizers
        self.metrics = metrics
        self.callbacks = callbacks
        self.hooks = hooks
        self.preprocessors = preprocessors
        self.augmentations = augmentations
        self.postprocessors = postprocessors
        self.prediction = prediction
        self.evaluations = evaluations

    def save_configs(self, output_dir: Path):
        for name, attribute in self.__dict__.items():
            config_path = inspect.getfile(attribute)

            configs_dir = output_dir / "configs"
            configs_dir.mkdir(parents=True, exist_ok=True)

            copy(config_path, configs_dir / (f"{name}.py"))
