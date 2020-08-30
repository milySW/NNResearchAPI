import inspect
from pathlib import Path
from shutil import copy

from configs.training_template import DefaultTraining
from configs.models import DefaultModel


class DefaultConfig:
    def __init__(self, model: DefaultModel, training: DefaultTraining):
        self.model = model
        self.training = training

    def save_configs(self, output_dir: Path):
        for name, attribute in self.__dict__.items():
            config_path = inspect.getfile(attribute.__class__)

            configs_dir = output_dir / "configs"
            configs_dir.mkdir(parents=True, exist_ok=True)

            copy(config_path, configs_dir / (f"{name}.py"))
