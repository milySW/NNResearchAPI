from pathlib import Path
from typing import List

from configs import DefaultConfig
from src.utils.loaders import load_variable


def get_config(paths: List[Path]) -> DefaultConfig:
    params = dict()

    for path in paths:
        name = path.name.split(".")[0]

        if name == "__pycache__":
            continue

        key = name
        name = name if name != "model" else "Resnet"

        params[key] = load_variable(f"Default{name.capitalize()}", path)

    config = DefaultConfig(**params)

    return config
