import os
from pathlib import Path


def create_save_path(output_path: Path, file_name: str) -> Path:
    root = Path(output_path) / file_name
    root.mkdir(parents=True, exist_ok=True)
    indices = [int(i) for i in os.listdir(root) if i.isdigit()] or [0]
    model_index = max(indices) + 1
    model_path = root / str(model_index)
    return model_path
