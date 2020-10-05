import itertools

from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List


def flatten(iterable: List[List[Any]]) -> itertools.chain:
    return itertools.chain.from_iterable(iterable)


def filter_class(iterable: List[Any], class_type: Any) -> bool:
    return any(filter(lambda x: issubclass(x.__class__, class_type), iterable))


def add_path_to_dict(data: Dict[str, Path], string: str) -> Dict[str, Path]:
    return {key: path / string for key, path in data.items()}


def ignore_none(data: Iterable) -> Iterable:
    return [i for i in data if i is not None]


def stripped_data(data: Iterable) -> Iterable:
    return [i.strip() for i in data]


def unpack_paths(
    root_path: str, experiments: str, supported_files: str
) -> Generator[Dict[str, Path], str, None]:

    supported_files = stripped_data(supported_files.split(","))
    experiments = stripped_data(experiments.split(","))

    roots = {exp: root_path / exp / "metrics" for exp in experiments}
    return (add_path_to_dict(roots, filename) for filename in supported_files)
