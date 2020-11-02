import itertools

from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterable, List


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


def batch_generator(
    iterable: Iterable, batch_size: int
) -> Generator[Iterable, None, None]:

    length = len(iterable)
    for start in range(0, length, batch_size):
        yield iterable[slice(start, min(start + batch_size, length))]


def batch_list(iterable: Iterable, batch_size: int) -> List[Iterable]:
    return list(batch_generator(iterable=iterable, batch_size=batch_size))


def check_prefix(string: str, prefix_list: List[str]) -> bool:
    return any([string.startswith(prefix) for prefix in prefix_list])


def filter_list(
    data: List[Any], filters: List[Any], filtering_func: Callable
) -> List[Any]:

    return [var for var in data if filtering_func(var, filters)]


def filter_by_prefix(data: List[str], prefixes: List[str]) -> List[str]:
    func = check_prefix
    return filter_list(data=data, filters=prefixes, filtering_func=func)


def split(name: str, start: int, stop: int, delimeter=".") -> str:
    return delimeter.join(name.split(delimeter)[start:stop])


def unique_keys(collection: Iterable[str]) -> List[Any]:
    return list(OrderedDict.fromkeys(collection))
