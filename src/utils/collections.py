from typing import Any, List


def filter_class(iterable: List[Any], class_type: Any):
    return any(filter(lambda x: issubclass(x.__class__, class_type), iterable))
