from configs import BaseConfig
from src.callbacks import (
    CollectBestMetrics,
    CollectBestClassMetrics,
)


class DefaultCallbacks(BaseConfig):
    """
    Config responsible for passing callbacks of :class:`BaseCallback` type.
    Providing new callbacks require adding new class field field with any name
    """

    best_class_metric = CollectBestClassMetrics()
    best_metric = CollectBestMetrics()
