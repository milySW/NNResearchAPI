from configs.base.base import BaseConfig
from src.callbacks import (
    CollectBestClassMetrics,
    CollectBestMetrics,
    LightProgressBar,
)


class DefaultCallbacks(BaseConfig):
    """
    Config responsible for passing callbacks of :class:`BaseCallback`.
    Providing new callbacks require adding new class field with any name
    """

    bar = LightProgressBar()
    best_class_metric = CollectBestClassMetrics()
    best_metric = CollectBestMetrics()
