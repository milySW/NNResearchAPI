import configs

from src.callbacks import (
    CollectBestClassMetrics,
    CollectBestMetrics,
    LightProgressBar,
)


class DefaultCallbacks(configs.BaseConfig):
    """
    Config responsible for passing callbacks of :class:`BaseCallback` type.
    Providing new callbacks require adding new class field field with any name
    """

    bar = LightProgressBar()
    best_class_metric = CollectBestClassMetrics()
    best_metric = CollectBestMetrics()
