from configs.base.base import BaseConfig
from src.callbacks import (
    CollectBestClassMetrics,
    CollectBestMetrics,
    EarlyStopping,
    LightProgressBar,
)


class DefaultCallbacks(BaseConfig):
    """
    Config responsible for passing callbacks of :class:`BaseCallback`.
    Providing new callbacks require adding new class field with any name
    """

    bar = LightProgressBar()
    e_stopping = EarlyStopping(monitor="val_accuracy", patience=5, mode="max")

    best_class_metric = CollectBestClassMetrics(variants=["val", "test"])
    best_metric = CollectBestMetrics(variants=["val", "test"])
