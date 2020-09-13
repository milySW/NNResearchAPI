from configs import BaseConfig
from src.callbacks import (
    CollectBestMetrics,
    CollectBestClassMetrics,
)


class DefaultCallbacks(BaseConfig):
    best_class_metric = CollectBestClassMetrics()
    best_metric = CollectBestMetrics()
