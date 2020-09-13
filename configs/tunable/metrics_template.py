from configs import BaseConfig
from src.metrics import Accuracy


class DefaultMetrics(BaseConfig):
    accuracy = dict(metric=Accuracy, plot=True, kwargs={})
