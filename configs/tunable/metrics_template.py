from configs.base.base import BaseConfig
from src.metrics import F1, Accuracy, Precision, Recall


class DefaultMetrics(BaseConfig):
    """
    Config responsible for passing metrics of :class:`BaseMetric`.
    Providing new metrics require adding new class field as dict with any name

    Underneath description of field parameters

    Parameters:

        BaseMetric metric: metric class
        bool plot: flag responsible for plotting final metrics
        dict kwargs: dict with parameters for metric function

    """

    accuracy = dict(metric=Accuracy, plot=True, kwargs={})
    precision = dict(metric=Precision, plot=True, kwargs={},)
    recall = dict(metric=Recall, plot=True, kwargs={})
    f1_score = dict(metric=F1, plot=True, kwargs={},)
