from configs import BaseConfig
from src.metrics import Accuracy


class DefaultMetrics(BaseConfig):
    """
    Config responsible for passing metrics of :class:`BaseMetric` type.
    Providing new metrics require adding new class field as dict with any name

    Underneath description of field parameters

    Parameters:

        BaseMetric metric: metric class
        bool plot: flag responsible for plotting final metrics
        dict kwargs: dict with parameters for metric function

    """

    accuracy = dict(metric=Accuracy, plot=True, kwargs={})
