from pytorch_lightning.metrics.classification import Accuracy as PLAccuracy

from src.base.metric import BaseMetric


class Accuracy(BaseMetric, PLAccuracy):
    __doc__ = PLAccuracy.__doc__
    extremum = "max"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
