from pytorch_lightning.metrics.classification import Accuracy as PLAccuracy
from src.metrics import BaseMetric


class Accuracy(BaseMetric, PLAccuracy):
    __doc__ = PLAccuracy.__doc__
    extremum = "max"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
