from typing import Any, Dict, List, Tuple

import numpy as np

from pytorch_lightning.trainer.trainer import Trainer

from src.callbacks.calculate_metrics import CalculateMetrics
from src.metrics import BaseMetric
from src.models.base import LitModel


class CalculateClassMetrics(CalculateMetrics):
    """
    Callbck callculating metrics per class after every epoch.
    Metrics are saved in output directory "metrics"
    inside experiment directory.

    Output:

        - `metrics_group_all.csv`: Metrics calculated for every epoch
        - `metrics_group_last.csv`: Final metrics

    """

    def __init__(self):
        super().__init__()
        self.all_file_name = "metrics_group_all.csv"
        self.last_file_name = "metrics_group_last.csv"

        self.classes: int = NotImplemented

    @property
    def cols(self) -> List[str]:
        cols = self.metrics.keys()
        cols = cols * self.classes
        cols = [f"{name}_{i%self.classes}" for i, name in enumerate(cols)]
        return cols

    @property
    def plots(self) -> List[bool]:
        plots = [metric["plot"] for metric in self.metrics.values()]
        return plots * self.classes

    def calculate_metric(
        self, metric: BaseMetric, kwargs: Dict[str, Any], group: int
    ) -> float:

        indices = np.where(self.labels == group)
        preds = self.preds[indices]
        labels = self.labels[indices]

        stat = metric(num_classes=self.classes, *kwargs)(preds, labels)
        return round(stat.item(), 4)

    def calculate(
        self, name: str, metric_data: Dict[str, Tuple[BaseMetric, dict, Any]]
    ):

        metric, _, kwargs = metric_data.values()
        for group in range(self.classes):
            self.series[f"{name}_{group}"] = self.calculate_metric(
                metric=metric, kwargs=kwargs, group=group,
            )

    def on_train_start(self, trainer: Trainer, pl_module: LitModel):
        _, y = trainer.train_dataloader.dataset.pop()
        self.classes = len(y)

        super().on_train_start(trainer=trainer, pl_module=pl_module)

    def manage_metrics(self, trainer: Trainer, prefix: str):
        self.calculate_metrics(prefix=prefix)

    def log_metrics(self, **kwargs):
        pass
