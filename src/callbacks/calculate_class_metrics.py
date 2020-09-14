from typing import Dict, Tuple, Any

import pandas as pd
import numpy as np
from pytorch_lightning.trainer.trainer import Trainer

from src.callbacks import CalculateMetrics
from src.models.base import LitModel
from src.metrics import BaseMetric


class CalculateClassMetrics(CalculateMetrics):
    """
    Callbck callculating metrics per class after every epoch.
    Metrics are saved in output directory "metrics"
    inside experiment directory.

    Output:

        - `metrics_group_all.csv`: metrics calculated for every epoch
        - `metrics_group_last.csv`: final metrics

    """

    def __init__(self):
        super().__init__()
        self.all_file_name = "metrics_group_all.csv"
        self.last_file_name = "metrics_group_last.csv"

        self.classes: int = NotImplemented

    @property
    def cols(self):
        cols = self.metrics.keys()
        cols = cols * self.classes
        cols = [f"{name}_{i%self.classes}" for i, name in enumerate(cols)]
        return cols

    @property
    def plots(self):
        plots = [metric["plot"] for metric in self.metrics.values()]
        return plots * self.classes

    def calculate_metric(self, metric: BaseMetric, kwargs: dict, group: int):
        indices = np.where(self.labels == group)
        preds = self.preds[indices]
        labels = self.labels[indices]

        stat = metric(num_classes=self.classes, *kwargs)(preds, labels)
        return round(stat.item(), 4)

    def calculate(
        self,
        series: pd.Series,
        name: str,
        metric_data: Dict[str, Tuple[BaseMetric, dict, Any]],
    ) -> pd.Series:

        metric, _, kwargs = metric_data.values()
        for group in range(self.classes):
            series[f"{name}_{group}"] = self.calculate_metric(
                metric=metric, kwargs=kwargs, group=group,
            )
        return series

    def on_train_start(self, trainer: Trainer, pl_module: LitModel):
        _, y = trainer.train_dataloader.dataset.pop()
        self.classes = len(y)

        super().on_train_start(trainer=trainer, pl_module=pl_module)

    def on_epoch_end(self, trainer: Trainer, pl_module: LitModel):
        series = self.calculate_metrics()
        self.load_save_dataframe(series=series)
