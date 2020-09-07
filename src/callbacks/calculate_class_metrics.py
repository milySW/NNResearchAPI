import pandas as pd

import numpy as np
from pytorch_lightning.trainer.trainer import Trainer

from src.callbacks import CalculateMetrics
from src.models.lightning import LitModel
from src.metrics import BaseMetric


class CalculateClassMetrics(CalculateMetrics):
    def __init__(self):
        super().__init__()
        self.all_file_name = "metrics_group_all.csv"
        self.last_file_name = "metrics_group_last.csv"

        self.classes: int = NotImplemented

    @property
    def cols(self):
        cols = [metric["name"] for metric in self.metrics]
        cols = cols * self.classes
        cols = [f"{name}_{i%self.classes}" for i, name in enumerate(cols)]
        return cols

    @property
    def plots(self):
        plots = [metric["plot"] for metric in self.metrics]
        return plots * self.classes

    def calculate_metric(self, metric: BaseMetric, kwargs: dict, group: int):
        indices = np.where(self.labels == group)
        preds = self.preds[indices]
        labels = self.labels[indices]

        stat = metric(num_classes=self.classes, *kwargs)(preds, labels)
        return round(stat.item(), 4)

    def on_train_start(self, trainer: Trainer, pl_module: LitModel):
        _, y = trainer.train_dataloader.dataset.pop()
        self.classes = len(y)

        super().on_train_start(trainer=trainer, pl_module=pl_module)

    def on_epoch_end(self, trainer: Trainer, pl_module: LitModel):
        series = pd.Series(dtype="str")
        for metric_data in trainer.model.metrics:
            name, metric, kwargs, _ = metric_data.values()

            for group in range(self.classes):
                series[f"{name}_{group}"] = self.calculate_metric(
                    metric=metric, kwargs=kwargs, group=group,
                )

        self.load_save_dataframe(series=series)
