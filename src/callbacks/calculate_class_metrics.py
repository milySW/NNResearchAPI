import pandas as pd
import numpy as np
from pytorch_lightning.trainer.trainer import Trainer

from src.callbacks import CalculateMetrics
from src.models.lightning import LitModel


class CalculateClassMetrics(CalculateMetrics):
    def on_train_start(self, trainer: Trainer, pl_module: LitModel):
        self.classes = len(trainer.train_dataloader.dataset[0][1])
        self.all_group_path = self.metrics_dir / "metrics_group_all.csv"
        self.last_path = self.metrics_dir / "metrics_group_last.csv"

        cols = [metric["name"] for metric in trainer.model.metrics]
        cols = cols * self.classes
        cols = [f"{name}_{i%self.classes}" for i, name in enumerate(cols)]

        self.data_frame = pd.DataFrame(columns=cols)
        self.data_frame = self.data_frame.rename_axis(index="epoch")
        self.data_frame.to_csv(self.all_group_path)

    def on_epoch_end(self, trainer: Trainer, pl_module: LitModel):
        series = pd.Series(dtype="str")
        for metric_data in trainer.model.metrics:
            name, metric, kwargs = metric_data.values()

            for group in range(self.classes):
                indices = np.where(self.labels == group)
                preds = self.preds[indices]
                labels = self.labels[indices]
                stat = metric(num_classes=self.classes, *kwargs)(preds, labels)
                series[f"{name}_{group}"] = round(stat.item(), 4)

        self.data_frame = self.data_frame.append(series, ignore_index=True)
        self.data_frame.to_csv(self.all_group_path, mode="a", header=False)
