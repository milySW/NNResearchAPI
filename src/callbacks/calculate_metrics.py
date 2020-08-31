from pytorch_lightning.callbacks.base import Callback
import torch
import pandas as pd
from pathlib import Path


class CalculateMetrics(Callback):
    def __init__(self):
        self.preds = []
        self.labels = []

    def on_train_start(self, trainer, pl_module):
        sub_dir = "metrics"
        self.metrics_dir = Path(trainer.default_root_dir) / sub_dir

        self.all_path = self.metrics_dir / "metrics_all.csv"
        self.last_path = self.metrics_dir / "metrics_last.csv"

        cols = [metric["name"] for metric in trainer.model.metrics]
        self.data_frame = pd.DataFrame(columns=cols)
        self.data_frame = self.data_frame.rename_axis(index="epoch")

        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.data_frame.to_csv(self.all_path)

    def on_epoch_start(self, trainer, pl_module):
        self.preds = torch.empty(0)
        self.labels = torch.empty(0)

    def on_train_batch_end(
        self, trainer, pl_module, batch, batch_idx: int, dataloader_idx: int
    ):
        self.preds = torch.cat((self.preds, trainer.hiddens["predictions"]))
        self.labels = torch.cat((self.labels, trainer.hiddens["targets"]))

    def on_epoch_end(self, trainer, pl_module):
        series = pd.Series(dtype="str")
        for metric_data in trainer.model.metrics:
            name, metric, kwargs = metric_data.values()
            stat = metric(*kwargs)(self.preds, self.labels)
            series[name] = round(stat.item(), 4)

        self.data_frame = self.data_frame.append(series, ignore_index=True)
        self.data_frame.to_csv(self.all_path, mode="a", header=False)

    def on_fit_end(self, trainer, pl_module):
        last = self.data_frame.tail(1).rename_axis(index="epoch")
        last.to_csv(self.last_path)
