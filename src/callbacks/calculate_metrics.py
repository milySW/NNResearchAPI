from pathlib import Path
from typing import List

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.trainer.trainer import Trainer
import torch
import pandas as pd

from src.models.lightning import LitModel


class CalculateMetrics(Callback):
    def __init__(self):
        self.preds = []
        self.labels = []

    def on_fit_start(self, trainer: Trainer, pl_module: LitModel):
        sub_dir = "metrics"
        self.metrics_dir = Path(trainer.default_root_dir) / sub_dir

        self.all_path = self.metrics_dir / "metrics_all.csv"
        self.last_path = self.metrics_dir / "metrics_last.csv"

        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def on_train_start(self, trainer: Trainer, pl_module: LitModel):
        cols = [metric["name"] for metric in trainer.model.metrics]
        self.data_frame = pd.DataFrame(columns=cols)
        self.data_frame = self.data_frame.rename_axis(index="epoch")
        self.data_frame.to_csv(self.all_path)

    def on_epoch_start(self, trainer: Trainer, pl_module: LitModel):
        self.preds = torch.empty(0)
        self.labels = torch.empty(0)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LitModel,
        batch: List[torch.Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ):
        self.preds = torch.cat((self.preds, trainer.hiddens["predictions"]))
        self.labels = torch.cat((self.labels, trainer.hiddens["targets"]))

    def on_epoch_end(self, trainer: Trainer, pl_module: LitModel):
        series = pd.Series(dtype="str")
        for metric_data in trainer.model.metrics:
            name, metric, kwargs = metric_data.values()
            stat = metric(*kwargs)(self.preds, self.labels)
            series[name] = round(stat.item(), 4)

        self.data_frame = self.data_frame.append(series, ignore_index=True)
        self.data_frame.to_csv(self.all_path, mode="a", header=False)

    def on_train_end(self, trainer: Trainer, pl_module: LitModel):
        last = self.data_frame.tail(1).rename_axis(index="epoch")
        last.to_csv(self.last_path)
