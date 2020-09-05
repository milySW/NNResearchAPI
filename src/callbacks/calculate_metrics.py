from pathlib import Path
from typing import List

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.trainer.trainer import Trainer
import torch
import pandas as pd

from src.models.lightning import LitModel


class CalculateMetrics(Callback):
    def __init__(self):
        super().__init__()

        self.preds = []
        self.labels = []

        self.subdir = "metrics"
        self.index_name = "epoch"
        self.all_file_name = "metrics_all.csv"
        self.last_file_name = "metrics_last.csv"

        self.metrics_dir: Path = NotImplemented
        self.data_frame: pd.DataFrame = NotImplemented

    @property
    def all_path(self):
        return self.metrics_dir / self.all_file_name

    @property
    def last_path(self):
        return self.metrics_dir / self.last_file_name

    def initial_load_save_dataframe(self, cols: List[str]):
        self.data_frame = pd.DataFrame(columns=cols)
        self.data_frame.to_csv(self.all_path)

    def load_save_dataframe(self, series: pd.Series):
        self.data_frame = self.data_frame.append(series, ignore_index=True)
        self.data_frame = self.data_frame.rename_axis(index=self.index_name)
        self.data_frame.to_csv(self.all_path)

    def save_final_metrics(self):
        last = self.data_frame.tail(1).rename_axis(index=self.index_name)
        last.to_csv(self.last_path)

    def on_fit_start(self, trainer: Trainer, pl_module: LitModel):
        self.metrics_dir = Path(trainer.default_root_dir) / self.subdir
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def on_train_start(self, trainer: Trainer, pl_module: LitModel):
        cols = [metric["name"] for metric in trainer.model.metrics]
        self.initial_load_save_dataframe(cols=cols)

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

        self.load_save_dataframe(series=series)

    def on_train_end(self, trainer: Trainer, pl_module: LitModel):
        self.save_final_metrics()
