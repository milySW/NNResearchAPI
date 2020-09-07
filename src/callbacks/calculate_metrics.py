from pathlib import Path
from typing import List, Tuple, Dict

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.trainer.trainer import Trainer
import torch
import pandas as pd
import matplotlib.pyplot as plt

from src.models.lightning import LitModel


class CalculateMetrics(Callback):
    def __init__(self):
        super().__init__()

        self.preds = []
        self.labels = []

        self.subdir = "metrics"
        self.plot_dir = "plots"
        self.index_name = "epoch"
        self.all_file_name = "metrics_all.csv"
        self.last_file_name = "metrics_last.csv"

        self.metrics_dir: Path = NotImplemented
        self.data_frame: pd.DataFrame = NotImplemented
        self.metrics: Tuple[Dict] = NotImplemented

    @property
    def all_path(self):
        return self.metrics_dir / self.all_file_name

    @property
    def last_path(self):
        return self.metrics_dir / self.last_file_name

    @property
    def cols(self):
        return [metric["name"] for metric in self.metrics]

    @property
    def plots(self):
        return [metric["plot"] for metric in self.metrics]

    def initial_load_save_dataframe(self):
        self.data_frame = pd.DataFrame()
        self.data_frame.to_csv(self.all_path)

    def load_save_dataframe(self, series: pd.Series):
        self.data_frame = self.data_frame.append(series, ignore_index=True)
        self.data_frame = self.data_frame.rename_axis(index=self.index_name)
        self.data_frame.to_csv(self.all_path)

    def save_final_metrics(self):
        last = self.data_frame.tail(1).rename_axis(index=self.index_name)
        last.to_csv(self.last_path)

    def plot_metric(self, metric_name: str):
        plot_root_dir = self.metrics_dir.parent / self.plot_dir
        plot_root_dir.mkdir(parents=True, exist_ok=True)

        plt.figure()
        plt.plot(self.data_frame[metric_name])
        plt.savefig(plot_root_dir / f"{metric_name}.png", transparent=True)

    def plot_metrics(self):
        for name, flag in zip(self.cols, self.plots):
            self.plot_metric(metric_name=name) if flag else None

    def on_fit_start(self, trainer: Trainer, pl_module: LitModel):
        self.metrics_dir = Path(trainer.default_root_dir) / self.subdir
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def on_train_start(self, trainer: Trainer, pl_module: LitModel):
        self.metrics = trainer.model.metrics
        self.initial_load_save_dataframe()

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
        for metric_data in self.metrics:
            name, metric, kwargs, _ = metric_data.values()
            stat = metric(*kwargs)(self.preds, self.labels)
            series[name] = round(stat.item(), 4)

        self.load_save_dataframe(series=series)

    def on_train_end(self, trainer: Trainer, pl_module: LitModel):
        self.save_final_metrics()
        self.plot_metrics()
