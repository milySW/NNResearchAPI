from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import torch

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.trainer.trainer import Trainer
from tabulate import tabulate

from src.metrics import BaseMetric
from src.models.base import LitModel
from src.utils.plots import save_columns_plot


class CalculateMetrics(Callback):
    """
    Callback callculating metrics after every epoch.
    Metrics are saved in output directory "metrics"
    inside experiment directory.

    Output:

        - `metrics_all.csv`: metrics calculated for every epoch
        - `metrics_last.csv`: final metrics

    """

    def __init__(self):
        super().__init__()

        self.preds = []
        self.labels = []
        self.losses = []

        self.subdir = "metrics"
        self.plot_dir = "plots"
        self.index_name = "epoch"
        self.all_file_name = "metrics_all.csv"
        self.last_file_name = "metrics_last.csv"
        self.train_prefix = "train"
        self.valid_prefix = "valid"

        self.metrics_dir: Path = NotImplemented
        self.data_frame: pd.DataFrame = NotImplemented
        self.metrics: Tuple[Dict] = NotImplemented

    @property
    def all_path(self) -> Path:
        return self.metrics_dir / self.all_file_name

    @property
    def last_path(self) -> Path:
        return self.metrics_dir / self.last_file_name

    @property
    def plots(self) -> List[bool]:
        plots = [metric["plot"] for metric in self.metrics.values()]
        return plots + [True] * (len(self.data_frame.columns) - len(plots))

    def initial_load_save_dataframe(self):
        self.data_frame = pd.DataFrame()
        self.data_frame.to_csv(self.all_path)

    def load_save_dataframe(self):
        data_frame = self.data_frame.append(self.series, ignore_index=True)
        data_frame = data_frame.rename_axis(index=self.index_name)
        self.data_frame = data_frame[self.series.index.tolist()]
        self.data_frame.to_csv(self.all_path)

    def calculate(
        self, name: str, metric_data: Dict[str, Tuple[BaseMetric, dict, Any]]
    ):

        metric, _, kwargs = metric_data.values()
        stat = metric(*kwargs)(self.preds, self.labels)
        self.series[name] = round(stat.item(), 4)

    def calculate_metrics(self, prefix: str):
        for name, metric_data in self.metrics.items():
            metric_name = f"{prefix}{name}"
            self.calculate(name=metric_name, metric_data=metric_data)

    def save_final_metrics(self):
        last = self.data_frame.tail(1).rename_axis(index=self.index_name)
        last.to_csv(self.last_path)

    def save_plot(self, df: pd.DataFrame, names: List[str], root_dir: Path):
        save_columns_plot(df, names, root_dir)

    def save_plots(self):
        plot_root_dir = self.metrics_dir.parent / self.plot_dir
        plot_root_dir.mkdir(parents=True, exist_ok=True)

        df = self.data_frame
        group = df.groupby(lambda x: x.rsplit("_", maxsplit=1)[-1], axis=1)

        for names, flag in zip(group.groups.values(), self.plots):
            self.save_plot(df, names, plot_root_dir) if flag else None

    def log_metrics(self, epoch: int, width: Union[float, int] = 12):
        columns_number = len(self.series)
        total_width = columns_number * (width + 2) + 2 * (columns_number - 1)

        names = self.series.keys().values
        values = self.series.values

        if epoch == 0:
            headers = [[f"|{i.center(width)}|" for i in names]]
            print("\n")
            print("=" * total_width)
            print(tabulate(headers, tablefmt="plain"))
            print("=" * total_width)

        metrics = [[f"|{str(i).center(width)}|" for i in values]]
        print("")
        print(tabulate(metrics, tablefmt="plain"))
        print("-" * total_width)

    def initialize_tensors(self):
        self.preds = torch.empty(0)
        self.labels = torch.empty(0)
        self.losses = torch.empty(0)

    def fill_tensors(self, trainer: Trainer):
        self.preds = torch.cat((self.preds, trainer.calculations["preds"]))
        self.labels = torch.cat((self.labels, trainer.calculations["labels"]))
        self.losses = torch.cat((self.losses, trainer.calculations["losses"]))

    def manage_metrics(self, trainer: Trainer, prefix: str):
        self.calculate_metrics(prefix=prefix)
        self.series[f"{prefix}loss"] = round(self.losses.mean().item(), 4)

    @staticmethod
    def sorted_series(series: pd.Series) -> pd.Series:
        cols = series.index.tolist()
        cols = sorted(cols, key=lambda x: x.split("_")[-1])
        return series[cols]

    def on_fit_start(self, trainer: Trainer, pl_module: LitModel):
        self.metrics_dir = Path(trainer.default_root_dir) / self.subdir
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def on_train_start(self, trainer: Trainer, pl_module: LitModel):
        self.metrics = trainer.model.metrics
        self.initial_load_save_dataframe()

    def on_epoch_start(self, trainer, pl_module):
        self.series = pd.Series()

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LitModel):
        self.initialize_tensors()

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LitModel):
        self.initialize_tensors()

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LitModel):
        self.initialize_tensors()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LitModel,
        batch: List[torch.Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ):
        self.fill_tensors(trainer=trainer)
        if trainer.num_training_batches - batch_idx == 1:
            self.manage_metrics(trainer=trainer, prefix="")

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LitModel,
        batch: List[torch.Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ):
        self.fill_tensors(trainer=trainer)
        if trainer.num_val_batches[dataloader_idx] - batch_idx == 1:
            self.manage_metrics(trainer=trainer, prefix="val_")

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LitModel,
        batch: List[torch.Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ):
        self.fill_tensors(trainer=trainer)
        if trainer.num_test_batches[dataloader_idx] - batch_idx == 1:
            self.manage_metrics(trainer=trainer, prefix="test_")

    def on_epoch_end(self, trainer: Trainer, pl_module: LitModel):
        self.series = self.sorted_series(series=self.series)
        self.load_save_dataframe()
        self.log_metrics(epoch=trainer.current_epoch)

    def on_train_end(self, trainer: Trainer, pl_module: LitModel):
        self.save_final_metrics()
        self.save_plots()
