from pathlib import Path
from shutil import rmtree, copytree

import pandas as pd
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks.base import Callback

from src.callbacks import CalculateMetrics, CalculateClassMetrics
from src.models.lightning import LitModel


class CollectBest:
    def __init__(self):
        super().__init__()

        self.best_index = "best"
        self.history_filename = "history.csv"
        self.best_dir_path = Path("best_metrics")

        self.metrics_dir = NotImplemented

    def check_conflicts(self, trainer: Trainer, callback: Callback):
        condition = any(type(i) == callback for i in trainer.callbacks)
        info = "Don't use {} wrapper with parent callback {}"
        names = self.__class__.__name__, callback.__name__
        assert not condition, info.format(*names)

    def get_paths(self, model_root_dir: Path, name: str):
        metric_subdir = self.best_dir_path / f"{self.best_index}_{name}"
        best_metric_dir = model_root_dir / metric_subdir
        best_file = best_metric_dir / self.subdir / self.last_file_name

        return best_metric_dir, best_file

    def get_new_metric(self, data: dict, root_dir: Path):
        new_metric = pd.DataFrame(data=data)
        new_metric = new_metric.rename_axis(index=self.best_index)
        return new_metric

    def add_new_metric(self, new_metric: pd.DataFrame, best_metric_dir: Path):
        hist = pd.read_csv(best_metric_dir / self.history_filename)
        hist = hist.set_index(self.best_index)

        hist = pd.concat([new_metric, hist], ignore_index=True,)
        return hist.rename_axis(index=self.best_index)

    def create_best_dir(self, history: pd.DataFrame, best_dir: Path):
        copytree(self.last_path.parent.parent, best_dir)
        history.to_csv(best_dir / self.history_filename)

    def replace_best_dir(self, history: pd.DataFrame, best_dir: Path):
        rmtree(best_dir)
        copytree(self.last_path.parent.parent, best_dir)
        history.to_csv(best_dir / self.history_filename)

    def prepare_vars(self, name: str):
        stat = float(self.data_frame.tail(1)[name].values.item())
        root_dir = self.metrics_dir.parent.parent
        best_metric_dir, best_file = self.get_paths(root_dir, name)

        data = {"model_index": [self.metrics_dir.parent.name], name: [stat]}
        new = self.get_new_metric(data=data, root_dir=self.metrics_dir.parent)

        return stat, new, best_metric_dir, best_file

    def collect(
        self,
        name: str,
        stat: float,
        new_metric: pd.DataFrame,
        best_file: Path,
        best_metric_dir: Path,
    ):
        best_metrics = pd.read_csv(best_file)
        assert name in best_metrics.columns, f"Column {name} doesn't exist"

        if best_metrics[name].values.item() < stat:
            history = self.add_new_metric(new_metric, best_metric_dir)
            self.replace_best_dir(history, best_metric_dir)

    def collect_best_metric(self, name: str):
        stat, new, best_metric_dir, best_file = self.prepare_vars(name)

        if best_file.is_file():
            self.collect(
                name=name,
                stat=stat,
                new_metric=new,
                best_file=best_file,
                best_metric_dir=best_metric_dir,
            )

        else:
            self.create_best_dir(history=new, best_dir=best_metric_dir)

    def collect_best_metrics(self, trainer: Trainer):
        self.save_final_metrics()

        for name in self.data_frame.columns:
            self.collect_best_metric(name=name)


class CollectBestMetrics(CalculateMetrics, CollectBest):
    def __init__(self):
        super().__init__()

    def on_fit_start(self, trainer: Trainer, pl_module: LitModel):
        super().on_fit_start(trainer, pl_module)
        self.check_conflicts(trainer=trainer, callback=CalculateMetrics)

    def on_fit_end(self, trainer: Trainer, pl_module: LitModel):
        self.collect_best_metrics(trainer=trainer)


class CollectBestClassMetrics(CalculateClassMetrics, CollectBest):
    def __init__(self):
        super().__init__()

    def on_fit_start(self, trainer: Trainer, pl_module: LitModel):
        super().on_fit_start(trainer, pl_module)
        self.check_conflicts(trainer=trainer, callback=CalculateClassMetrics)

    def on_fit_end(self, trainer: Trainer, pl_module: LitModel):
        self.collect_best_metrics(trainer=trainer)
