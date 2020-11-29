import operator

from pathlib import Path
from shutil import copytree, rmtree
from typing import List, Tuple

import pandas as pd

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.trainer.trainer import Trainer
from torch.nn import Module

from src.base.losses import BaseLoss
from src.callbacks.calculate_class_metrics import CalculateClassMetrics
from src.callbacks.calculate_metrics import CalculateMetrics


class CollectBest:
    """Callback collecting best metrics."""

    def __init__(self, variants: str = ["val"], best=True):
        self.supported = self.check_variant(variants=variants)
        self.variants = variants
        self.best = best
        super().__init__()

        self.best_index = "best"
        self.history_filename = "history.csv"
        self.best_dir_path = Path("best_metrics")

        self.metrics_dir = NotImplemented

    @staticmethod
    def check_variant(variants: List[str]):
        supported = ["val", "test", ""]
        info = f"Variants not supported. Supported variants: {supported}"

        con = [True if variant in supported else False for variant in variants]
        assert all(con), info

        return supported

    def check_conflicts(self, trainer: Trainer, callback: Callback):
        condition = any(type(i) == callback for i in trainer.callbacks)
        info = "Don't use {} wrapper with parent callback {}."
        names = self.__class__.__name__, callback.__name__

        assert not condition, info.format(*names)

    def condtition(self, col: str):
        return any(True for variant in self.variants if variant in col)

    def extract_name(self, name: str):
        splited = name.split("_")
        if splited[0] in self.supported:
            del splited[0]
        if splited[-1].isnumeric():
            del splited[-1]

        return "".join(splited)

    def set_paths(self, model_root_dir: Path, name: str) -> Tuple[Path, Path]:

        metric_subdir = self.best_dir_path / f"{self.best_index}_{name}"

        best_metric_dir = model_root_dir / metric_subdir
        history_file = best_metric_dir / self.history_filename
        best_last_file = best_metric_dir / self.subdir / self.last_file_name

        self.history_file = history_file
        self.best_last_file = best_last_file
        self.best_metric_dir = best_metric_dir

    def get_new_metric(self, data: dict, root_dir: Path):
        new_metric = pd.DataFrame(data=data)
        new_metric = new_metric.rename_axis(index=self.best_index)

        return new_metric

    def add_new_metric(self, new_metric: pd.DataFrame) -> pd.DataFrame:

        hist = pd.read_csv(self.history_file)
        hist = hist.set_index(self.best_index)

        hist = pd.concat([new_metric, hist], ignore_index=True,)
        return hist.rename_axis(index=self.best_index)

    def create_best_dir(self, history: pd.DataFrame):
        copytree(self.all_path.parent.parent, self.best_metric_dir)
        history.to_csv(self.history_file)

    def replace_best_dir(self, history: pd.DataFrame):
        rmtree(self.best_metric_dir)
        copytree(self.all_path.parent.parent, self.best_metric_dir)
        history.to_csv(self.history_file)

    def prepare(
        self, name: str, extremum: str
    ) -> Tuple[float, pd.DataFrame, Path, Path]:

        func = max if extremum == "max" else min

        if "test" not in name and self.best:
            df = self.data_frame
        else:
            df = pd.read_csv(self.last_path)

        stat = float(func(df[name].values))

        root_dir = self.metrics_dir.parent.parent
        self.set_paths(root_dir, name)

        data = {"model_index": [self.metrics_dir.parent.name], name: [stat]}
        new = self.get_new_metric(data=data, root_dir=self.metrics_dir.parent)

        return stat, new

    def get_extremum(self, name: str) -> str:

        metric_name = self.extract_name(name)
        var = self.metrics.get(metric_name, {})
        extremum = var.get("metric", BaseLoss).extremum

        return extremum

    def collect(
        self,
        name: str,
        current_statistic: float,
        new_metric: pd.DataFrame,
        extremum: str,
    ):
        best_metrics = pd.read_csv(self.history_file)
        assert name in best_metrics.columns, f"Column {name} doesn't exist"

        sign = operator.lt if extremum == "max" else operator.gt
        if sign(best_metrics[name].values[0], current_statistic):
            history = self.add_new_metric(new_metric)
            self.replace_best_dir(history)

    def collect_best_metric(self, name: str, extremum: str):
        current_statistic, new_metric = self.prepare(name, extremum)

        if self.best_last_file.is_file() and self.history_file.is_file():
            self.collect(
                name=name,
                current_statistic=current_statistic,
                new_metric=new_metric,
                extremum=extremum,
            )

        else:
            self.create_best_dir(history=new_metric)

    def collect_best_metrics(self, trainer: Trainer):
        self.save_final_metrics()

        cols = [col for col in self.data_frame.columns if self.condtition(col)]
        for name in cols:
            extremum = self.get_extremum(name)

            self.collect_best_metric(name=name, extremum=extremum)


class CollectBestMetrics(CalculateMetrics, CollectBest):
    """
    Extension of :class:`CalculateMetrics` callback. It's forbidden to
    use them together. Callback first use initiate directory with
    best metrics per currently used architecture. At the end of next
    runs final metrics will be compared with best metrics currently
    saved in directory "best_{metric_name}". Every record better
    than previous ones will be added to file history.csv with info
    about experiment number. Furthermore experiment directory
    content will be saved as backup in "best_{metric_name}" directory.

    Parameters:

        List[str] variants: List of supported variants ['', 'test', 'val']
        bool best: Flag responsible for checking best metrics instead of last

    Output:

        - `history.csv`: File with history of models with best metric
        - content of experiment run directory

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_fit_start(self, trainer: Trainer, pl_module: Module):
        super().on_fit_start(trainer, pl_module)
        self.check_conflicts(trainer=trainer, callback=CalculateMetrics)

    def on_fit_end(self, trainer: Trainer, pl_module: Module):
        self.collect_best_metrics(trainer=trainer)


class CollectBestClassMetrics(CalculateClassMetrics, CollectBest):
    """
    Extension of :class:`CalculateClassMetrics` callbck. It's forbidden to
    use them together. Callback first use initiate directory with
    best metrics per currently used architecture. At the end of next
    runs final metrics will be compared with best metrics currently
    saved in directory "best_{metric_name}". Every record better
    than previous ones will be added to file history.csv with info
    about experiment number. Furthermore experiment directory
    content will be saved as backup in "best_{metric_name}" directory.

    Parameters:

        List[str] variants: List of supported variants ['', 'test', 'val']
        bool best: Flag responsible for checking best metrics instead of last

    Output:

        - `history.csv`: File with history of models with best metric
        - content of experiment run directory

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_fit_start(self, trainer: Trainer, pl_module: Module):
        super().on_fit_start(trainer, pl_module)
        self.check_conflicts(trainer=trainer, callback=CalculateClassMetrics)

    def on_fit_end(self, trainer: Trainer, pl_module: Module):
        self.collect_best_metrics(trainer=trainer)
