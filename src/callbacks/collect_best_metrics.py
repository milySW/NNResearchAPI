from pathlib import Path
from shutil import rmtree, copytree

import pandas as pd
from pytorch_lightning.trainer.trainer import Trainer

from src.callbacks import CalculateMetrics, CalculateClassMetrics
from src.models.lightning import LitModel


class CollectBestMetrics(CalculateMetrics):
    def on_fit_end(self, trainer: Trainer, pl_module: LitModel):
        collect_best_metrics(
            trainer=trainer,
            metrics=self.data_frame,
            input_file_path=self.last_path,
            metric_csv_path="metrics_last.csv",
        )


class CollectBestClassMetrics(CalculateClassMetrics):
    def on_fit_end(self, trainer: Trainer, pl_module: LitModel):
        collect_best_metrics(
            trainer=trainer,
            metrics=self.data_frame,
            input_file_path=self.last_path,
            metric_csv_path="metrics_group_last.csv",
        )


def collect_best_metrics(
    trainer: Trainer,
    metrics: pd.DataFrame,
    input_file_path: Path,
    metric_csv_path: Path,
):
    last = metrics.tail(1).rename_axis(index="epoch")
    last.to_csv(input_file_path)

    cols = metrics.columns
    for name in cols:
        stat = float(metrics.tail(1)[name].values.item())
        model_root_dir = Path(trainer.default_root_dir).parent

        best_metric_dir = model_root_dir / "best_metrics" / f"best_{name}"
        best_file = best_metric_dir / "metrics" / metric_csv_path
        history_filename = "history.csv"

        model_index = Path(trainer.default_root_dir).name
        data = {"model_index": [model_index], name: [stat]}
        new_metric = pd.DataFrame(data=data).rename_axis(index="best")

        if best_file.is_file():
            best_metrics = pd.read_csv(best_file)

            if best_metrics[name].values.item() < stat:
                hist = pd.read_csv(best_metric_dir / history_filename)
                hist = hist.set_index("best")

                data = [new_metric, hist]
                hist = pd.concat(data, ignore_index=True,)
                hist = hist.rename_axis(index="best")

                rmtree(best_metric_dir)
                copytree(input_file_path.parent.parent, best_metric_dir)
                hist.to_csv(best_metric_dir / history_filename)

        else:
            copytree(input_file_path.parent.parent, best_metric_dir)
            hist = new_metric
            hist.to_csv(best_metric_dir / history_filename)
