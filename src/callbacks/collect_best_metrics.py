from pathlib import Path
from shutil import rmtree, copytree
import pandas as pd

from src.callbacks import CalculateMetrics


class CollectBestMetrics(CalculateMetrics):
    def on_fit_end(self, trainer, pl_module):
        last = self.data_frame.tail(1).rename_axis(index="epoch")
        last.to_csv(self.last_path)

        cols = self.data_frame.columns
        for name in cols:
            stat = float(self.data_frame.tail(1)[name].values.item())
            model_root_dir = Path(trainer.default_root_dir).parent

            best_metric_dir = model_root_dir / "best_metrics" / f"best_{name}"
            best_file = best_metric_dir / "metrics" / "metrics_last.csv"
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
                    copytree(self.metrics_dir.parent, best_metric_dir)
                    hist.to_csv(best_metric_dir / history_filename)

            else:
                copytree(self.metrics_dir.parent, best_metric_dir)
                hist = new_metric
                hist.to_csv(best_metric_dir / history_filename)
