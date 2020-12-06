from itertools import count, permutations
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.base.evaluation import BaseEvaluation
from src.base.loss import BaseLoss
from src.base.model import LitModel
from src.utils.plots import save_data_plot


class TopLosses(BaseEvaluation):
    def __init__(self, loss, k=100, save_reports=True, save_plots=True):
        self.check_loss(loss)

        self.loss = loss
        self.k = k
        self.save_reports = save_reports
        self.save_plots = save_plots

        self.preds = NotImplemented
        self.targets = NotImplemented
        self.losses = NotImplemented

    @property
    def folder_name(self):
        return "top_losses"

    @property
    def unique(self):
        unique = torch.cat([self.targets, self.preds]).unique()
        return [i.item() for i in unique]

    @property
    def conns(self):
        return list(permutations(self.unique, 2))

    @property
    def freq(self):
        return [self.targets.eq(cat).sum().item() for cat in self.unique]

    @property
    def con_freq(self):
        return [self.check_connection(con).sum().item() for con in self.conns]

    @property
    def top_losses_df(self):
        var = {"target": self.targets, "pred": self.preds, "loss": self.losses}
        return pd.DataFrame(var)

    @property
    def freq_data_df(self):
        var = {"category": self.unique, "frequency": self.freq}
        return pd.DataFrame(var)

    @property
    def con_data_df(self):
        var = {"target|pred": self.conns, "conns_freq": self.con_freq}
        return pd.DataFrame(var)

    @staticmethod
    def check_loss(loss: BaseLoss):
        info = "Suitable reduction type for this evaluation is `none`"
        assert loss.reduction == "none", info

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor):
        pass

    def check_connection(self, connection: Tuple[int]):
        return self.targets.eq(connection[0]) & self.preds.eq(connection[1])

    def manage_reports(
        self,
        losses: torch.Tensor,
        targets: torch.Tensor,
        preds: torch.Tensor,
        data: torch.Tensor,
        output_path: Path,
        image: bool,
    ):
        self.targets = targets
        self.losses = [round(loss.item(), 4) for loss in losses]
        self.preds = torch.argmax(preds.squeeze(), 1)
        self.image = image

        self.log_reports()

        if self.save_reports:
            self.save_report(output_path=output_path)

        if self.save_plots:
            self.save_trajectories_to_png(data, output_path=output_path)

    def log_reports(self):
        for df in [self.top_losses_df, self.freq_data_df, self.con_data_df]:
            print("\n", df.to_string(index=False))

    def save_trajectories_to_png(self, data: torch.Tensor, output_path: Path):
        root = output_path / self.folder_name / "trajectories"
        root.mkdir(parents=True, exist_ok=True)

        zipped_data = zip(count(), self.preds, self.targets, self.losses, data)

        for i, pred, target, loss, volume in zipped_data:
            name = f"top={i + 1}_target={target}_pred={pred}_loss={loss}.png"

            shape = volume.shape

            if self.image:
                size = int(shape[1:].numel() ** (1 / 2))

                reshaped_volume = volume.reshape([volume.shape[0], size, size])
                volume = reshaped_volume.permute(1, 2, 0).numpy()

                plt.imsave(root / name, volume)
            else:
                save_data_plot(data=volume.squeeze().T, path=root / name)

    def save_report(self, output_path: Path):
        root = output_path / self.folder_name / "reports"
        root.mkdir(parents=True, exist_ok=True)

        self.save_report_plot(root)
        self.save_to_csv(root)

    def save_report_plot(self, root: Path):
        path = root / "top_losses_curve.png"
        save_data_plot(data=[self.top_losses_df["loss"]], path=path)

    def save_to_csv(self, root: Path):
        self.top_losses_df.to_csv(root / "top_losses.csv")
        self.freq_data_df.to_csv(root / "class_frequecy.csv")
        self.con_data_df.to_csv(root / "connections_frequency.csv")

    def get_top_losses(self, losses: torch.Tensor) -> Tuple[torch.Tensor]:
        k = self.k if self.k < losses.size().numel() else losses.size().numel()
        topk = torch.topk(losses, k=k)
        return topk.indices, topk.values

    def manage_evals(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        data: torch.Tensor,
        output: Path,
        image: bool,
    ):
        targets = LitModel.manage_labels(targets)
        losses = self.loss(predictions, targets)
        indices, values = self.get_top_losses(losses)

        self.manage_reports(
            losses=values,
            targets=targets[indices],
            preds=predictions[indices],
            data=data[indices],
            output_path=output,
            image=image,
        )
