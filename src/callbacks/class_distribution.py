from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch

from pytorch_lightning.trainer.trainer import Trainer
from torch.nn import Module

from src.base.callback import BaseCallback


class ClassDistribution(BaseCallback):
    def __init__(
        self,
        classes: Optional[Iterable] = None,
        variants: Tuple[str] = ["val"],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.preds = []
        self.labels = []
        self.losses = []

        self._classes = classes
        self.variants = self.check_variant(variants=variants)

    @property
    def classes(self):
        all_classes = self.labels.unique()

        if self._classes is None:
            return all_classes

        elif set(self._classes) <= set(all_classes):
            return self._classes

        else:
            info = f"Supported classes are: {all_classes}"
            ValueError(f"Can't find selected classes. {info}")

    def initialize_tensors(self):
        self.preds = torch.empty(0)
        self.labels = torch.empty(0)
        self.losses = torch.empty(0)

    def fill_tensors(self, trainer: Trainer):
        self.preds = torch.cat((self.preds, trainer.calculations["preds"]))
        self.labels = torch.cat((self.labels, trainer.calculations["labels"]))
        self.losses = torch.cat((self.losses, trainer.calculations["losses"]))

    def save_distributions(self, trainer: Trainer):
        for _class in self.classes:
            category_preds = self.preds[self.labels == _class]
            bars = category_preds.sum(dim=0) / category_preds.sum()

            plt.bar(self.classes, bars, width=0.4)

            plt.xlabel("Rozkład prawdopodobieństwa")
            plt.ylabel("Prawdopodobieńswo")
            plt.title(f"Pewność przy podejmowaniu decyzji dla klasy {_class}")

            path = self.dist_dir / f"class_{int(_class)}"
            path.mkdir(parents=True, exist_ok=True)

            path = path / f"epoch_{trainer.current_epoch}"

            plt.savefig(path, transparent=True)
            plt.close()

    def on_fit_start(self, trainer: Trainer, pl_module: Module):
        self.dist_dir = Path(trainer.default_root_dir) / "distributions"

    def on_train_epoch_start(self, trainer: Trainer, pl_module: Module):
        self.initialize_tensors()

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: Module):
        self.initialize_tensors()

    def on_test_epoch_start(self, trainer: Trainer, pl_module: Module):
        self.initialize_tensors()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: Module,
        outputs: List[Any],
        batch: List[torch.Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ):
        self.fill_tensors(trainer=trainer)

        condition_1 = trainer.num_training_batches - batch_idx == 1
        condition_2 = "" in self.variants

        if condition_1 and condition_2:
            self.save_distributions(trainer=trainer)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: Module,
        outputs: List[Any],
        batch: List[torch.Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ):
        self.fill_tensors(trainer=trainer)

        condition_1 = trainer.num_val_batches[dataloader_idx] - batch_idx == 1
        condition_2 = "val" in self.variants

        if condition_1 and condition_2:
            self.save_distributions(trainer=trainer)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: Module,
        outputs: List[Any],
        batch: List[torch.Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ):
        self.fill_tensors(trainer=trainer)

        condition_1 = trainer.num_test_batches[dataloader_idx] - batch_idx == 1
        condition_2 = "test" in self.variants

        if condition_1 and condition_2:
            self.save_distributions(trainer=trainer)
