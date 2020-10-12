import math

from copy import deepcopy
from typing import Any, List

import torch

from src.hooks.base import BaseHook
from src.utils.logging import get_logger

logger = get_logger("Selective Bakckpropagation Hook")


class SelectiveBackprop(BaseHook):
    """
    Implemented according to paper https://arxiv.org/pdf/1910.00762.pdf
    Key idea:
        Train only on part of batch to speed up the training.
        Calculate loss value with forward pass (which is roughly 3 times
        faster than backward pass), then choose elements which contribute
        to overall loss value the most. Finally calculate backward pass
        and update model parameters with chosen elements.

    Parameters:
        float min_sample: (0.0, 1.0]
            used to sample fraction of elements with biggest losses

        float min_loss: (0.0, 1.0]
            used to sample elements responsible for min_loss_perc loss value
    """

    def __init__(self, min_sample: float, min_loss: float):
        self.check_parameters(min_sample, min_loss)

        self.min_sample = min_sample
        self.min_loss = min_loss

        self.used_losses_sum, self.losses_sum = 0.0, 0.0
        self.used_samples, self.samples = 0.0, 0.0

    @property
    def criterion(self):
        loss_function = deepcopy(self.pl_module.loss_function)
        loss_function.reduction = "none"
        return loss_function

    @property
    def epoch(self):
        return self.pl_module.current_epoch

    @property
    def device(self):
        return self.pl_module.device.type

    @staticmethod
    def check_parameters(min_sample: float, min_loss: float):
        def check_parameter(value: float, name: str):
            info = f"{name} value is not in the range (0, 1]"
            assert 0 < value <= 1, info

        check_parameter(value=min_sample, name="min_sample")
        check_parameter(value=min_loss, name="min_loss")

    def on_epoch_start(self):
        self.used_losses_sum, self.losses_sum = 0.0, 0.0
        self.used_samples, self.samples = 0.0, 0.0

    def on_train_batch_start(
        self, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:

        if not self.pl_module.training or self.epoch == 0:
            return

        x, y = [data.to(self.device) for data in batch]
        with torch.no_grad():
            output = self.pl_module(x)
            y = torch.argmax(y.squeeze(), 1)

        losses = self.criterion(output, y).detach()
        indices = self.get_loss_idxs(losses)

        self.used_losses_sum += losses[indices].sum()
        self.losses_sum += losses.sum()
        self.used_samples += len(indices)
        self.samples += len(losses)

        batch[:] = [data[indices] for data in batch]

    def on_epoch_end(self):

        if self.epoch > 0:
            loss_perc = self.used_losses_sum / self.losses_sum
            sample_perc = self.used_samples / self.samples
        elif self.epoch == 0:
            loss_perc, sample_perc = 1.0, 1.0

        logger.info(f"Loss ratio used in epoch {self.epoch}: {loss_perc}")
        logger.info(f"Sample ratio used in epoch {self.epoch}: {sample_perc}")

    def get_loss_idxs(self, losses: torch.Tensor) -> List:
        min_sample = math.ceil(len(losses) * self.min_sample)
        sorted_losses, indices = torch.sort(losses, descending=True)

        sorted_losses /= sorted_losses.sum()
        losses_cdf = torch.cumsum(sorted_losses, dim=0)

        min_loss = torch.min(losses_cdf >= self.min_loss, dim=0)[1] + 1
        indices = indices[: max(min_sample, min_loss)].tolist()
        return indices
