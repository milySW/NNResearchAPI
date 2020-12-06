from abc import ABC
from pathlib import Path

import torch


class BaseEvaluation(ABC):
    @property
    def name(self):
        return self.__class__.__name__

    def manage_evals(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        data: torch.Tensor,
        output: Path,
        image: bool,
    ):

        return NotImplemented
