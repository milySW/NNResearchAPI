# NOTE: File for experiments with voting classifiers
from typing import List

import torch

from src.loaders import DataLoader
from src.utils.features import emsd  # noqa


def two_classifiers(
    model_preds: List[torch.Tensor], loader: DataLoader,
) -> torch.Tensor:

    preds_1, preds_2 = model_preds

    def move_threshold(preds: torch.Tensor, threshold: float) -> torch.Tensor:
        a = torch.zeros(preds_2.shape[0], 1)
        b = torch.ones(preds_2.shape[0], 1) * threshold
        threshold_tensor = torch.cat([a, b], dim=1)

        return preds + threshold_tensor

    # preds_2 = move_threshold(preds=preds_2, threshold=0.2)
    flat_preds_1, _ = preds_1.argmax(dim=1), preds_2.argmax(dim=1)

    for index in range(flat_preds_1.shape.numel()):
        x, y = loader.dataset[index]
        # main_choice = flat_preds_1[index]
        # support_choice = flat_preds_2[index]

    return preds_1
