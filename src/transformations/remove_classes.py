from typing import Iterable, List, Tuple, Union

import torch

from src.base.transformation import BaseTransformation


class RemoveClasses(BaseTransformation):
    def __init__(self, classes: List[int], **kwargs):
        self.classes = classes
        super().__init__(**kwargs)

    def transformation(self, data: Iterable) -> torch.Tensor:
        return self.core_transofmation(data, classes=self.classes)

    @staticmethod
    def core_transofmation(
        data: Iterable, classes: List[Union[int, float]]
    ) -> Tuple[torch.Tensor]:

        x, y = data
        flat_y = y.argmax(dim=1)

        mask = [True if label not in classes else False for label in flat_y]
        labels = flat_y[mask].long()

        unique = labels.unique()
        label_map = unique.sort(descending=False)

        labels = [torch.where(label_map.values == label) for label in labels]
        labels = torch.tensor(labels).squeeze()
        labels = torch.eye(len(unique))[labels, :]

        return [x[mask], labels]
