from __future__ import annotations

from typing import Any, Iterable

from torch.utils.data.dataloader import DataLoader as PLDataloader

import configs


class DataLoader(PLDataloader):
    def __init__(
        self,
        dataset: Iterable,
        config: configs.DefaultConfig,
        dataset_type: str,
        **kwargs
    ):
        self.set_params(config=config, ds_type=dataset_type)
        transformed_dataset = self.apply_tfms(dataset, dataset_type)

        super().__init__(
            dataset=transformed_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            **kwargs
        )

    def set_params(self, config: configs.DefaultConfig, ds_type: str) -> Any:

        self.batch_size = config.training.batch_size
        self.shuffle = True if ds_type == "train" else False
        self.transformations = config.transformations.value_list()

    def apply_tfms(self, data: Iterable, ds_type: str):
        for transformation in self.transformations:
            data = transformation(data, ds_type)

        return list(zip(data[0], data[1]))
