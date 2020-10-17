from functools import cached_property
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from torch.utils.data.dataloader import DataLoader as PLDataloader
from tqdm import tqdm

from configs import DefaultConfig
from src.transformations import BaseTransformation
from src.utils.logging import get_logger

logger = get_logger("PrepareDataloader")


class DataLoader(PLDataloader):
    def __init__(
        self, dataset: Iterable, config: DefaultConfig, ds_type: str, **kwargs,
    ):
        self.set_params(config=config, ds_type=ds_type)
        transformed_dataset = self.apply_tfms(dataset, ds_type)

        super().__init__(
            dataset=transformed_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            **kwargs,
        )

    @classmethod
    def get_loader(
        cls,
        x_data: np.array,
        labels: np.array,
        config: DefaultConfig,
        dataset_type: str = "train",
    ):
        data = [x_data, labels]
        loader = cls(dataset=data, config=config, ds_type=dataset_type)
        return loader

    @classmethod
    def get_loaders(cls, path_to_data: Path, config: DefaultConfig):

        loading_func = config.training.loader_func
        dtype = config.training.dtype

        sets = loading_func(path_to_data, dtype)
        for key, data_set in sets.items():
            loader = cls.get_loader(*data_set, config=config, dataset_type=key)
            yield loader

    @cached_property
    def preprocessors(self):
        return [tfms for tfms in self.all_preprocessors if self.check_ds(tfms)]

    @property
    def disable(self):
        return len(self.preprocessors) == 0

    def set_params(self, config: DefaultConfig, ds_type: str) -> Any:

        self.ds_type = ds_type
        self.batch_size = config.training.batch_size
        self.shuffle = True if ds_type == "train" else False
        self.all_preprocessors = config.preprocessors.value_list()

    def check_ds(self, tfms: BaseTransformation):
        return getattr(tfms, self.ds_type) is True

    def apply_tfms(self, data: Iterable, ds_type: str):
        info = f"Applying {ds_type} preprocessors ..."

        for tfms in tqdm(self.preprocessors, desc=info, disable=self.disable):
            logger.info(f"Applying {tfms.name} on {ds_type} set ...")
            data = tfms(data)

        return list(zip(data[0], data[1]))
