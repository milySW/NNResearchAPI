from abc import ABC, abstractmethod
from random import random
from typing import Iterable, List

import torch

from src.utils.checkers import image_folder
from src.utils.collections import collection_is_none


class BaseTransformation(ABC):
    @abstractmethod
    def __init__(
        self,
        x: bool = True,
        y: bool = False,
        train: bool = True,
        valid: bool = False,
        test: bool = False,
        ratio: float = 1.0,
    ):
        self.x = x
        self.y = y
        self.train = train
        self.valid = valid
        self.test = test
        self.ratio = ratio

    @property
    def name(self):
        return self.__class__.__name__

    def transformation(self, data: Iterable, **kwargs) -> List[Iterable]:
        return self.core_transofmation(data, **kwargs)

    @staticmethod
    def core_transofmation(data: Iterable, **kwargs) -> List[Iterable]:
        return NotImplemented

    def __call__(self, data: Iterable) -> List[Iterable]:
        if self.ratio < random():
            return data

        if isinstance(data, torch.Tensor):
            return self.transformation(data)

        x, y = data

        if image_folder(x):
            return data

        if self.x and not collection_is_none(x):
            x = self.transformation(x)
        if self.y and not collection_is_none(y):
            y = self.transformation(y)

        return [x, y]
