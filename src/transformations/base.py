from abc import ABC, abstractmethod
from random import random
from typing import Iterable


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

    def transformation(self, data: Iterable, **kwargs):
        return self.core_transofmation(data, **kwargs)

    @staticmethod
    def core_transofmation(data: Iterable, **kwargs):
        return NotImplemented

    def __call__(self, data: Iterable):
        if self.ratio < random():
            return data

        x, y = data

        if self.x:
            x = self.transformation(x)
        if self.y:
            y = self.transformation(y)

        return [x, y]
