from abc import ABC, abstractmethod
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
    ):
        self.x = x
        self.y = y
        self.train = train
        self.valid = valid
        self.test = test

    @property
    def name(self):
        return self.__class__.__name__

    def transformation(self, data: Iterable):
        return NotImplemented

    def __call__(self, data: Iterable):
        x, y = data

        if self.x:
            x = self.transformation(x)
        if self.y:
            y = self.transformation(y)

        return [x, y]
