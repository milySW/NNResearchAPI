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

    @property
    def transformation_sets(self):
        return dict(train=self.train, valid=self.valid, test=self.test)

    def __call__(self, data: Iterable, ds_type: str):
        if ds_type == "train" and self.train:
            return self.apply_transformations(data)

        elif ds_type == "valid" and self.train:
            return self.apply_transformations(data)

        elif ds_type == "test" and self.train:
            return self.apply_transformations(data)

        else:
            return data

    def apply_transformations(self, data: Iterable):
        x, y = data

        if self.x:
            x = self.transformation(x)
        if self.y:
            y = self.transformation(y)

        return [x, y]
