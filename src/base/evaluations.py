from abc import ABC


class BaseEvaluation(ABC):
    @property
    def name(self):
        return self.__class__.__name__
