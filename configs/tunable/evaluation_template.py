import configs

from src.evaluations import TopLosses
from src.losses import CrossEntropyLoss


class DefaultEvaluation(configs.BaseConfig):
    """
    Config responsible for evaluations of :class:`BaseEvaluation` type.
    Providing new evaluation require adding new class field field with any name
    """

    top_losses = TopLosses(loss=CrossEntropyLoss(reduction="none"))
