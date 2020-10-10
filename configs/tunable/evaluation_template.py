from configs.base.base import BaseConfig
from src.evaluations import TopLosses
from src.losses import CrossEntropyLoss


class DefaultEvaluation(BaseConfig):
    """
    Config responsible for evaluations of :class:`BaseEvaluation`.
    Providing new evaluation require adding new class field field with any name
    """

    top_losses = TopLosses(loss=CrossEntropyLoss(reduction="none"))
