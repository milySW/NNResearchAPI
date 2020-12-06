from configs.base.base import BaseConfig
from src.evaluations import TopLosses
from src.losses import CrossEntropyLoss


class DefaultEvaluations(BaseConfig):
    """
    Config responsible for evaluations of :class:`BaseEvaluation`.
    Providing new evaluation require adding new class field with any name
    """

    top_losses = TopLosses(
        loss=CrossEntropyLoss(reduction="none"), k=200, save_plots=False
    )
