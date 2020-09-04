from typing import List, Tuple, Dict, Callable
from src.losses import BaseLoss, CrossEntropyLoss
from src.metrics import BaseMetric, Accuracy
from src.optim import BaseOptim, Adam
from src.callbacks import (
    Visualizator,
    CollectBestMetrics,
    CollectBestClassMetrics,
)


class DefaultTraining:
    def __init__(
        self,
        epochs: int = 2,
        batch_size: int = 128,
        checkpoint_callback: bool = False,
        seed: int = 42,
        loss: BaseLoss = CrossEntropyLoss(),
        metrics: Tuple[Dict[str, Tuple[str, BaseMetric, dict]]] = (
            dict(name="accuracy", metric=Accuracy, kwargs={}),
        ),
        optim: BaseOptim = dict(optimizer=Adam, kwargs={}),
        callbacks: List[Callable] = [
            CollectBestClassMetrics(),
            CollectBestMetrics(),
            Visualizator(),
        ],
    ):

        self.epochs = epochs
        self.batch_size = batch_size
        self.checkpoint_callback = checkpoint_callback
        self.seed = seed
        self.loss = loss
        self.metrics = metrics
        self.optim = optim
        self.callbacks = callbacks
