from typing import List, Tuple, Dict, Callable
from src.losses import BaseLoss, CrossEntropyLoss
from src.metrics import BaseMetric, Accuracy
from src.optim import BaseOptim, Adam
from src.util.loaders import load_default_sets
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
        loader_func=load_default_sets,
        loss: BaseLoss = CrossEntropyLoss(),
        metrics: Tuple[Dict[str, Tuple[str, BaseMetric, dict]]] = (
            dict(name="accuracy", metric=Accuracy, kwargs={}, plot=True),
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
        self.loader_func = loader_func
        self.loss = loss
        self.metrics = metrics
        self.optim = optim
        self.callbacks = callbacks
