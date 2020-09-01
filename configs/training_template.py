from typing import Any, List, Tuple, Dict, Optional, Callable
from src.losses import CrossEntropyLoss
from src.metrics import Accuracy
from src.callbacks import (
    CalculateClassMetrics,
    Visualizator,
    CollectBestMetrics,
)


class DefaultTraining:
    def __init__(
        self,
        epochs: int = 1,
        batch_size: int = 128,
        checkpoint_callback: bool = False,
        seed: int = 42,
        loss: Callable = CrossEntropyLoss(),
        metrics: Tuple[Dict[str, Optional[Any]], ...] = (
            dict(name="accuracy", metric=Accuracy, kwargs={}),
        ),
        callbacks: List[Callable] = [
            CalculateClassMetrics(),
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
        self.callbacks = callbacks
