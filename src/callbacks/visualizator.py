from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.trainer.trainer import Trainer

from src.models.base import LitModel


class Visualizator(Callback):
    def on_train_end(self, trainer: Trainer, pl_module: LitModel):
        print("do something when training ends")
