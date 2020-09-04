from pytorch_lightning.callbacks.base import Callback
from src.models.lightning import LitModel
from pytorch_lightning.trainer.trainer import Trainer


class Visualizator(Callback):
    def on_train_end(self, trainer: Trainer, pl_module: LitModel):
        print("do something when training ends")
