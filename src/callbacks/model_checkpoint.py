from pytorch_lightning.callbacks import ModelCheckpoint as PLModelCheckpoint

from src.base.callback import BaseCallback


class ModelCheckpoint(BaseCallback, PLModelCheckpoint):
    __doc__ = PLModelCheckpoint.__doc__

    def __init__(self, save_best_only=True, dirpath="checkpoints", **kwargs):
        super().__init__(**kwargs)
