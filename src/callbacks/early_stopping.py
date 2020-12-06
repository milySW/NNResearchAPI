from pytorch_lightning.callbacks.early_stopping import EarlyStopping as PLEarly

from src.base.callback import BaseCallback


class EarlyStopping(BaseCallback, PLEarly):
    __doc__ = PLEarly.__doc__

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_validation_end(self, trainer, pl_module):
        # override this to disable early stopping at the end of val loop
        pass

    def on_epoch_end(self, trainer, pl_module):
        # instead, do it at the end of training loop
        self._run_early_stopping_check(trainer, pl_module)
