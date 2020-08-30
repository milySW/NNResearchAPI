from pytorch_lightning.callbacks.base import Callback


class Visualizator(Callback):
    def on_train_end(self, trainer, pl_module):
        print("do something when training ends")
