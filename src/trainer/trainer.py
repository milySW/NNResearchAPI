from pytorch_lightning import Trainer as PLTrainer


class Trainer(PLTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
