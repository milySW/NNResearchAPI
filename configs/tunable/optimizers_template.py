from configs import BaseConfig
from src.optimizers import Adam
from src.optimizers.schedulers import BaseScheduler, ExponentialLR


class SchedulerDict(BaseConfig):
    scheduler: BaseScheduler = None
    interval: str = "epoch"
    frequency: int = 1
    reduce_on_plateau: bool = False
    monitor: str = "val_loss"


class Optimizers(BaseConfig):
    adam = dict(optimizer=Adam, kwargs={}, character="normal")


class Schedulers(BaseConfig):
    adam = [
        dict(
            sched=ExponentialLR,
            scheduler_dict=BaseScheduler(**{}),
            scheduler_kwargs={"gamma": 0.9},
        )
    ]
