from configs.base.base import BaseConfig
from src.optimizers import Adam
from src.optimizers.schedulers import BaseScheduler, ExponentialLR


class SchedulerCommonKwargs(BaseConfig):
    """Class storing default parameters common for all schedulers"""

    scheduler: BaseScheduler = None
    interval: str = "epoch"
    frequency: int = 1
    reduce_on_plateau: bool = False
    monitor: str = "val_loss"


class DefaultOptimizers(BaseConfig):
    """
    Config responsible for passing optimizers of :class:`BaseOptimizer` type.
    Providing new optimizers require adding new class field as dict
    with any name.

    Underneath description of field parameters

    Parameters:

        BaseOptim scheduler: optimizer class
        dict kwargs: dict with parameters for optimizer class
        str character: 'normal' for standard model,'disc' for discriminator
            in GAN model, 'gen' for generator in GAN model

    """

    adam = dict(optimizer=Adam, kwargs={}, character="normal")


class DefaultSchedulers(BaseConfig):
    """
    Config responsible for passing schedulers of :class:`BaseScheduler` type.
    Providing new schedulers require adding new class field as dict with same
    name as field name of optimizer, which has to be used with scheduler.

    Underneath description of field parameters

    Parameters:

        BaseScheduler scheduler: scheduler class
        SchedulerCommonKwargs common_kwargs: params common for all schedulers
        dict scheduler_kwargs: dict with parameters for scheduler class

    """

    adam = [
        dict(
            sched=ExponentialLR,
            common_kwargs=BaseScheduler(**{}),
            scheduler_kwargs={"gamma": 0.9},
        )
    ]
