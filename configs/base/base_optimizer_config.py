from typing import Dict, List, Tuple

from src.optimizers import BaseOptim
from src.optimizers.schedulers import BaseScheduler
from configs import BaseConfig, Optimizers, Schedulers


class DefaultOptimizers(BaseConfig):
    optimizers: Dict[str, BaseOptim] = Optimizers.to_dict()
    schedulers: Dict[str, BaseScheduler] = Schedulers.to_dict()

    @classmethod
    def get_optimizer(cls, opt: dict, models: dict) -> BaseOptim:
        func = opt["optimizer"]
        kwargs = opt["kwargs"]
        parameters = models[opt["character"]].parameters()

        return func(parameters, **kwargs)

    @classmethod
    def get_optimizers(cls, opts: List[Tuple], models: dict) -> dict:
        return {name: cls.get_optimizer(opt, models) for name, opt in opts}

    @classmethod
    def get_scheduler(cls, opt: BaseOptim, scheduler: dict) -> BaseScheduler:
        scheduler_func = scheduler["sched"]
        kwargs = scheduler["scheduler_dict"].__dict__
        scheduler_kwargs = scheduler["scheduler_kwargs"]

        return scheduler_func(opt, **scheduler_kwargs, **kwargs)

    @classmethod
    def get_schedulers(cls, opts) -> List[Dict[str, Dict]]:
        all_schedulers = []

        for name in opts.keys():
            opt = opts[name]
            schedulers = cls.schedulers[name]

            var = [cls.get_scheduler(opt, sched) for sched in schedulers]
            all_schedulers.extend(var)

        return all_schedulers
