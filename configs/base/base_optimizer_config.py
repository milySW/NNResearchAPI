from typing import Dict, List, Optional, ItemsView

from src.models.base import LitModel
from src.optimizers import BaseOptim
from src.optimizers.schedulers import BaseScheduler
from configs import BaseConfig, DefaultOptimizers, DefaultSchedulers


class DefaultOptimizersAndSchedulers(BaseConfig):
    optimizers: Dict[str, BaseOptim] = DefaultOptimizers.to_dict()
    schedulers: Dict[str, BaseScheduler] = DefaultSchedulers.to_dict()

    @classmethod
    def get_optimizer(cls, opt: dict, models: dict) -> BaseOptim:
        func = opt["optimizer"]
        kwargs = opt["kwargs"]
        parameters = models[opt["character"]].parameters()

        return func(parameters, **kwargs)

    @classmethod
    def get_optimizers(
        cls,
        opts: ItemsView[str, Dict[str, BaseOptim]],
        models: Dict[str, Optional[LitModel]],
    ) -> Dict[str, BaseOptim]:

        return {name: cls.get_optimizer(opt, models) for name, opt in opts}

    @classmethod
    def get_sched(cls, opt: BaseOptim, scheduler: dict) -> BaseScheduler:
        scheduler_func = scheduler["sched"]
        kwargs = scheduler["common_kwargs"].__dict__
        scheduler_kwargs = scheduler["scheduler_kwargs"]

        return scheduler_func(opt, **scheduler_kwargs, **kwargs)

    @classmethod
    def get_scheds(
        cls, optimizers: Dict[str, BaseOptim]
    ) -> List[Dict[str, BaseScheduler]]:

        all_schedulers = []

        for name in optimizers.keys():
            opt = optimizers[name]
            schedulers = cls.schedulers.get(name, None)

            if schedulers is None:
                continue

            var = [cls.get_sched(opt, sched) for sched in schedulers]
            all_schedulers.extend(var)

        return all_schedulers
