from copy import deepcopy
from typing import Any, Dict, ItemsView, List, Optional

from torch.nn import Module

from configs.base.base import BaseConfig
from configs.tunable.optimizers_template import (
    DefaultOptimizers,
    DefaultSchedulers,
)
from src.base.optimizers import BaseOptim
from src.base.schedulers import BaseScheduler
from src.utils.logging import get_logger

logger = get_logger("SchedulerSetter")


class DefaultOptimizersAndSchedulers(BaseConfig):
    optimizers: Dict[str, BaseOptim] = DefaultOptimizers.to_dict()
    schedulers: Dict[str, BaseScheduler] = DefaultSchedulers.to_dict()

    @staticmethod
    def find_lr(optimizer_kwargs: Dict[str, Any], model):
        trainer = deepcopy(model.trainer)
        trainer.profile_connector.on_trainer_init(None)

        if hasattr(model, "dropout"):
            dropout = model.dropout
            model.dropout = 0

        if model.pretrained:
            model.freeze_pretrained_layers(freeze=False)
            lr_finder = trainer.tuner.lr_find(model)
            model.freeze_pretrained_layers(freeze=True)

        if not model.pretrained:
            lr_finder = trainer.tuner.lr_find(model)

        if hasattr(model, "dropout"):
            model.dropout = dropout

        new_lr = lr_finder.suggestion()
        logger.info(f"Set initial learning rate to: {new_lr}")
        optimizer_kwargs.update({"lr": new_lr})

        fig = lr_finder.plot(suggest=True)

        lr_finder_path = model.trainer.root_dir / "plots"
        lr_finder_path.mkdir(parents=True, exist_ok=True)

        fig.savefig(lr_finder_path / "lr_finder", transparent=True)

        return optimizer_kwargs

    @classmethod
    def get_optimizer(cls, optimizer: dict, models: dict) -> BaseOptim:

        func = optimizer["optimizer"]
        kwargs = optimizer["kwargs"]
        model = models[optimizer["character"]]

        if optimizer["auto_lr"] and not model.configured_optimizers:
            model.configured_optimizers = True
            kwargs = cls.find_lr(optimizer_kwargs=kwargs, model=model)

        parameters = models[optimizer["character"]].parameters()

        return func(parameters, **kwargs)

    @classmethod
    def get_optimizers(
        cls,
        opts: ItemsView[str, Dict[str, BaseOptim]],
        models: Dict[str, Optional[Module]],
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
