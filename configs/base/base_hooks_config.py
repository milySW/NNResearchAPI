from functools import cached_property
from itertools import product
from typing import Any, Callable, Dict, List, Optional

import torch

from pytorch_lightning.core.hooks import ModelHooks
from pytorch_lightning.utilities import AMPType

from configs.base.base import BaseConfig
from configs.tunable.hooks_template import DefaultHooks
from src.hooks import BaseHook
from src.losses import BaseLoss
from src.optimizers import BaseOptim


class DefaultBindedHooks(BaseConfig):
    hooks: Dict[str, BaseHook] = DefaultHooks.value_list()

    def __init__(self):
        self.check_for_collisions()
        self.log_if_collisions()

        self._pl_module = None

    @property
    def pl_module(self):
        return self._pl_module

    @pl_module.setter
    def pl_module(self, value):
        self._pl_module = value

    @cached_property
    def unbindable_dict(self) -> Dict[str, int]:
        return dict(
            backward=0.0,
            amp_scale_loss=0.0,
            transfer_batch_to_device=0.0,
            calculate_batch=0.0,
        )

    def log_collision_info(self, name: str, n: int) -> str:
        return f"Hook {name} should has at most 1 implementation, provided {n}"

    def check_for_collisions(self):
        for hook, name in product(self.hooks, self.unbindable_dict.keys()):
            value = int(hasattr(hook, name))
            self.unbindable_dict[name] += value

    def log_if_collisions(self):
        for name, implementations_number in self.unbindable_dict.items():
            info = self.log_collision_info(name, implementations_number)
            assert implementations_number <= 1, info

    def bind_identical_hooks(
        self,
        method_name: str,
        params: List[Any] = [],
        backup_function: Optional[Callable] = None,
    ):
        if not self.hooks and backup_function:
            return backup_function(*params)

        for hook in self.hooks:
            hook.pl_module = self.pl_module

            return self.call_if_exists(
                obj=hook,
                method_name=method_name,
                parameters=params,
                backup_function=backup_function,
            )

    @staticmethod
    def call_if_exists(
        obj: object,
        method_name: str,
        parameters: List[Any] = [],
        backup_function: Optional[Callable] = None,
    ):
        if hasattr(obj, method_name):
            return getattr(obj, method_name)(*parameters)
        elif backup_function:
            return backup_function(*parameters)

    def setup(self, stage: str):
        self.bind_identical_hooks("setup", params=[stage])

    def teardown(self, stage: str):
        self.bind_identical_hooks("teardown", params=[stage])

    def on_fit_start(self):
        self.bind_identical_hooks("on_fit_start")

    def on_fit_end(self):
        self.bind_identical_hooks("on_fit_end")

    def on_train_start(self) -> None:
        self.bind_identical_hooks("on_train_start")

    def on_train_end(self) -> None:
        self.bind_identical_hooks("on_train_end")

    def on_pretrain_routine_start(self) -> None:
        self.bind_identical_hooks("on_pretrain_routine_start")

    def on_pretrain_routine_end(self) -> None:
        self.bind_identical_hooks("on_pretrain_routine_end")

    def on_train_batch_start(
        self, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        params = [batch, batch_idx, dataloader_idx]
        self.bind_identical_hooks("on_train_batch_start", params=params)

    def on_train_batch_end(
        self, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        params = [batch, batch_idx, dataloader_idx]
        self.bind_identical_hooks("on_train_batch_end", params=params)

    def on_validation_batch_start(
        self, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        params = [batch, batch_idx, dataloader_idx]
        self.bind_identical_hooks("on_validation_batch_start", params=params)

    def on_validation_batch_end(
        self, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        params = [batch, batch_idx, dataloader_idx]
        self.bind_identical_hooks("on_validation_batch_end", params=params)

    def on_test_batch_start(
        self, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        params = [batch, batch_idx, dataloader_idx]
        self.bind_identical_hooks("on_test_batch_start", params=params)

    def on_test_batch_end(
        self, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        params = [batch, batch_idx, dataloader_idx]
        self.bind_identical_hooks("on_test_batch_end", params=params)

    def on_epoch_start(self) -> None:
        self.bind_identical_hooks("on_epoch_start")

    def on_epoch_end(self) -> None:
        self.bind_identical_hooks("on_epoch_end")

    def on_train_epoch_start(self) -> None:
        self.bind_identical_hooks("on_train_epoch_start")

    def on_train_epoch_end(self) -> None:
        self.bind_identical_hooks("on_train_epoch_end")

    def on_validation_epoch_start(self) -> None:
        self.bind_identical_hooks("on_validation_epoch_start")

    def on_validation_epoch_end(self) -> None:
        self.bind_identical_hooks("on_validation_epoch_end")

    def on_test_epoch_start(self) -> None:
        self.bind_identical_hooks("on_test_epoch_start")

    def on_test_epoch_end(self) -> None:
        self.bind_identical_hooks("on_test_epoch_end")

    def on_pre_performance_check(self) -> None:
        self.bind_identical_hooks("on_pre_performance_check")

    def on_post_performance_check(self) -> None:
        self.bind_identical_hooks("on_post_performance_check")

    def on_before_zero_grad(self, optim: BaseOptim) -> None:
        self.bind_identical_hooks("on_before_zero_grad", params=[optim])

    def on_after_backward(self) -> None:
        self.bind_identical_hooks("on_after_backward")

    def backward(
        self, trainer, loss: torch.Tensor, optim: BaseOptim, optim_idx: int
    ) -> None:
        self.bind_identical_hooks(
            method_name="backward",
            params=[trainer, loss, optim, optim_idx],
            backup_function=ModelHooks().backward,
        )

    def amp_scale_loss(
        self,
        unscaled_loss: BaseLoss,
        optim: BaseOptim,
        optim_idx: int,
        amp_backend: AMPType,
    ):
        return self.bind_identical_hooks(
            method_name="amp_scale_loss",
            params=[unscaled_loss, optim, optim_idx, amp_backend],
            backup_function=ModelHooks().amp_scale_loss,
        )

    def transfer_batch_to_device(self, batch: Any, dev: torch.device) -> Any:
        return self.bind_identical_hooks(
            method_name="transfer_batch_to_device",
            params=[batch, dev],
            backup_function=ModelHooks().transfer_batch_to_device,
        )

    def calculate_batch(self, batch: list, step: str) -> torch.Tensor:
        params = [batch, step]
        return self.bind_identical_hooks("calculate_batch", params=params)
