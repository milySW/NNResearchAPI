from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch

from pytorch_lightning.utilities import AMPType
from torch.utils.data.dataloader import DataLoader

import configs

from src.base.loss import BaseLoss
from src.base.metric import BaseMetric
from src.base.optimizer import BaseOptim
from src.base.scheduler import BaseScheduler


class LitModel(pl.LightningModule):
    def __init__(self, config: configs.DefaultConfig, **kwargs):
        self.config = config
        self.configured_optimizers = False
        self.pretrained_layers = []

        super().__init__(**kwargs)

    @staticmethod
    def model_check(
        current: configs.DefaultModel,
        expected: configs.DefaultModel,
        architecture_name: str,
    ):
        info = f"Passed config is not for {architecture_name} architecutre!"
        assert current.__name__ == expected.__name__, info

    @property
    def pretrained(self):
        return self.config.model.pretrained

    @property
    def name(self):
        return self.config.model.name

    @property
    def freezing_start(self):
        return self.config.model.freezing_start

    @property
    def freezing_stop(self):
        return self.config.model.freezing_stop

    @property
    def loss_function(self) -> BaseMetric:
        return self.config.training.loss

    @property
    def unfreezing_epoch(self) -> bool:
        return self.config.model.unfreezing_epoch

    @property
    def metrics(self) -> Dict[str, Dict[str, Union[dict, BaseMetric, bool]]]:
        return self.config.metrics.to_dict()

    @property
    def optim(self) -> List[BaseOptim]:
        return self.config.optimizers

    @property
    def hooks(self):
        hook = self.config.hooks()
        hook.pl_module = self
        return hook

    @property
    def augmentations(self):
        return self.config.augmentations.value_list()

    @property
    def data_dim(self):
        data_dim = self.config.model.data_dim
        supported = ["1D", "2D", "3D"]

        info = f"Data with dimnesions other than {supported} is not supported."
        cause = f"Change passed dimenstion {data_dim} to one of supported."
        assert data_dim in supported, f"{info}{cause}"

        return self.config.model.data_dim

    @property
    def layers_map(self):
        return NotImplemented

    @property
    def model_gen(self):
        return NotImplemented

    @property
    def model_disc(self):
        return NotImplemented

    def forward(self, x: torch.Tensor):
        return NotImplemented

    def set_example(self, train_loader: DataLoader):
        example = train_loader.dataset[0][0][None]
        self.example_input_array = example

    def configure_optimizers(self) -> Tuple[BaseOptim, List[BaseScheduler]]:
        """
        Set optimizers and learning-rate schedulers
        passed to config as DefaultOptimizer.
        Return:
            - Single optimizer.
        """
        models = dict(normal=self, gen=self.model_gen, disc=self.model_disc,)
        opts = self.optim.optimizers

        optimizers = self.optim.get_optimizers(opts.items(), models)
        schedulers = self.optim.get_scheds(optimizers)

        return list(optimizers.values()), schedulers

    @staticmethod
    def manage_labels(labels: torch.Tensor) -> torch.Tensor:
        if dim := len(labels.shape) == 1:
            pass

        elif len(labels.shape) == 2:
            labels = torch.argmax(labels.squeeze(), 1)

        else:
            problem = "Only flat tensor and one hots are supported"
            raise ValueError(f"Label tensor is {dim} dimensional. {problem}")

        return labels

    def standard_calculate_batch(
        self, batch: list, precalculated_preds: torch.Tensor = None
    ) -> torch.Tensor:

        x, y = batch
        if precalculated_preds is not None:
            y_hat = precalculated_preds
        else:
            y_hat = self(x.float())

        labels = self.manage_labels(labels=y)
        loss = self.loss_function(y_hat, labels)

        calculations = dict(
            inputs=x.detach().cpu(),
            preds=y_hat.detach().cpu(),
            labels=labels.detach().cpu(),
            losses=loss.detach().cpu().unsqueeze(dim=0),
        )

        return loss, calculations

    def calculate_batch(self, batch: list, step: str) -> torch.Tensor:

        if output := self.hooks.calculate_batch(batch=batch, step=step):
            return output
        else:
            return self.standard_calculate_batch(batch=batch)

    def training_step(self, batch: list, batch_idx: int) -> torch.Tensor:
        for tfms in self.augmentations:
            batch = tfms(batch)

        loss, calculations = self.calculate_batch(batch, step="train")
        self.trainer.calculations = calculations

        return loss

    def validation_step(self, batch: list, batch_idx: int) -> torch.Tensor:

        loss, calculations = self.calculate_batch(batch, step="validation")
        self.trainer.calculations = calculations

        return loss

    def test_step(self, batch: list, batch_idx: int) -> torch.Tensor:

        loss, calculations = self.calculate_batch(batch, step="test")
        self.trainer.calculations = calculations

        return loss

    def setup(self, stage: str):
        self.hooks.setup(stage)

    def teardown(self, stage: str):
        self.hooks.teardown(stage)

    def on_fit_start(self):
        self.hooks.on_fit_start()

    def on_fit_end(self):
        self.hooks.on_fit_end()

    def on_train_start(self) -> None:
        self.hooks.on_train_start()

    def on_train_end(self) -> None:
        self.hooks.on_train_end()

    def on_pretrain_routine_start(self) -> None:
        self.hooks.on_pretrain_routine_start()

    def on_pretrain_routine_end(self) -> None:
        self.hooks.on_pretrain_routine_end()

    def on_train_batch_start(
        self, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:

        self.hooks.on_train_batch_start(batch, batch_idx, dataloader_idx)

    def on_train_batch_end(
        self,
        train_step_outputs: List[Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:

        params = [train_step_outputs, batch, batch_idx, dataloader_idx]
        self.hooks.on_train_batch_end(*params)

    def on_validation_batch_start(
        self, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:

        self.hooks.on_validation_batch_start(batch, batch_idx, dataloader_idx)

    def on_validation_batch_end(
        self,
        validation_step_outputs: List[Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:

        params = [validation_step_outputs, batch, batch_idx, dataloader_idx]
        self.hooks.on_validation_batch_end(*params)

    def on_test_batch_start(
        self, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:

        self.hooks.on_test_batch_start(batch, batch_idx, dataloader_idx)

    def on_test_batch_end(
        self,
        test_step_outputs: List[Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:

        params = [test_step_outputs, batch, batch_idx, dataloader_idx]
        self.hooks.on_test_batch_end(*params)

    def on_epoch_start(self) -> None:
        self.hooks.on_epoch_start()

    def on_epoch_end(self) -> None:
        self.hooks.on_epoch_end()

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == self.unfreezing_epoch:
            self.freeze_pretrained_layers(freeze=False)

        self.hooks.on_train_epoch_start()

    def on_train_epoch_end(self, outputs: List[Any]) -> None:
        self.hooks.on_train_epoch_end(outputs)

    def on_validation_epoch_start(self) -> None:
        self.hooks.on_validation_epoch_start

    def on_validation_epoch_end(self) -> None:
        self.hooks.on_validation_epoch_end()

    def on_test_epoch_start(self) -> None:
        self.hooks.on_test_epoch_start()

    def on_test_epoch_end(self) -> None:
        self.hooks.on_test_epoch_end()

    def on_pre_performance_check(self) -> None:
        self.hooks.on_pre_performance_check()

    def on_post_performance_check(self) -> None:
        self.hooks.on_post_performance_check()

    def on_before_zero_grad(self, optim: BaseOptim) -> None:
        self.hooks.on_before_zero_grad(optim)

    def on_after_backward(self) -> None:
        self.hooks.on_after_backward()

    def backward(
        self, loss: torch.Tensor, optimizer: BaseOptim, optimizer_idx: int
    ) -> None:

        self.hooks.backward(loss, optimizer, optimizer_idx)

    def amp_scale_loss(
        self,
        unscaled_loss: BaseLoss,
        optim: BaseOptim,
        optim_idx: int,
        amp_backend: AMPType,
    ):
        return self.hooks.amp_scale_loss(
            unscaled_loss=unscaled_loss,
            optimizer=optim,
            optimizers_idx=optim_idx,
            amp_backend=amp_backend,
        )

    def transfer_batch_to_device(self, batch: Any, dev: torch.device) -> Any:
        return self.hooks.transfer_batch_to_device(batch, dev)

    def update_pretrained_layers(self, layers: List[str]):
        self.pretrained_layers = layers

    def freeze_pretrained_layers(self, freeze: bool):
        length = sum(1 for x in self.parameters())

        for index, param in enumerate(self.parameters()):
            if self.freezing_start <= index <= length + self.freezing_stop:
                param.requires_grad = not freeze
