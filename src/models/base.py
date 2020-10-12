from typing import Any, Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch

from pytorch_lightning.utilities import AMPType
from torch.utils.data.dataloader import DataLoader

from src.losses import BaseLoss
from src.metrics import BaseMetric
from src.optimizers import BaseOptim
from src.optimizers.schedulers import BaseScheduler


class LitModel(pl.LightningModule):
    def __init__(self, *kwargs):
        self.conifg = NotImplemented
        super().__init__()

    @property
    def loss_function(self) -> BaseMetric:
        return self.config.training.loss

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
    def model_gen(self):
        return NotImplemented

    @property
    def model_disc(self):
        return NotImplemented

    def forward(self, x: torch.Tensor):
        return NotImplemented

    def set_example(self, train_loader: DataLoader, dtype=torch.float32):
        example = train_loader.dataset[0][0][None]
        self.example_input_array = torch.tensor(example, dtype=dtype)

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

    def standard_calculate_batch(
        self, batch: list, precalculated_preds: torch.Tensor = None
    ) -> torch.Tensor:

        x, y = batch
        if precalculated_preds is not None:
            y_hat = precalculated_preds
        else:
            y_hat = self(x.float())

        labels = torch.argmax(y.squeeze(), 1)
        preds = torch.argmax(y_hat.squeeze(), 1)

        loss = self.loss_function(y_hat, labels)

        calculations = dict(
            inputs=x.detach().cpu(),
            preds=preds.detach().cpu(),
            labels=labels.detach().cpu(),
            losses=loss.detach().cpu().unsqueeze(dim=0),
        )

        return loss, calculations

    def calculate_batch(self, batch: list, step: str) -> torch.Tensor:

        if output := self.hooks.calculate_batch(batch=batch, step=step):
            return output
        else:
            return self.standard_calculate_batch(batch=batch)

    def training_step(self, batch: list, batch_idx: int) -> pl.TrainResult:
        loss, calculations = self.calculate_batch(batch, step="train")
        result = pl.TrainResult(loss)

        self.trainer.calculations = calculations
        return result

    def validation_step(self, batch: list, batch_idx: int) -> pl.EvalResult:
        loss, calculations = self.calculate_batch(batch, step="validation")
        result = pl.EvalResult(checkpoint_on=loss)

        self.trainer.calculations = calculations
        return result

    def test_step(self, batch: list, batch_idx: int) -> pl.EvalResult:
        loss, calculations = self.calculate_batch(batch, step="test")
        result = pl.EvalResult(checkpoint_on=loss)

        self.trainer.calculations = calculations
        return result

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
        self, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        self.hooks.on_train_batch_end(batch, batch_idx, dataloader_idx)

    def on_validation_batch_start(
        self, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        self.hooks.on_validation_batch_start(batch, batch_idx, dataloader_idx)

    def on_validation_batch_end(
        self, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        self.hooks.on_validation_batch_end(batch, batch_idx, dataloader_idx)

    def on_test_batch_start(
        self, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        self.hooks.on_test_batch_start(batch, batch_idx, dataloader_idx)

    def on_test_batch_end(
        self, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        self.hooks.on_test_batch_end(batch, batch_idx, dataloader_idx)

    def on_epoch_start(self) -> None:
        self.hooks.on_epoch_start()

    def on_epoch_end(self) -> None:
        self.hooks.on_epoch_end()

    def on_train_epoch_start(self) -> None:
        self.hooks.on_train_epoch_start()

    def on_train_epoch_end(self) -> None:
        self.hooks.on_train_epoch_end

    def on_validation_epoch_start(self) -> None:
        self.hooks.on_validation_epoch_start

    def on_validation_epoch_end(self) -> None:
        self.hooks.on_validation_epoch_end

    def on_test_epoch_start(self) -> None:
        self.hooks.on_test_epoch_start()

    def on_test_epoch_end(self) -> None:
        self.hooks.on_test_epoch_end

    def on_pre_performance_check(self) -> None:
        self.hooks.on_pre_performance_check()

    def on_post_performance_check(self) -> None:
        self.hooks.on_post_performance_check()

    def on_before_zero_grad(self, optim: BaseOptim) -> None:
        self.hooks.on_before_zero_grad(optim)

    def on_after_backward(self) -> None:
        self.hooks.on_after_backward()

    def backward(
        self, trainer, loss: torch.Tensor, optim: BaseOptim, optim_idx: int
    ) -> None:
        self.hooks.backward(trainer, loss, optim, optim_idx)

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
