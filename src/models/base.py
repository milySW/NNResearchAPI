import pytorch_lightning as pl
import torch

from src.optimizers import BaseOptim


class LitModel(pl.LightningModule):
    """BASE MODEL"""

    def __init__(self, *kwargs):
        self.conifg = NotImplemented
        super().__init__()

    @property
    def loss_function(self):
        return self.config.training.loss

    @property
    def metrics(self):
        return self.config.metrics.to_dict()

    @property
    def optim(self):
        return self.config.optimizers

    @property
    def model_gen(self):
        NotImplemented

    @property
    def model_disc(self):
        NotImplemented

    def forward(self, x):
        return NotImplemented

    def configure_optimizers(self) -> BaseOptim:
        r"""
        Set optimizers and learning-rate schedulers
        passed to config as DefaultOptimizer.
        Return:
            - Single optimizer.
        """
        models = dict(normal=self, gen=self.model_gen, disc=self.model_disc,)
        opts = self.optim.optimizers

        optimizers = self.optim.get_optimizers(opts.items(), models)
        schedulers = self.optim.get_schedulers(optimizers)

        return list(optimizers.values()), schedulers

    def calculate_batch(self, batch: list) -> torch.tensor:
        x, y = batch
        y_hat = self(x.float())

        labels = torch.argmax(y.squeeze(), 1)
        preds = torch.argmax(y_hat.squeeze(), 1)

        loss = self.loss_function(y_hat, labels)

        hiddens = dict(
            inputs=x.detach().cpu(),
            predictions=preds.detach().cpu(),
            targets=labels.detach().cpu(),
            loss=loss.detach().cpu().unsqueeze(dim=0),
        )

        return loss, hiddens

    def training_step(self, batch: list, batch_idx: int) -> pl.TrainResult:
        loss, hiddens = self.calculate_batch(batch)
        result = pl.TrainResult(loss)

        result.hiddens = hiddens
        return result

    def validation_step(self, batch: list, batch_idx: int) -> pl.EvalResult:
        loss, hiddens = self.calculate_batch(batch)
        result = pl.EvalResult(checkpoint_on=loss)

        result.hiddens = hiddens
        return result

    def test_step(self, batch: list, batch_idx: int) -> pl.EvalResult:
        loss, hiddens = self.calculate_batch(batch)
        result = pl.EvalResult(checkpoint_on=loss)

        result.hiddens = hiddens
        return result
