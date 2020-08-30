import pytorch_lightning as pl
import torch


class LitModel(pl.LightningModule):
    def __init__(self, *kwargs):
        super().__init__()

    @property
    def loss_function(self):
        return NotImplemented

    @property
    def metrics(self):
        return NotImplemented

    def forward(self, x):
        return NotImplemented

    def configure_optimizers(self):
        return NotImplemented

    def calculate_batch(self, batch: list) -> torch.tensor:
        x, y = batch
        y_hat = self(x.float())

        label = torch.argmax(y.squeeze(), 1)
        loss = self.loss_function(y_hat, label)

        hiddens = dict(
            inputs=x.detach().cpu(),
            predictions=y_hat.detach().cpu(),
            targets=label.detach().cpu(),
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
