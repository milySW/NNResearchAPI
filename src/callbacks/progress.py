import sys

from pytorch_lightning.callbacks import progress

from src.base.callback import BaseCallback


class LightProgressBar(BaseCallback, progress.ProgressBar):
    """
    Callback logging progresss bar. Wrapper for progress.ProgressBar,
    which provides progress bar hidding after operation end.

    Output:

        - `metrics_all.csv`: Metrics calculated for every epoch
        - `metrics_last.csv`: Final metrics

    """

    def init_train_tqdm(self) -> progress.tqdm:
        bar = progress.tqdm(
            desc="Training",
            initial=self.train_batch_idx,
            position=(2 * self.process_position) + 1,
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )
        return bar

    def init_test_tqdm(self) -> progress.tqdm:
        """ Override this to customize the tqdm bar for testing. """
        bar = progress.tqdm(
            desc="Testing",
            position=(2 * self.process_position) + 1,
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar
