from configs.base.base import BaseConfig
from src.hooks import SelectiveBackprop


class DefaultHooks(BaseConfig):
    """
    Config responsible for passing hooks of :class:`BaseHook`.
    Providing new hooks require adding new class field field with any name.
    Functions implemented inside provided hooks will combine with each other
    depending on order. Finally, result functions will overwrite
    original hooks of `pytorch_lightning.LightningModule` class.

    Warning:
        Since hook functions bind to each other and return a cumulative
        function, the objects provided in the configuration cannot together
        exceed the number of one implementations per functions in this set:
        [`backward`, `amp_scale_loss`, `transfer_batch_to_device`].
        The main reason is that each of these functions provides
        certain behavior or returns certain data, which makes them unbindable.
    """

    selective_backprop = SelectiveBackprop
