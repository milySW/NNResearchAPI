from configs.base.base import BaseConfig


class DefaultPostprocessors(BaseConfig):
    """
    Config responsible for postprocessors of :class:`BaseTransformation`.
    Providing new postprocessor require adding new class field with any name
    """

    # argmax = ArgMax(x=False, y=True, dim=1, keepdim=True)
