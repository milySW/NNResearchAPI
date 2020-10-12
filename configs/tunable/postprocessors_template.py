from configs.base.base import BaseConfig
from src.transformations import Flip


class DefaultPostprocessors(BaseConfig):
    """
    Config responsible for postprocessors of :class:`BaseTransformation`.
    Providing new postprocessor require adding new class field with any name
    """

    flip = Flip(x=True, y=False, dims=[1, 2])
