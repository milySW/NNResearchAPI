from configs.base.base import BaseConfig
from src.transformations import Flip


class DefaultTransformations(BaseConfig):
    """
    Config responsible for transformations of :class:`BaseTransformation`.
    Providing new transforms require adding new class field field with any name
    """

    flip = Flip(x=True, y=False, dims=[1, 2])
