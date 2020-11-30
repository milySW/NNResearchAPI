from configs.base.base import BaseConfig

# from src.transformations import Flip


class DefaultAugmentations(BaseConfig):
    """
    Config responsible for transformations of :class:`BaseTransformation`.
    Providing new transforms require adding new class field with any name
    """

    # flip = Flip(x=True, y=False, dims=[1, 2], ratio=0.5)
