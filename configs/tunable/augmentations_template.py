from configs.base.base import BaseConfig
from src.transformations import Flip  # noqa
from src.transformations.gaussian_noise import GaussianNoise  # noqa


class DefaultAugmentations(BaseConfig):
    """
    Config responsible for transformations of :class:`BaseTransformation`.
    Providing new transforms require adding new class field with any name
    """

    # flip = Flip(x=True, y=False, dims=[1, 2], ratio=0.5)
    # gaussian_noise = GaussianNoise(x=True, y=False, mean=1, std=.7, ratio=.5)
