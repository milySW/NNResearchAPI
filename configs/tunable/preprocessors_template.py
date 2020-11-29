from configs.base.base import BaseConfig
from src.transformations import Flatten


class DefaultPreprocessors(BaseConfig):
    """
    Config responsible for preprocessors of :class:`BaseTransformation`.
    Providing new preprocessor require adding new class field with any name
    """

    flatten = Flatten(
        start_dim=-2,
        end_dim=-1 * 1,
        x=True,
        y=False,
        train=True,
        valid=True,
        test=True,
    )
