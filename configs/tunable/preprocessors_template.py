from configs.base.base import BaseConfig


class DefaultPreprocessors(BaseConfig):
    """
    Config responsible for preprocessors of :class:`BaseTransformation`.
    Providing new preprocessor require adding new class field with any name
    """

    # flip = Flip(x=True, y=False, dims=[1, 2])
