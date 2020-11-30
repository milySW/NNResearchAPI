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

    # binarize_labels = LabelBinarize(
    #     main_class=2, x=False, y=True, train=True, valid=True, test=True,
    # )

    # auto_covariation = AutoCov(
    #     size=64 * 1 * 1 * 1 * 1 * 1 * 1,
    #     x=True,
    #     y=False,
    #     train=True,
    #     valid=True,
    #     test=True,
    # )
