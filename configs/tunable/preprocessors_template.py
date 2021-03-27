from configs.base.base import BaseConfig
from src.transformations import (  # noqa
    AutoCov,
    ConstantChannel,
    Diff,
    Flatten,
    Histogramize,
    LabelBinarize,
    Permute,
    RemoveClasses,
)
from src.utils.features import efficiency2d  # noqa


class DefaultPreprocessors(BaseConfig):
    """
    Config responsible for preprocessors of :class:`BaseTransformation`.
    Providing new preprocessor require adding new class field with any name
    """

    permute = Permute(
        dims=(0, 1, 3, 2), x=True, y=False, train=True, valid=True, test=True,
    )

    # constant_channel = ConstantChannel(
    #     func=efficiency2d,
    #     new_channel=True,
    #     x=True,
    #     y=False,
    #     train=True,
    #     valid=True,
    #     test=True,
    # )

    # flatten = Flatten(
    #     start_dim=-2,
    #     end_dim=-1 * 1,
    #     x=True,
    #     y=False,
    #     train=True,
    #     valid=True,
    #     test=True,
    # )

    # remove_classes = RemoveClasses(
    #     classes=[0, 1 * 1],
    #     x=True,
    #     y=True,
    #     train=True,
    #     valid=True,
    #     test=True,
    #     both=True,
    # )

    # histogramize = Histogramize(
    #     bins=200 * 1,
    #     new_channel=True,
    #     x=True,
    #     y=False,
    #     train=True,
    #     valid=True,
    #     test=True,
    # )

    # diff = Diff(
    #     lag=128 * 1 * 1,
    #     new_channel=True,
    #     x=True,
    #     y=False,
    #     train=True,
    #     valid=True,
    #     test=True,
    # )

    # binarize_labels = LabelBinarize(
    #     main_class=2, x=False, y=True, train=True, valid=True, test=True,
    # )

    # auto_covariation_1 = AutoCov(
    #     size=8 * 1 * 1 * 1 * 1 * 1 * 1,
    #     x=True,
    #     y=False,
    #     train=True,
    #     valid=True,
    #     test=True,
    # )

    # auto_covariation_2 = AutoCov(
    #     size=16 * 1 * 1 * 1 * 1 * 1 * 1,
    #     x=True,
    #     y=False,
    #     train=True,
    #     valid=True,
    #     test=True,
    # )

    # auto_covariation_3 = AutoCov(
    #     size=24 * 1 * 1 * 1 * 1 * 1 * 1,
    #     x=True,
    #     y=False,
    #     train=True,
    #     valid=True,
    #     test=True,
    # )
