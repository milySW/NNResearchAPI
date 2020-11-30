from typing import Iterable, Optional

import torch

from configs import DefaultPostprocessors, DefaultPreprocessors
from src.utils.collections import collection_is_none


def pred_transform(preds: Iterable, postprocessors: DefaultPostprocessors):
    tfms_list = postprocessors.value_list()
    data = list([torch.zeros_like(preds), preds])

    for tfms in tfms_list:
        data = tfms(data)

    return data[1]


def input_transform(
    input_data: Optional[Iterable],
    input_labels: Optional[Iterable],
    preprocessors: DefaultPreprocessors,
):
    if collection_is_none(input_data) and collection_is_none(input_labels):
        raise ValueError("Both data inputs are None!")

    tfms_list = preprocessors.value_list()
    data = list([input_data, input_labels])

    for tfms in tfms_list:
        data = tfms(data)

    return data
