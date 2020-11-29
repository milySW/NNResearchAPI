from typing import Iterable

import torch

from configs import DefaultPostprocessors, DefaultPreprocessors


def pred_transform(preds: Iterable, postprocessors: DefaultPostprocessors):
    tfms_list = postprocessors.value_list()
    data = list([torch.zeros_like(preds), preds])

    for tfms in tfms_list:
        data = tfms(data)

    return data[1]


def input_transform(inputs: Iterable, preprocessors: DefaultPreprocessors):
    tfms_list = preprocessors.value_list()
    data = list([inputs, torch.zeros_like(inputs)])

    for tfms in tfms_list:
        data = tfms(data)

    return data[0]
