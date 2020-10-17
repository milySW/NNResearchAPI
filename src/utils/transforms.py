from typing import Iterable

import torch

from configs import DefaultPostprocessors


def pred_transform(preds: Iterable, postprocessors: DefaultPostprocessors):
    tfms_list = postprocessors.value_list()
    data = list([torch.zeros_like(preds), preds])

    for tfms in tfms_list:
        data = tfms(data)

    return data[1]
