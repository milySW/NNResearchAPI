from __future__ import annotations

from typing import Iterable

import torch

from configs import DefaultPostprocessors


def pred_transform(preds: Iterable, postprocessors: DefaultPostprocessors):
    tfms_list = postprocessors.value_list()
    data = list([torch.zeros_like(preds), preds])

    for tfms in tfms_list:
        _, predictions = tfms(data)

    return predictions
