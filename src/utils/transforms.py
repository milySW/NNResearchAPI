from __future__ import annotations

from typing import Iterable

import tqdm

from configs import DefaultPostprocessors


def pred_transform(data: Iterable, postprocessors: DefaultPostprocessors):
    info = "Applying postprocessors ..."
    tfms_list = postprocessors.value_list()
    disable = len(tfms_list) == 0

    for tfms in tqdm(tfms_list, desc=info, disable=disable):
        predictions = tfms(data)

    return predictions
