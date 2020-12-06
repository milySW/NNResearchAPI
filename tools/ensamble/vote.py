from typing import List

import torch


def two_classifiers(model_preds: List[torch.Tensor]) -> torch.Tensor:
    preds_1, preds_2 = model_preds

    def move_threshold(preds: torch.Tensor, threshold: float) -> torch.Tensor:
        a = torch.zeros(preds_2.shape[0], 1)
        b = torch.ones(preds_2.shape[0], 1) * threshold
        threshold_tensor = torch.cat([a, b], dim=1)

        return preds + threshold_tensor

    preds_2 = move_threshold(preds=preds_2, threshold=0.2)
    flat_preds_1, flat_preds_2 = preds_1.argmax(dim=1), preds_2.argmax(dim=1)

    counter_a = 0
    counter_b = 0
    counter_c = 0
    counter_d = 0

    for index in range(flat_preds_1.shape.numel()):
        main_choice = flat_preds_1[index]
        support_choice = flat_preds_2[index]

        if main_choice == 3 and support_choice == 1:
            # a = preds_2[index][0]
            # b = preds_2[index][1]
            a = preds_1[index][main_choice]
            b = preds_1[index].topk(k=2).values[-1]

            if a / b > 1.05:
                const = preds_2[index][1]
                preds_1[index] += torch.tensor([0, 0, const, 0])
                counter_b += 1

            counter_a += 1

        elif flat_preds_2[index] != 1 and flat_preds_1[index] == 2:
            # var = preds_1[index].max()
            # preds_1[index][2] = 0

            # max_index = preds_1[index].argmax()
            # preds_1[index][max_index] += var
            a = preds_2[index][0]
            b = preds_2[index][1]

            if a / b < 0.7:
                preds_1[index] = torch.tensor([a, a, b, a])
                counter_d += 1

            counter_c += 1
    return preds_1
