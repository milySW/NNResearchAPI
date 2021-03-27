import numpy as np
import torch

from sklearn.metrics import confusion_matrix


class Accuracy:
    extremum = "max"

    def __call__(self, preds: torch.Tensor, target: torch.Tensor):
        if len(target.unique()) == 4:
            preds = preds.argmax(dim=1).type(torch.long)
            target = target.type(torch.long)

            matrix = confusion_matrix(target, preds)
            numerator = np.sum(matrix.diagonal())

        else:
            matrix = confusion_matrix(target, preds)
            numerator = matrix[0][0] + matrix[1][1]

        return numerator / torch.tensor(matrix).sum().item()
