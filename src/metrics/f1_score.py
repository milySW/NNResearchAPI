import torch

from sklearn.metrics import confusion_matrix


class F1:
    def __call__(self, preds: torch.Tensor, target: torch.Tensor):
        if len(target.unique()) == 4:
            preds = preds.argmax(dim=1).type(torch.long)
            target = target.type(torch.long)

            matrix = confusion_matrix(target, preds)

            recall_1 = matrix[0][0] / (
                matrix[0][0] + matrix[0][1] + matrix[0][2] + matrix[0][3]
            )
            recall_2 = matrix[1][1] / (
                matrix[1][0] + matrix[1][1] + matrix[1][2] + matrix[1][3]
            )
            recall_3 = matrix[2][2] / (
                matrix[2][0] + matrix[2][1] + matrix[2][2] + matrix[2][3]
            )
            recall_4 = matrix[3][3] / (
                matrix[3][0] + matrix[3][1] + matrix[3][2] + matrix[3][3]
            )

            recall = (recall_1 + recall_2 + recall_3 + recall_4) / 4

            precision_1 = matrix[0][0] / (
                matrix[0][0] + matrix[1][0] + matrix[2][0] + matrix[3][0]
            )
            precision_2 = matrix[1][1] / (
                matrix[0][1] + matrix[1][1] + matrix[2][1] + matrix[3][1]
            )
            precision_3 = matrix[2][2] / (
                matrix[0][2] + matrix[1][2] + matrix[2][2] + matrix[3][2]
            )
            precision_4 = matrix[3][3] / (
                matrix[0][3] + matrix[1][3] + matrix[2][3] + matrix[3][3]
            )

            precision = precision_1 + precision_2 + precision_3 + precision_4
            precision /= 4

        else:
            matrix = confusion_matrix(target, preds)
            precision = matrix[1][1] / (matrix[0][1] + matrix[1][1])
            recall = matrix[1][1] / (matrix[1][0] + matrix[1][1])

        return (2 * precision * recall) / (precision + recall)
