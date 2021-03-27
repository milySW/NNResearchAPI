import torch

from sklearn.metrics import confusion_matrix


class F1:
    def __call__(self, preds: torch.Tensor, target: torch.Tensor):
        if len(target.unique()) == 4:
            preds = preds.argmax(dim=1).type(torch.long)
            target = target.type(torch.long)

            matrix = confusion_matrix(target, preds)

            prec_1 = matrix[0][0] / (
                matrix[0][0] + matrix[0][1] + matrix[0][2] + matrix[0][3]
            )
            prec_2 = matrix[1][1] / (
                matrix[1][0] + matrix[1][1] + matrix[1][2] + matrix[1][3]
            )
            prec_3 = matrix[2][2] / (
                matrix[2][0] + matrix[2][1] + matrix[2][2] + matrix[2][3]
            )
            prec_4 = matrix[3][3] / (
                matrix[3][0] + matrix[3][1] + matrix[3][2] + matrix[3][3]
            )

            prec = (prec_1 + prec_2 + prec_3 + prec_4) / 4

            recall_1 = matrix[0][0] / (
                matrix[0][0] + matrix[1][0] + matrix[2][0] + matrix[3][0]
            )
            recall_2 = matrix[1][1] / (
                matrix[0][1] + matrix[1][1] + matrix[2][1] + matrix[3][1]
            )
            recall_3 = matrix[2][2] / (
                matrix[0][2] + matrix[1][2] + matrix[2][2] + matrix[3][2]
            )
            recall_4 = matrix[3][3] / (
                matrix[0][3] + matrix[1][3] + matrix[2][3] + matrix[3][3]
            )

            recall = (recall_1 + recall_2 + recall_3 + recall_4) / 4

        else:
            matrix = confusion_matrix(target, preds)
            prec = matrix[0][0] / (matrix[0][0] + matrix[0][1])
            recall = matrix[0][0] / (matrix[0][0] + matrix[1][0])

        return (2 * prec * recall) / (prec + recall)
