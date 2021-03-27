import math

import numpy as np
import torch

from scipy import spatial


def asymmetry2d(data: torch.Tensor) -> float:
    x, y = data

    N = x.shape.numel()

    var_1 = 1 / N * ((x - x.mean()) ** 2).sum()
    var_2 = 1 / N * ((x - x.mean()) * (y - y.mean())).sum()
    var_3 = var_2
    var_4 = 1 / N * ((y - y.mean()) ** 2).sum()

    b = var_1 + var_4
    delta = b ** 2 - 4 * (var_1 * var_2 - var_2 * var_3)

    lambda_1 = (-b + delta ** (1 / 2)) / 2
    lambda_2 = (-b - delta ** (1 / 2)) / 2

    nominator = (lambda_1 - lambda_2) ** 2
    denominator = 2 * (lambda_1 + lambda_2)

    return -np.log(1 - (nominator / denominator))


def distances(data: torch.Tensor, lag: int = 1) -> float:
    x, y = data
    result = torch.zeros_like(x)

    for index in range(x.shape.numel() - lag):
        a = x[index + lag] - x[index]
        b = y[index + lag] - y[index]

        result[index] = (a ** 2 + b ** 2) ** (1 / 2)

    return result


def largest_distance(data: torch.Tensor, patience: int = 1000) -> float:
    for _ in range(patience):
        points = data.T.numpy()

        # points which are fruthest apart will occur as vertices of convex hull
        candidates = points[spatial.ConvexHull(points).vertices]

        # get distances between each pair of candidate points
        dist_mat = spatial.distance_matrix(candidates, candidates)

        # get indices of candidates that are furthest apart
        i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)

        x, y = torch.tensor(candidates[i]), torch.tensor(candidates[j])
        result = distances([x, y]).sum().item()

        if not math.isnan(result) and result > 0:
            return result

    raise ValueError(f"Maxiumum patience - {patience} was exceeded")


def fractal_dimension2d(data: torch.Tensor) -> float:
    x, y = data

    N = x.shape.numel()
    L = distances([x, y]).sum().item()
    d = largest_distance(data)

    nominator = np.log(N)
    denominator = np.log(N * d * L ** (-1))

    return nominator / denominator


def emsd2d(data: torch.Tensor, lag: int) -> float:
    x, y = data
    N = x.shape[-1]
    r = (x[lag:N] - x[: N - lag]) ** 2 + (y[lag:N] - y[: N - lag]) ** 2

    return r.mean()


def efficiency2d(data: torch.Tensor) -> float:
    N = data.shape[-1]

    nominator = emsd2d(data, lag=N - 1)
    denominator = emsd2d(data, lag=1)

    return nominator / denominator
