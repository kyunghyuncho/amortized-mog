import scipy.optimize
import torch
from torch import Tensor


def mean_distance_score(
    predicted_existences: Tensor,
    predicted_means: Tensor,
    means,
    components,
) -> (Tensor, Tensor):
    predicted_components = torch.sum(
        predicted_existences,
        dim=1,
    )

    mean_distance = 0

    for i in range(predicted_means.shape[0]):
        pred_means_i = predicted_means[i, : int(predicted_components[i].item()), :]

        true_means_i = means[i, : components[i], :]

        if pred_means_i.shape[0] == 0 or true_means_i.shape[0] == 0:
            continue

        distances = torch.cdist(pred_means_i, true_means_i)

        distances = distances.detach().cpu().numpy()

        row_ind, col_ind = scipy.optimize.linear_sum_assignment(distances)

        mean_distance = mean_distance + distances[row_ind, col_ind].mean()

    if predicted_means.shape[0] > 0:
        mean_distance = mean_distance / predicted_means.shape[0]
    else:
        mean_distance = torch.tensor(0.0)

    return mean_distance
