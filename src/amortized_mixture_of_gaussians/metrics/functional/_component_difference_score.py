import torch
from torch import Tensor


def component_difference_score(prediction: Tensor, target: Tensor) -> Tensor:
    score = torch.sum(prediction, dim=1) == target

    score = score.to(dtype=torch.get_default_dtype())

    return torch.mean(score)
