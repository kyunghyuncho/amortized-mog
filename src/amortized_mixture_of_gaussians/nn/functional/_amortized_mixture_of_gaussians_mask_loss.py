import torch
from torch import Tensor


def amortized_mixture_of_gaussians_mask_loss(input: Tensor, target: Tensor) -> Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(
        torch.squeeze(
            target[:, : input.shape[1], :],
            dim=-1,
        ),
        input,
    )
