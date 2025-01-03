import torch
from torch import Tensor
from torch.distributions import Normal


def amortized_mixture_of_gaussians_loss(
        input: Tensor,
        target: Tensor,
        mask: Tensor,
) -> Tensor:
    output = 0

    m = target[0].shape[0]
    n = target[0].shape[1]

    for i in range(m):
        for j in range(n):
            if mask[i, j] == 0:
                continue

            x = input[i, j, :]

            if len(x.shape) == 1:
                x = torch.unsqueeze(x, dim=0)

            output = torch.add(
                output,
                torch.squeeze(
                    torch.negative(
                        torch.sum(
                            torch.logsumexp(
                                torch.add(
                                    torch.unsqueeze(
                                        torch.unsqueeze(
                                            torch.log(target[0][i, j]),
                                            dim=0,
                                        ),
                                        dim=-1,
                                    ),
                                    Normal(
                                        target[1][i, j],
                                        target[2][i, j],
                                    ).log_prob(
                                        torch.unsqueeze(
                                            x,
                                            dim=1,
                                        ),
                                    ),
                                ),
                                dim=1,
                            ),
                            dim=-1,
                        ),
                    ),
                ),
            )

    if m > 0:
        output = output / m

    return output
