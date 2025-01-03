import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import Module, Linear

from .._reshape_last_dim import reshape_last_dim


class MixtureDensityNetwork(Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        components: int,
    ):
        super().__init__()

        self.components = components

        self.fc_pi = Linear(input_dimension, components)

        self.fc_mu = Linear(input_dimension, components * output_dimension)

        self.fc_sigma = Linear(input_dimension, components * output_dimension)

    def forward(self, input: Tensor) -> (Tensor, Tensor, Tensor):
        mixing_coefficients = torch.nn.functional.softmax(self.fc_pi(input), -1)

        means = reshape_last_dim(self.fc_mu(input), self.components, -1)

        variances = reshape_last_dim(torch.exp(self.fc_sigma(input)), self.components, -1)

        return mixing_coefficients, means, variances

    def sample(self, input: Tensor, argmax: bool = False) -> Tensor:
        mixing_coefficients, means, sigma = self.forward(input)

        if argmax:
            pis = torch.argmax(mixing_coefficients, -1).unsqueeze(-1).unsqueeze(-1)
        else:
            pis = Categorical(mixing_coefficients).sample().unsqueeze(-1).unsqueeze(-1)

        samples = torch.gather(
            means,
            dim=-2,
            index=pis.repeat(1, 1, 1, means.size(-1)),
        )

        if not argmax:
            samples = samples + torch.randn_like(samples) * torch.gather(
                sigma, dim=-2, index=pis.repeat(1, 1, 1, sigma.size(-1))
            )

        return torch.squeeze(samples, dim=-2)
